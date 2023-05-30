import os
import json
import random
import re
import warnings
from collections import defaultdict
from functools import partial
from string import punctuation
from typing import List

import nltk
import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from pymystem3 import Mystem
from scipy.spatial.distance import cosine, euclidean, jensenshannon, minkowski, sqeuclidean
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_random_state, column_or_1d
from tqdm import tqdm

sw = nltk.download("stopwords")

colors = open('../files/colors.txt').read().split('\n')
colors = {color: i for i, color in enumerate(colors, start=1)}
colors[None] = 0
colors_mapping = json.load(open('../files/colors_mapping.json'))


def process_char(name, chars):
    if chars is None:
       return str(name)
    try:
        char = ' '.join(sorted([f'[CHAR]{char}[VAL]{val}' for char, val in json.loads(chars).items()]))
        res = f'{name} {char}'
    except:
        res = f'{name} {chars}'
    return res


def postprocess(score, cat):
    if cat in ['Гаджет', 'Графический планшет']:
        return max(score / 2, 0.003)
    return score


def make_characteristics_df(df, characteristics_names):
    chars_df = [dict() for i in range(df.shape[0] + 1)]
    for i, pair in tqdm(df.iterrows(), total=df.shape[0]):
        pair = pair.to_dict()
        for char_name in characteristics_names:
            value1 = pair[f'{char_name}1']
            value2 = pair[f'{char_name}2']
            if pd.isna(value1) and pd.isna(value2):
                continue
            if pd.isna(value1) or pd.isna(value2):
                score = 50
            if isinstance(value1, float) and isinstance(value2, float):
                if value1 == value2:
                    score = 100
                elif 0 < value1 < 1 and 0 < value2 < 1:
                    score = abs(value1 - value2) * 100
                else:
                    score = abs(value1 - value2)
            else:
                score = fuzz.token_set_ratio(str(value1), str(value2))
            chars_df[i][f'score_{char_name}'] = np.log(score + 0.000000000001)
    chars_df = pd.DataFrame(chars_df)
    chars_df.fillna(np.log(0.000000000001), inplace=True)
    return chars_df


def process_characteristics(etl):
    characteristics = etl[['characteristic_attributes_mapping', 'cat3_grouped']].progress_apply(lambda x: process_characteristics_map(x[0], x[1]), axis=1)
    characteristics = pd.json_normalize(characteristics)
    characteristics_names = characteristics.columns.to_list()
    etl = pd.merge(etl, characteristics, left_index=True, right_index=True)
    return etl, characteristics_names


def process_characteristics_map(chars, category):
    if chars is None:
        return {}
    d = json.loads(chars)
    new_d = {}
    for key, value in d.items():
        value = ' ;'.join(value)
        try:
            value = float(value)
        except ValueError:
            pass
        new_d[f'{category}_{key}'] = value
        new_d[key] = value
    return new_d


def process_colors(colors1, colors2):
    is_same_color = 1
    new_colors1, new_colors2 = set(), set()
    # encoded_colors1, encoded_colors2 = [], []
    partial_ratio = 0
    if colors1 is not None:
        new_colors1 = set([colors_mapping.get(color, color) for color in colors1])
        # encoded_colors1 = sorted([colors[color] for color in new_colors1])
    if colors2 is not None:
        new_colors2 = set([colors_mapping.get(color, color) for color in colors2])
        # encoded_colors2 = sorted([colors[color] for color in new_colors2])
    if new_colors1 and new_colors2:
        partial_ratio = fuzz.partial_ratio(' '.join(sorted(new_colors1)), ' '.join(sorted(new_colors2)))
        if not new_colors1.intersection(new_colors2):
            is_same_color = 0
        else:
            is_same_color = -1
    return (is_same_color, 
            # encoded_colors1, encoded_colors2,
            partial_ratio)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pr_auc_macro(
    target_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped"
) -> dict:

    df = target_df.merge(predictions_df, on=["variantid1", "variantid2"])

    y_true = df["target"]
    y_pred = df["scores"]
    categories = df[cat_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    # calculate metric for each big category
    for i, category in enumerate(unique_cats):
        # take just a certain category
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        # if there is no matches in the category then PRAUC=0
        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # get coordinates (x, y) for (recall, precision) of PR-curve
        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        
        # reverse the lists so that x's are in ascending order (left to right)
        y = y[::-1]
        x = x[::-1]
        
        # get indices for x-coordinate (recall) where y-coordinate (precision) 
        # is higher than precision level (75% for our task)
        good_idx = np.where(y >= prec_level)[0]
        
        # if there are more than one such x's (at least one is always there, 
        # it's x=0 (recall=0)) we get a grid from x=0, to the rightest x 
        # with acceptable precision
        if len(good_idx) > 1:
            gt_prec_level_idx = np.arange(0, good_idx[-1] + 1)
        # if there is only one such x, then we have zeros in the top scores 
        # and the curve simply goes down sharply at x=0 and does not rise 
        # above the required precision: PRAUC=0
        else:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # calculate category weight anyway
        weights.append(counts[i] / len(categories))
        # calculate PRAUC for all points where the rightest x 
        # still has required precision 
        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
        except ValueError:
            pr_aucs.append(0)
    
    pr_auc = np.average(pr_aucs, weights=weights)
    return {'pr_auc': pr_auc,
            'categories': unique_cats,
            'weights': weights,
            'pr_aucs': pr_aucs}


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.
    This cross-validation object is a variation of StratifiedKFold attempts to
    return stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class as much as possible given the
    constraint of non-overlapping groups between splits.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [1 1 2 2 4 5 5 5 5 8 8]
           [0 0 1 1 1 0 0 0 0 0 0]
     TEST: [3 3 3 6 6 7]
           [1 1 1 0 0 0]
    TRAIN: [3 3 3 4 5 5 5 5 6 6 7]
           [1 1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 2 2 8 8]
           [0 0 1 1 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]
    Notes
    -----
    The implementation is designed to:
    * Mimic the behavior of StratifiedKFold as much as possible for trivial
      groups (e.g. when each group contains only one sample).
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Stratify based on samples as much as possible while keeping
      non-overlapping groups constraint. That means that in some cases when
      there is a small number of groups containing a large number of samples
      the stratification will not be possible and the behavior will be close
      to GroupKFold.
    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (n_smallest_class, self.n_splits)), UserWarning)
        n_classes = len(y_cnt)
        
        
        _, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True)
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            rng.shuffle(y_counts_per_group)

        # Stable sort to keep shuffled order for groups with the same
        # class distribution variance
        sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1),
                                       kind='mergesort')

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold, y_cnt=y_cnt,
                group_y_counts=group_y_counts)
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group_idx in enumerate(groups_inv)
                            if group_idx in groups_per_fold[i]]
            yield test_indices

    def _find_best_fold(
            self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(
                y_counts_per_fold / y_cnt.reshape(1, -1),
                axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval or
                np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold


def get_pic_features(main_pic_embeddings_1,
                     main_pic_embeddings_2,
                     percentiles: List[int]):
    """Calculate distances percentiles for 
    pairwise pic distances. Percentiles are useful 
    when product has several pictures.
    """
    
    if main_pic_embeddings_1 is not None and main_pic_embeddings_2 is not None:
        main_pic_embeddings_1 = np.array([x for x in main_pic_embeddings_1])
        main_pic_embeddings_2 = np.array([x for x in main_pic_embeddings_2])
        
        dist_m = pairwise_distances(
            main_pic_embeddings_1, main_pic_embeddings_2
        )
    else:
        dist_m = np.array([[-1]])

    pair_features = []
    pair_features += np.percentile(dist_m, percentiles).tolist()

    return pair_features


def text_dense_distances(ozon_embedding, comp_embedding):
    """Calculate Euclidean and Cosine distances between
    ozon_embedding and comp_embedding.
    """
    pair_features = []
    if ozon_embedding is None or comp_embedding is None:
        pair_features = [-1, -1]
    elif len(ozon_embedding) == 0 or len(comp_embedding) == 0:
        pair_features = [-1, -1]
    else:
        pair_features.append(
            euclidean(ozon_embedding, comp_embedding)
        )
        
        cosine_value = cosine(ozon_embedding, comp_embedding)
        jensenshannon_value = jensenshannon(ozon_embedding, comp_embedding)
        minkowski_value = minkowski(ozon_embedding, comp_embedding)
        sqeuclidean_value = sqeuclidean(ozon_embedding, comp_embedding)

        pair_features.append(cosine_value)
        pair_features.append(jensenshannon_value)
        pair_features.append(minkowski_value)
        pair_features.append(sqeuclidean_value)
    return pair_features


# from sentence_transformers import SentenceTransformer
# STRANSFORMERS = {
#     'sentence-transformers/paraphrase-mpnet-base-v2': ('mpnet', 768),
#     'sentence-transformers/bert-base-wikipedia-sections-mean-tokens': ('wikipedia', 768)
# }
# def get_encode(df, encoder, name):    
#     device = torch.device(
#         "cuda:0" if torch.cuda.is_available() else "cpu")

#     model = SentenceTransformer(
#         encoder, 
#         cache_folder=f'./hf_{name}/'
#     )
#     model.to(device)
#     model.eval()
#     return np.array(model.encode(df['excerpt']))
# def get_embeddings(df, emb=None, tolist=True):
    
#     ret = pd.DataFrame(index=df.index)
    
#     for e, s in STRANSFORMERS.items():
#         if emb and s[0] not in emb:
#             continue
        
#         ret[s[0]] = list(get_encode(df, e, s[0]))
#         if tolist:
#             ret = pd.concat(
#                 [ret, pd.DataFrame(
#                     ret[s[0]].tolist(),
#                     columns=[f'{s[0]}_{x}' for x in range(s[1])],
#                     index=ret.index)],
#                 axis=1, copy=False, sort=False)
    
#     return ret


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


# Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")


# Preprocess function
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    return text


get_pic_features_func = partial(
    get_pic_features,
    percentiles=[0, 25, 50]
)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_model(text, model, tokenizer, max_length, device):
    model.to(device)
    # use padding, truncation of long sequences and return pytorch tensors
    t = tokenizer(text, padding=True, truncation=True,
                  max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}

    with torch.no_grad():
        # move all tensors on the same device as model
        model_output = model(**t)
    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, t['attention_mask'])
    return sentence_embeddings.cpu().numpy()


def labse_encode(text, model, tokenizer, max_length, device):
    model.to(device)
    encoded_input = tokenizer(text, 
                              padding=True,
                              truncation=True,
                              max_length=max_length,
                              return_tensors='pt')
    t = {k: v.to(model.device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**t)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()


def matching_numbers(name1: str, name2: str):
    name1_numbers = set(re.findall(r'[0-9]+', name1))
    name2_numbers = set(re.findall(r'[0-9]+', name2))    
    union = name1_numbers.union(name2_numbers)
    intersection = name1_numbers.intersection(name2_numbers)

    if len(name1_numbers)==0 and len(name2_numbers) == 0:
        return 1
    return len(intersection) / len(union)
