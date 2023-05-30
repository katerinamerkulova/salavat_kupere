import os
import json
import random
import re
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
from tqdm import tqdm

sw = nltk.download("stopwords")

# Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")

colors = open("../files/colors.txt").read().split("\n")
colors = {color: i for i, color in enumerate(colors, start=1)}
colors[None] = 0
colors_mapping = json.load(open("../files/colors_mapping.json"))


def process_char(name, chars):
    if chars is None:
       return str(name)
    try:
        char = " ".join(sorted([f"[CHAR]{char}[VAL]{val}" for char, val in json.loads(chars).items()]))
        res = f"{name} {char}"
    except:
        res = f"{name} {chars}"
    return res


def make_characteristics_df(df, characteristics_names):
    chars_df = [dict() for i in range(df.shape[0] + 1)]
    for i, pair in tqdm(df.iterrows(), total=df.shape[0]):
        pair = pair.to_dict()
        for char_name in characteristics_names:
            value1 = pair[f"{char_name}1"]
            value2 = pair[f"{char_name}2"]
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
            chars_df[i][f"score_{char_name}"] = np.log(score + 0.000000000001)
    chars_df = pd.DataFrame(chars_df)
    chars_df.fillna(np.log(0.000000000001), inplace=True)
    return chars_df


def process_characteristics(etl):
    characteristics = etl[["characteristic_attributes_mapping", "cat3_grouped"]].progress_apply(
        lambda x: process_characteristics_map(x[0], x[1]), axis=1)
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
        value = " ;".join(value)
        try:
            value = float(value)
        except ValueError:
            pass
        new_d[f"{category}_{key}"] = value
        new_d[key] = value
    return new_d


def process_colors(colors1, colors2):
    is_same_color = 1
    new_colors1, new_colors2 = set(), set()
    partial_ratio = 0
    if colors1 is not None:
        new_colors1 = set([colors_mapping.get(color, color) for color in colors1])
    if colors2 is not None:
        new_colors2 = set([colors_mapping.get(color, color) for color in colors2])
    if new_colors1 and new_colors2:
        partial_ratio = fuzz.partial_ratio(" ".join(sorted(new_colors1)), " ".join(sorted(new_colors2)))
        if not new_colors1.intersection(new_colors2):
            is_same_color = 0
        else:
            is_same_color = -1
    return (is_same_color, partial_ratio)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        
        # reverse the lists so that x"s are in ascending order (left to right)
        y = y[::-1]
        x = x[::-1]
        
        # get indices for x-coordinate (recall) where y-coordinate (precision) 
        # is higher than precision level (75% for our task)
        good_idx = np.where(y >= prec_level)[0]
        
        # if there are more than one such x"s (at least one is always there, 
        # it"s x=0 (recall=0)) we get a grid from x=0, to the rightest x 
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
    return {"pr_auc": pr_auc,
            "categories": unique_cats,
            "weights": weights,
            "pr_aucs": pr_aucs}


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


def matching_numbers(name1: str, name2: str):
    name1_numbers = set(re.findall(r"[0-9]+", name1))
    name2_numbers = set(re.findall(r"[0-9]+", name2))    
    union = name1_numbers.union(name2_numbers)
    intersection = name1_numbers.intersection(name2_numbers)

    if len(name1_numbers)==0 and len(name2_numbers) == 0:
        return 1
    return len(intersection) / len(union)
