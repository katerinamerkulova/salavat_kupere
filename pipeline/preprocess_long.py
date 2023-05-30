import json
import time
import warnings

import jellyfish as jf
import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import embed_model, get_pic_features_func, matching_numbers, labse_encode, preprocess_text, process_char, text_dense_distances


# ignore warnings
warnings.filterwarnings("ignore")
tqdm.pandas()


# LaBSEtokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
# LaBSE_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

# RuBERTtokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
# RuBERT_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('Current device:', device)

minilm_name = SentenceTransformer('/home/merkulova/hackathon/output/all-MiniLM-L6-v2-2023-05-20_17-54-44', device=0)
minilm_name_cat = SentenceTransformer('/home/merkulova/hackathon/hackathon_master_kate/chars_names_minilm/all-MiniLM-L6-v2-2023-05-26_21-54-53', device=0)


def etl_data(train=True):
    start_time = time.time()
    print(f'Start preprocessing dataframe with train={train}' )
    if train:
        dataset = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/train_pairs.parquet")
        etl = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/train_data.parquet")
    else:
        dataset = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/test_pairs_wo_target.parquet")
        etl = pd.read_parquet("../../data/hackathon_files_for_participants_ozon/test_data.parquet")
    # etl['name_rubert_tiny'] = etl['name'].progress_apply(lambda text: embed_model(text, LaBSE_model, LaBSEtokenizer, 64, device=device))
    # etl['name_labse'] = etl['name'].progress_apply(lambda text: labse_encode(text, LaBSE_model, LaBSEtokenizer, 64, device=device))
    etl['name_minilm'] = etl['name'].progress_apply(lambda text: minilm_name.encode([text])[0])
    etl['name_cat'] = etl[['name', 'characteristic_attributes_mapping']].progress_apply(lambda x: process_char(x[0], x[1]), axis=1)
    etl.fillna('', inplace=True)
    etl['name_cat_minilm'] = etl['name_cat'].progress_apply(lambda text: minilm_name_cat.encode([text])[0])
    
    etl['clean_name'] = etl['name'].progress_apply(preprocess_text)
    df = (
        dataset
        .merge(
            etl
            .add_suffix('1'),
            on="variantid1"
        )
        .merge(
            etl
            .add_suffix('2'),
            on="variantid2"
        )
    )
    df[["pic_dist_0_perc", "pic_dist_25_perc", "pic_dist_50_perc"]] = (
        df[["pic_embeddings_resnet_v11", "pic_embeddings_resnet_v12"]].progress_apply(
            lambda x: pd.Series(get_pic_features_func(*x)), axis=1
        )
    )
    df['fuzzywuzzy_ratio_cleaned_name'] = df.progress_apply(lambda x: fuzz.ratio(x['clean_name1'], x['clean_name2']), axis=1)
    df['fuzzywuzzy_ratio_name'] = df.progress_apply(lambda x: fuzz.ratio(x.name1, x.name2), axis=1)
    
    # df[["euclidean_name_labse_dist", 
    #     "cosine_name_labse_dist",
    #     "jensenshannon_name_bert_dist",
    #     "minkowski_name_labse_dist",
    #     "sqeuclidean_name_labse_dist"
    #     ]] = (
    #     df[["name_labse1", "name_labse2"]].progress_apply(
    #         lambda x: pd.Series(text_dense_distances(*x)), axis=1
    #     )
    # )
    # df[["euclidean_name_rubert_dist", 
    #     "cosine_name_rubert_dist",
    #     "jensenshannon_name_bert_dist",
    #     "minkowski_name_rubert_dist",
    #     "sqeuclidean_name_rubert_dist"]] = (
    #     df[["name_rubert_tiny1", "name_rubert_tiny2"]].progress_apply(
    #         lambda x: pd.Series(text_dense_distances(*x)), axis=1
    #     )
    # )

    df[["euclidean_name_bert_dist", 
        "cosine_name_bert_dist",
        "jensenshannon_name_bert_dist",
        "minkowski_name_bert_dist",
        "sqeuclidean_name_bert_dist"]] = (
        df[["name_bert_641", "name_bert_642"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )
    df[["euclidean_name_minilm_dist", 
        "cosine_name_minilm_dist",
        "jensenshannon_name_minilmt_dist",
        "minkowski_name_minilm_dist",
        "sqeuclidean_name_minilm_dist"]] = (
        df[["name_minilm1", "name_minilm2"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )
    df[["euclidean_name_cat_minilm_dist", 
        "cosine_name_cat_minilm_dist",
        "jensenshannon_name_cat_minilmt_dist",
        "minkowski_name_cat_minilm_dist",
        "sqeuclidean_name_cat_minilm_dist"]] = (
        df[["name_cat_minilm1", "name_cat_minilm2"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )

    df["cat3"] = df["categories1"].progress_apply(lambda x: json.loads(x)["3"])
    cat3_counts = df["cat3"].value_counts().to_dict()
    cntr = 0
    for cat3 in cat3_counts:
        if cat3_counts[cat3] < 1_000:
            cntr += cat3_counts[cat3]

    df['main_pic_embeddings_resnet_v11'] = df['main_pic_embeddings_resnet_v11'].progress_apply(lambda x: x[0])
    df['main_pic_embeddings_resnet_v12'] = df['main_pic_embeddings_resnet_v12'].progress_apply(lambda x: x[0])
    df["cat3_grouped"] = df["cat3"].progress_apply(lambda x: x if cat3_counts[x] > 1000 else "rest")


    df[["euclidean_main_pic_embeddings_resnet_dist", 
        "cosine_main_pic_embeddings_resnet_dist",
        "jensenshannon_main_pic_embeddings_resnet_dist",
        "minkowski_main_pic_embeddings_resnet_dist",              
        "sqeuclidean_main_pic_embeddings_resnet_dist"]] = (
        df[["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )
    # Feature engineering
    df['levenshtein_distance'] = df.progress_apply(
    lambda x: pd.Series(jf.levenshtein_distance(x['name1'],
                                                x['name2'])), axis=1)

    df['damerau_levenshtein_distance'] = df.progress_apply(
    lambda x: jf.damerau_levenshtein_distance(x['name1'],
                                              x['name2']), axis=1)

    df['hamming_distance'] = df.progress_apply(
    lambda x: jf.hamming_distance(x['name1'],
                                  x['name2']), axis=1)

    df['jaro_similarity'] = df.progress_apply(
    lambda x: jf.jaro_similarity(x['name1'],
                                  x['name2']), axis=1)

    df['jaro_winkler_similarity'] = df.progress_apply(
    lambda x: jf.jaro_winkler_similarity(x['name1'],
                                         x['name2']), axis=1)

    df['partial_ratio'] = df.progress_apply(
    lambda x: fuzz.partial_ratio(x['name1'],
                                 x['name2']), axis=1)

    df['token_sort_ratio'] = df.progress_apply(
    lambda x: fuzz.token_sort_ratio(x['name1'],
                                    x['name2']), axis=1)

    df['token_set_ratio'] = df.progress_apply(
    lambda x: fuzz.token_set_ratio(x['name1'],
                                   x['name2']), axis=1)

    df['w_ratio'] = df.progress_apply(
    lambda x: fuzz.WRatio(x['name1'],
                          x['name2']), axis=1)

    df['uq_ratio'] = df.progress_apply(
    lambda x: fuzz.UQRatio(x['name1'],
                          x['name2']), axis=1)

    df['q_ratio'] = df.progress_apply(
    lambda x: fuzz.QRatio(x['name1'],
                          x['name2']), axis=1)    

    df['matching_numbers'] = df.progress_apply(
    lambda x: matching_numbers(x['name1'],
                               x['name2']), axis=1)

    df['matching_numbers_log'] = (df['matching_numbers']+1).progress_apply(np.log)

    df['log_fuzz_score'] = (df['fuzzywuzzy_ratio_name'] + df['partial_ratio'] + 
                            df['token_sort_ratio'] + df['token_set_ratio']).progress_apply(np.log)

    # df['log_fuzz_score_numbers'] = df['log_fuzz_score'] + (df['matching_numbers']).progress_apply(np.log)

    run_time = format(round((time.time() - start_time) / 60, 2))
    print("All columns of df:",df.columns)
    print('Total time of feature engineering:',run_time)
    if  train:
        df.to_parquet('train_processed_minilm_cat.parquet', index=False)
    else:
        df.to_parquet('test_processed_minilm_cat.parquet', index=False)        
    return df

etl_data(train=True)
etl_data(train=False)