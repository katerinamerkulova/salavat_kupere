import argparse
import json
import logging
import os
import time
import warnings

import jellyfish as jf
import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import get_pic_features_func, text_dense_distances, preprocess_text, process_characteristics, process_char
from utils import process_colors, matching_numbers, make_characteristics_df

tqdm.pandas()
warnings.filterwarnings("ignore")
pd.set_option("use_inf_as_na", True)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

minilm_name = SentenceTransformer('../models/name_minilm')  # todo PATH TO RELEASE
minilm_name_cat = SentenceTransformer('../models/full_minilm')


def etl_data(DATA_DIR = "../../data/hackathon_files_for_participants_ozon"):
    start_time = time.time()
    train_pairs = pd.read_parquet(os.path.join(DATA_DIR, "train_pairs.parquet"))
    test_pairs = pd.read_parquet(os.path.join(DATA_DIR, "test_pairs_wo_target.parquet"))

    train = pd.read_parquet(os.path.join(DATA_DIR, "train_data.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_data.parquet"))

    data = pd.concat([train, test])

    data["name_minilm"] = data["name"].progress_apply(lambda text: minilm_name.encode([text])[0])
    data["name_cat"] = data[["name", "characteristic_attributes_mapping"]].progress_apply(lambda x: process_char(x[0], x[1]), axis=1)
    data["name_cat"] = data["name_cat"].fillna("")
    data["name_cat_minilm"] = data["name_cat"].progress_apply(lambda text: minilm_name_cat.encode([text])[0])

    data["clean_name"] = data["name"].progress_apply(preprocess_text)
    data["main_pic_embeddings_resnet_v1"] = data["main_pic_embeddings_resnet_v1"].progress_apply(lambda x: x[0]) # unpack

    data["cat3"] = data["categories"].progress_apply(lambda x: json.loads(x)["3"])
    data["cat4"] = data["categories"].progress_apply(lambda x: json.loads(x)["4"])
    cat3_counts = data["cat3"].value_counts().to_dict()
    cntr = 0
    COUNT_GROUP = 50
    for cat3 in cat3_counts:
        if cat3_counts[cat3] < COUNT_GROUP: #1_000
            cntr += cat3_counts[cat3]
    data["cat3_grouped"] = data["cat3"].progress_apply(lambda x: x if cat3_counts[x] > COUNT_GROUP else "rest")

    print(f"Preprocessing characteristics" )
    data, characteristics = process_characteristics(data)
    print("Characteristics features", len(characteristics))
    with open("chars.txt", "w", encoding="utf-8") as out:
        out.write("\n".join(sorted(characteristics)))

    df = (
        train_pairs
        .merge(
            data
            .add_suffix("1"),
            on="variantid1"
        )
        .merge(
            data
            .add_suffix("2"),
            on="variantid2"
        )
    )
    df["Train"] = True
    print("Train", df.shape)
    df_test = (
        test_pairs
        .merge(
            data
            .add_suffix("1"),
            on="variantid1"
        )
        .merge(
            data
            .add_suffix("2"),
            on="variantid2"
        )
    )
    df_test["Train"] = False
    print("Test", df_test.shape)
    df = pd.concat([df, df_test])
    print("concat", df.shape)

    characteristics_df = make_characteristics_df(df, characteristics)
    df.drop(columns=[char + "1" for char in characteristics] + [char + "2" for char in characteristics], inplace=True)
    df = df.join(characteristics_df)

    df[["pic_dist_0_perc", "pic_dist_25_perc", "pic_dist_50_perc"]] = (
        df[["pic_embeddings_resnet_v11", "pic_embeddings_resnet_v12"]].progress_apply(
            lambda x: pd.Series(get_pic_features_func(*x)), axis=1
        )
    )
    df["fuzzywuzzy_ratio_cleaned_name"] = df.progress_apply(lambda x: fuzz.ratio(x["clean_name1"], x["clean_name2"]), axis=1)
    df["fuzzywuzzy_ratio_name"] = df.progress_apply(lambda x: fuzz.ratio(x.name1, x.name2), axis=1)

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
        "jensenshannon_name_minilm_dist",
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

    # process color
    df[[
        "is_same_color",
        "partial_ratio_color"
    ]] = df[["color_parsed1", "color_parsed2"]].progress_apply(
        lambda x: pd.Series(process_colors(*x)), axis=1
    )
    for i, (cat, is_same_color) in df[["cat3_grouped1", "is_same_color"]].iterrows():
        df.loc[i, f"is_same_color_{cat}"] = is_same_color

    df["concat_main_pic_resnet_bert1"] = df.apply(lambda row: torch.cat((torch.tensor(row["main_pic_embeddings_resnet_v11"] ), torch.tensor(row["name_bert_641"])) ).cpu().numpy(), axis=1)
    df["concat_main_pic_resnet_bert2"] = df.apply(lambda row: torch.cat((torch.tensor(row["main_pic_embeddings_resnet_v12"] ), torch.tensor(row["name_bert_642"])) ).cpu().numpy(), axis=1)
    df["concat_main_pic_resnet_minilm1"] = df.apply(lambda row: torch.cat((torch.tensor(row["main_pic_embeddings_resnet_v11"] ), torch.tensor(row["name_minilm1"])) ).cpu().numpy(), axis=1)
    df["concat_main_pic_resnet_minilm2"] = df.apply(lambda row: torch.cat((torch.tensor(row["main_pic_embeddings_resnet_v12"] ), torch.tensor(row["name_minilm2"])) ).cpu().numpy(), axis=1)

    df[["euclidean_minilm_main_resnet_dist",
        "cosine_minilm_main_resnet_resnet_dist",
        "jensenshannon_minilm_main_resnet_resnet_dist",
        "minkowski_minilm_main_resnet_resnet_dist",
        "sqeuclidean_minilm_main_resnet_resnet_dist"]] = (
        df[["concat_main_pic_resnet_minilm1", "concat_main_pic_resnet_minilm2"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )

    df[["euclidean_bert_main_resnet_dist",
        "cosine_bert_main_resnet_resnet_dist",
        "jensenshannon_bert_main_resnet_resnet_dist",
        "minkowski_bert_main_resnet_resnet_dist",
        "sqeuclidean_bert_main_resnet_resnet_dist"]] = (
        df[["concat_main_pic_resnet_bert1", "concat_main_pic_resnet_bert2"]].progress_apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )

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
    df["levenshtein_distance"] = df.progress_apply(
        lambda x: pd.Series(jf.levenshtein_distance(x["clean_name1"],
                                                    x["clean_name2"])), axis=1)

    df["damerau_levenshtein_distance"] = df.progress_apply(
        lambda x: jf.damerau_levenshtein_distance(x["clean_name1"],
                                                  x["clean_name2"]), axis=1)

    df["hamming_distance"] = df.progress_apply(
        lambda x: jf.hamming_distance(x["clean_name1"],
                                      x["clean_name2"]), axis=1)

    df["jaro_similarity"] = df.progress_apply(
        lambda x: jf.jaro_similarity(x["clean_name1"],
                                     x["clean_name2"]), axis=1)

    df["jaro_winkler_similarity"] = df.progress_apply(
        lambda x: jf.jaro_winkler_similarity(x["clean_name1"],
                                             x["clean_name2"]), axis=1)

    df["partial_ratio"] = df.progress_apply(
        lambda x: fuzz.partial_ratio(x["clean_name1"],
                                     x["clean_name2"]), axis=1)

    df["token_sort_ratio"] = df.progress_apply(
        lambda x: fuzz.token_sort_ratio(x["clean_name1"],
                                        x["name2"]), axis=1)

    df["token_set_ratio"] = df.progress_apply(
        lambda x: fuzz.token_set_ratio(x["clean_name1"],
                                       x["clean_name2"]), axis=1)

    df["w_ratio"] = df.progress_apply(
        lambda x: fuzz.WRatio(x["clean_name1"],
                              x["clean_name2"]), axis=1)

    df["uq_ratio"] = df.progress_apply(
        lambda x: fuzz.UQRatio(x["clean_name1"],
                               x["clean_name2"]), axis=1)

    df["q_ratio"] = df.progress_apply(
        lambda x: fuzz.QRatio(x["clean_name1"],
                              x["clean_name2"]), axis=1)

    df["matching_numbers"] = df.progress_apply(
        lambda x: matching_numbers(x["clean_name1"],
                                   x["clean_name2"]), axis=1)

    df["matching_numbers_log"] = (df["matching_numbers"] + 1).progress_apply(np.log)

    df["log_fuzz_score"] = (df["fuzzywuzzy_ratio_name"] + df["partial_ratio"] +
                            df["token_sort_ratio"] + df["token_set_ratio"]).progress_apply(np.log)

    columns_to_drop = [
        "name1", "name2",
        "clean_name1", "clean_name2",
        "categories1", "categories2",
        "color_parsed1", "color_parsed2",
        "characteristic_attributes_mapping1",
        "characteristic_attributes_mapping2",
        "pic_embeddings_resnet_v11",
        "pic_embeddings_resnet_v12",
        "main_pic_embeddings_resnet_v11",
        "main_pic_embeddings_resnet_v12",
        "concat_main_pic_resnet_bert1",
        "concat_main_pic_resnet_bert2",
        "concat_main_pic_resnet_minilm1",
        "concat_main_pic_resnet_minilm2",
        "name_bert_641", "name_bert_642",
        "name_minilm1", "name_minilm2",
        "name_cat_minilm1", "name_cat_minilm1",
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    run_time = format(round((time.time() - start_time)/60,2))
    print("All columns of df:", df.columns.shape)
    print('Total time of feature engineering:', run_time)

    train = df[df['Train'] == True]
    test = df[df['Train'] == False]
    
    train.drop(columns=['Train'])
    train.to_parquet(os.path.join(DATA_DIR, 'train_processed_minilm_chars.parquet'), index=False)
    
    test.drop(columns=['Train'])
    test.to_parquet(os.path.join(DATA_DIR, 'test_processed_minilm_chars.parquet'), index=False)
    return df


if __name__ == "__main__":
    etl_data()
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data_dir", 
                        default=None,
                        type=str,
                        required=True,
                        help="Path to dir with data")

    args = argParser.parse_args()
    data_dir=args.data_dir
    if data_dir is not None:
        etl_data(DATA_DIR=data_dir)
    else:
        logging.error("no --data_dir param as input")
