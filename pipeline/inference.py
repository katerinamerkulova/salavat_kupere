import argparse
import logging
import os

import pandas as pd
from catboost import CatBoostClassifier
from transliterate import translit

from features import feats, categorical_feats, characteristic_feats

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
pd.set_option('use_inf_as_na', True)

MODEL_DIR = '../models/catboost/'


def run_inference(DATA_DIR="../../data/"):
    test_features = pd.read_parquet(os.path.join(DATA_DIR, "test_processed_chars_by_cat_color.parquet"))
    test_features.fillna(0, inplace=True)

    data_features = feats + characteristic_feats + categorical_feats + ['variantid1', 'variantid2']
    model_name = 'salavat_kupere'

    new_model = CatBoostClassifier()
    submission_example = test_features.copy()
    groups = submission_example['cat3_grouped1']
    for group in set(groups):
        group_name = '_'.join(group.replace('/', ' ').split())
        model_path = translit(f"{model_name}_{group_name}", 'ru', reversed=True)
        model = new_model.load_model(os.path.join(MODEL_DIR, model_path, '.cbm'), format='cbm')
        X = submission_example.loc[submission_example['cat3_grouped1'] == group]
        submission_example.loc[X.index, 'target'] = model.predict_proba(X[data_features])[:, 1]

    submission_example = submission_example[["variantid1", "variantid2", "target", "cat3_grouped1"]]
    submission_example.drop_duplicates(subset=['variantid1', 'variantid2']).merge(
        test_features[["variantid1", "variantid2"]].drop_duplicates(["variantid1", "variantid2"]),
        on=["variantid1", "variantid2"]
    ).to_csv(f"../submission_{model_name}.csv", index=False)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Path to dir with data"
    )

    args = argParser.parse_args()
    data_dir=args.data_dir
    if data_dir is not None:
        run_inference(DATA_DIR=data_dir)
    else:
        logging.error("no --data_dir param as input") 

