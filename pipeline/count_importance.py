import argparse
import json
import logging
import os

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from utils import seed_everything

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
pd.set_option("use_inf_as_na", True)

GLOBAL_SEED = 42
seed_everything(GLOBAL_SEED)

categorical_feats = [
    "cat3_grouped1",
    "cat3_grouped2",
    "cat41",
    "cat42",
]


def importance(
        DATA_DIR="../../data/hackathon_files_for_participants_ozon"
):
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_processed_minilm_chars.parquet"))
    train_df.fillna(0, inplace=True)

    X_train, X_test = train_test_split(
        train_df, test_size=0.15, random_state=GLOBAL_SEED, stratify=train_df[["target", "cat3_grouped1"]]
    )
    y_train = X_train[["target", "variantid1", "variantid2"]]
    y_test = X_test[["target", "variantid1", "variantid2"]]
    params = {"random_seed": GLOBAL_SEED, "eval_metric": "PRAUC", "cat_features": categorical_feats}
    model = CatBoostClassifier(**params)
    train_pool = Pool(
        data=X_train.drop(columns=["target"]),
        label=y_train,
        cat_features=categorical_feats
    )
    eval_pool = Pool(
        data=X_test.drop(columns=["target"]),
        label=y_test,
        cat_features=categorical_feats
    )
    model.fit(
        train_pool,
        eval_set=eval_pool,
        plot=True,
        verbose=True,
        use_best_model=True,
        early_stopping_rounds=100,
    )

    imp = pd.Series(model.get_feature_importance(), index=model.feature_names_) > 0
    features = json.load(open("features.json", encoding="utf-8"))
    features["characteristic_feats"] = sorted(imp.index.to_list())
    with open("../features.json", "w", encoding="utf-8") as out:
        json.dump(features, out, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data_dir",
                           default=None,
                           type=str,
                           required=True,
                           help="Path to dir with data")

    args = argParser.parse_args()
    data_dir = args.data_dir
    if data_dir is not None:
        importance(DATA_DIR=data_dir)
    else:
        logging.error("no --data_dir param as input")
