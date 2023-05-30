import argparse
import json
import logging
import os

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from transliterate import translit

from utils import pr_auc_macro, seed_everything

pd.set_option("use_inf_as_na", True)

#### Just some code to print debug information to stdout
logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


GLOBAL_SEED = 42
seed_everything(GLOBAL_SEED)


def run_training(
        DATA_DIR="../../data/hackathon_files_for_participants_ozon",
        MODEL_DIR="../models"
):
    model_dir = os.path.join(MODEL_DIR, "catboost")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_processed_minilm_chars.parquet"))
    train_df.fillna(0, inplace=True)
    features = json.load(open("features.json", encoding="utf-8"))

    print("train_df.columns:", train_df.columns)
    print("characteristic_feats:", len(features["characteristic_feats"]))
    data_features = features["feats"] + features["characteristic_feats"] + features["categorical_feats"] + ["variantid1", "variantid2"]
    model_name = "salavat_kupere"

    X_train, X_test = train_test_split(
        train_df[data_features + ["target"]],
        test_size=0.15, random_state=GLOBAL_SEED, stratify=train_df[["target", "cat3_grouped1"]]
    )
    y_test = X_test[["target", "variantid1", "variantid2"]]
    X_test = X_test.drop(columns=["target"])

    params = {"random_seed": GLOBAL_SEED, "eval_metric": "PRAUC", "cat_features": features["categorical_feats"]}
    model = CatBoostClassifier(**params)

    groups = X_train["cat3_grouped1"]
    # train
    for group in set(groups):
        X = X_train.loc[X_train["cat3_grouped1"] == group]
        y = X["target"]
        X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.15, random_state=GLOBAL_SEED, stratify=X[["target"]])
        group_name = "_".join(group.replace("/", " ").split())
        model_path = translit(f"{model_name}_{group_name}", "ru", reversed=True)
        train_pool = Pool(
            data=X_t.drop(columns=["target"]),
            label=y_t,
            cat_features=features["categorical_feats"]
        )
        eval_pool = Pool(
            data=X_val.drop(columns=["target"]),
            label=y_val,
            cat_features=features["categorical_feats"]
        )
        model.fit(
            train_pool,
            eval_set=eval_pool,
            plot=True,
            verbose=True,
            use_best_model=True,
            early_stopping_rounds=100,
        )
        model.save_model(os.path.join(model_dir, model_path, ".cbm"))

    # validate
    for group in set(groups):
        group_name = "_".join(group.replace("/", " ").split())
        model_path = translit(f"{model_name}_{group_name}", "ru", reversed=True)
        model = model.load_model(os.path.join(model_dir, model_path, ".cbm"), format="cbm")
        X = X_test.loc[X_test["cat3_grouped1"] == group]
        X_test.loc[X.index, "scores"] = model.predict_proba(X[data_features])[:, 1]

        pr_auc_macro_dict = pr_auc_macro(
            target_df=y_test, 
            predictions_df=X_test,
            prec_level=0.75,
            cat_column="cat3_grouped1"
        )
        pr_auc_macro_metr = pr_auc_macro_dict["pr_auc"]
        print("pr_auc_macro_metr:", pr_auc_macro_metr)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data_dir",
                           default=None,
                           type=str,
                           required=True,
                           help="Path to dir with data")
    argParser.add_argument("--model_dir",
                           default=None,
                           type=str,
                           required=True,
                           help="Path to dir with model")

    args = argParser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    if data_dir is not None:
        run_training(DATA_DIR=data_dir, MODEL_DIR=model_dir)
    else:
        logging.error("no --data_dir param as input")
