import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from transliterate import translit

from features import feats, categorical_feats, characteristic_feats
from utils import pr_auc_macro, seed_everything

pd.set_option("use_inf_as_na", True)

GLOBAL_SEED = 42
RUN_NAME = "chars_by_cat_color_catboost_by_cat"

seed_everything(GLOBAL_SEED)

train_df = pd.read_parquet("../../data/train_processed_chars_by_cat_color.parquet")
train_df.fillna(0, inplace=True)

print("train_df.columns:", train_df.columns)
print("characteristic_feats:", len(characteristic_feats))
feats = feats + characteristic_feats + categorical_feats

all_groups = train_df["cat3_grouped1"]

X_train, X_test = train_test_split(
    train_df[feats + ["target", "variantid1", "variantid2"]],
    test_size=0.15, random_state=GLOBAL_SEED, stratify=train_df[["target", "cat3_grouped1"]]
)
y_test = X_test[["target", "variantid1", "variantid2"]]
X_test = X_test.drop(columns=["target"])

params = {"random_seed": GLOBAL_SEED, "eval_metric": "PRAUC", "cat_features": categorical_feats}
model = CatBoostClassifier(**params)

groups = X_train["cat3_grouped1"]
for group in set(groups):
    X = X_train.loc[X_train["cat3_grouped1"] == group]
    y = X["target"]
    X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.15, random_state=GLOBAL_SEED, stratify=X[["target"]])
    group_name = "_".join(group.replace("/", " ").split())
    model_path = translit(f"catboost/{RUN_NAME}_{group_name}", "ru", reversed=True)
    train_pool = Pool(
        data=X_t.drop(columns=["target"]),
        label=y_t,
        cat_features=categorical_feats
    )
    eval_pool = Pool(
        data=X_val.drop(columns=["target"]),
        label=y_val,
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
    model.save_model(model_path + ".cbm")

for group in set(groups):
    group_name = "_".join(group.replace("/", " ").split())
    model_path = translit(f"catboost/{RUN_NAME}_{group_name}", "ru", reversed=True)
    model = model.load_model(model_path + ".cbm", format="cbm")
    X = X_test.loc[X_test["cat3_grouped1"] == group]
    X_test.loc[X.index, "scores"] = model.predict_proba(X[feats + ["variantid1", "variantid2"]])[:, 1]

pr_auc_macro_dict = pr_auc_macro(
    target_df=y_test,
    predictions_df=X_test,
    prec_level=0.75,
    cat_column="cat3_grouped1"
)

pr_auc_macro_metr = pr_auc_macro_dict["pr_auc"]
print("pr_auc_macro_metr:", pr_auc_macro_metr)