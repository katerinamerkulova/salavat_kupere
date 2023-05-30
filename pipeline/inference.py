from joblib import load

import pandas as pd
from catboost import CatBoostClassifier
from transliterate import translit

from features import feats, categorical_feats, characteristic_feats
from utils import postprocess


pd.set_option('use_inf_as_na', True)
GLOBAL_SEED = 42
BY_CATEGORY = True
test_features = pd.read_parquet("../../data/test_processed_chars_by_cat_color.parquet")
test_features.fillna(0, inplace=True)

feats = feats + characteristic_feats + categorical_feats + ['variantid1', 'variantid2']
# feats = feats + characteristic_feats
model_name = 'CatBoost_chars_by_cat_color_catboost_by_cat'

new_model = CatBoostClassifier()
submission_example = test_features.copy()

if BY_CATEGORY:
    groups = submission_example['cat3_grouped1']
    for group in set(groups):
        group_name = '_'.join(group.replace('/', ' ').split())
        model_path = translit(f"../models/categories/{model_name}_{group_name}", 'ru', reversed=True)
        model = new_model.load_model(model_path + '.cbm', format='cbm')
        X = submission_example.loc[submission_example['cat3_grouped1'] == group]
        submission_example.loc[X.index, 'target'] = model.predict_proba(X[feats])[:, 1]
else:
    model = new_model.load_model(f'../models/{model_name}.cbm', format='cbm')
    # model = load(f'../models/{model_name}.joblib')
    submission_example["target"] = model.predict_proba(test_features[feats])[:, 1]
    submission_example["target"] = submission_example.apply(lambda x: postprocess(x['target'], x['cat3_grouped1']), axis=1)

submission_example = submission_example[["variantid1", "variantid2", "target", "cat3_grouped1"]]
submission_example.drop_duplicates(subset=['variantid1', 'variantid2']).merge(
    test_features[["variantid1", "variantid2"]].drop_duplicates(["variantid1", "variantid2"]),
    on=["variantid1", "variantid2"]
).to_csv(f"../submission_files/submission_{model_name}.csv", index=False)
