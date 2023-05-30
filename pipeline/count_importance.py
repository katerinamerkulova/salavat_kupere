import json

import pandas as pd
from catboost import CatBoostClassifier


new_model = CatBoostClassifier()
model_name = "catboost_chars_by_cat"
model = new_model.load_model(f"../models/{model_name}.cbm", format="cbm")

imp = pd.Series(model.get_feature_importance(), index=model.feature_names_)
imp = imp.sort_values(ascending=False).to_dict()
with open(f"../importance/feature_importance_{model_name}.json", "w", encoding="utf-8") as out:
    json.dump(imp, out, ensure_ascii=False, indent=1)