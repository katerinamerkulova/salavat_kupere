from joblib import dump, load

import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from transliterate import translit

from features import feats, categorical_feats, characteristic_feats
from utils import StratifiedGroupKFold, pr_auc_macro, seed_everything, postprocess


# CFG
N_SPLITS = 10
mlflow.set_tracking_uri("http://10.100.10.70:5080")
pd.set_option('use_inf_as_na', True)
USE_CV = False
VOTING = False
# MLFLOW_USER ='guskov'
MLFLOW_USER = 'kate'

GLOBAL_SEED = 42
RUN_NAME = f"chars_by_cat_color_catboost_by_cat"

seed_everything(GLOBAL_SEED)


EXPERIMENT_NAME = "minilm_ozon"
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
if not EXPERIMENT_ID:
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)


train_df = pd.read_parquet('../../data/train_processed_chars_by_cat_color.parquet')
train_df.fillna(0, inplace=True)

print('train_df.columns:', train_df.columns)
print('characteristic_feats:', len(characteristic_feats))
feats = feats + characteristic_feats + categorical_feats

# Let's try StratifiedGroupKFold
all_groups = train_df['cat3_grouped1']

X_train, X_test = train_test_split(
	train_df[feats + ["target", "variantid1", "variantid2"]],
	test_size=0.15, random_state=GLOBAL_SEED, stratify=train_df[["target", "cat3_grouped1"]]
)
y_test = X_test[["target", "variantid1", "variantid2"]]
X_test = X_test.drop(columns=["target"])

with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run: 
    # Retrieve run id
    RUN_ID = run.info.run_id    
    # Track parameters
    
    mlflow.log_text(', '.join(feats), "feats.txt")
    mlflow.log_text(', '.join(categorical_feats), "categorical_feats.txt")
    mlflow.log_param("GLOBAL_SEED", GLOBAL_SEED)


    params = {"random_seed": GLOBAL_SEED, "eval_metric": 'PRAUC', "cat_features": categorical_feats}
    model = CatBoostClassifier(**params)
    model_name = "CatBoost"

    groups = X_train['cat3_grouped1']
    for group in set(groups):
        X = X_train.loc[X_train['cat3_grouped1'] == group]
        y = X['target']
        X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.15, random_state=GLOBAL_SEED, stratify=X[["target"]])
        group_name = '_'.join(group.replace('/', ' ').split())
        model_path = translit(f"../models/categories/{model_name}_{RUN_NAME}_{group_name}", 'ru', reversed=True)
        train_pool = Pool(
            data=X_t.drop(columns=['target']),
            label=y_t,
            cat_features=categorical_feats
        )
        eval_pool = Pool(
            data=X_val.drop(columns=['target']),
            label=y_val,
            cat_features=categorical_feats
        )
        model.fit(
            train_pool,
            eval_set=eval_pool,
            plot=True,
            verbose=True if group == 'Компьютер' else False,
            use_best_model=True,
            early_stopping_rounds=100,
        )
        model.save_model(model_path + '.cbm')
    
    for group in set(groups):
        group_name = '_'.join(group.replace('/', ' ').split())
        model_path = translit(f"../models/categories/{model_name}_{RUN_NAME}_{group_name}", 'ru', reversed=True)
        model = model.load_model(model_path + '.cbm', format='cbm')
        X = X_test.loc[X_test['cat3_grouped1'] == group]
        X_test.loc[X.index, 'scores'] = model.predict_proba(X[feats + ["variantid1", "variantid2"]])[:, 1]

    print('Model saved with file name', model_path)
    mlflow.log_param("model_path", model_path)

    pr_auc_macro_dict = pr_auc_macro(
        target_df=y_test, 
        predictions_df=X_test,
        prec_level=0.75,
        cat_column="cat3_grouped1"
    )

    pr_auc_macro_metr = pr_auc_macro_dict['pr_auc']
    pd.DataFrame(pr_auc_macro_dict).to_csv(f'../../pr_auc/{RUN_NAME}.csv', index=False)
    X_test[["target", "variantid1", "variantid2"]] = y_test
    X_test[["target", "variantid1", "variantid2", "scores", "cat3_grouped1", "cat3_grouped2"]].to_csv(f'../../data/test_processed_{RUN_NAME}.csv', index=False)
    mlflow.log_param("USE_CV", USE_CV)
    mlflow.log_param("N_SPLITS", N_SPLITS)
    mlflow.log_param("MLFLOW_USER", MLFLOW_USER)
    # to visualize in mlflow
    mlflow.log_param("pr_auc_macro", pr_auc_macro_metr)
    # Track metrics
    mlflow.log_metric("pr_auc_macro_metr", pr_auc_macro_metr)
    print('pr_auc_macro_metr:', pr_auc_macro_metr)
    # mlflow.pytorch.log_model(MiniLM, artifact_path='all-MiniLM-L6-v2-2023-05-20_17-54-44')