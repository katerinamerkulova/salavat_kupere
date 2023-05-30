from joblib import dump, load

import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

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
RUN_NAME = f"chars_by_cat_color"

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
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED)
all_groups = train_df['cat3_grouped1']

X_train, X_test = train_test_split(
	train_df[feats + ["target", "variantid1", "variantid2"]],
	test_size=0.15, random_state=GLOBAL_SEED, stratify=train_df[["target", "cat3_grouped1"]]
)
y_test = X_test[["target", "variantid1", "variantid2"]]
X_test = X_test.drop(columns=["target"])
X = X_train[feats]
y = X_train["target"]

with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run: 
    # Retrieve run id
    RUN_ID = run.info.run_id    
    # Track parameters
    
    mlflow.log_text(', '.join(feats), "feats.txt")
    mlflow.log_text(', '.join(categorical_feats), "categorical_feats.txt")
    mlflow.log_param("GLOBAL_SEED", GLOBAL_SEED)

    if VOTING:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier
        import lightgbm as lgb
        classifiers = {
        # "DummyClassifier_stratified": DummyClassifier(strategy='stratified', random_state=GLOBAL_SEED),
        "XGBClassifier": XGBClassifier(n_estimators=1000, learning_rate=0.1, seed=GLOBAL_SEED, eval_metric='aucpr'),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=GLOBAL_SEED),
        "RandomForestClassifier": RandomForestClassifier(random_state=GLOBAL_SEED),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=GLOBAL_SEED),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=GLOBAL_SEED),
        "MLP": MLPClassifier(random_state=GLOBAL_SEED),
        "XGBClassifer tuned": XGBClassifier(colsample_bytree=0.8,
                        gamma=0.9,
                        max_depth=20,
                        min_child_weight=1,
                        scale_pos_weight=12,
                        subsample=0.9,
                        n_estimators=50,
                        learning_rate=0.1,
                        seed=GLOBAL_SEED),
        "CatBoostClassifier": CatBoostClassifier(random_seed=GLOBAL_SEED, eval_metric='PRAUC', early_stopping_rounds=10),
        "LGBMClassifier": lgb.LGBMClassifier(random_seed=GLOBAL_SEED),
        }
        model = VotingClassifier([(k, v) for k, v in classifiers.items()], voting='soft', verbose=True)
        model_name = "VotingClassifier"
    else:
        params = {"random_seed": GLOBAL_SEED, "eval_metric": 'PRAUC', "cat_features": categorical_feats}
        model = CatBoostClassifier(**params)
        model_name = "CatBoost"

    if USE_CV:
        groups = X_train['cat3_grouped1']
        for train_index, val_index in cv.split(X, y, groups=groups):
            X_t, X_val = X.iloc[train_index], X.iloc[val_index]
            y_t, y_val = y.iloc[train_index], y.iloc[val_index]
            model_path = f"../models/{model_name}_{RUN_NAME}_StratifiedGroupKFold"
            if model_name == 'CatBoost':
                train_pool = Pool(
                    data=X_t,
                    label=y_t,
                    cat_features=categorical_feats
                )
                eval_pool = Pool(
                    data=X_val,
                    label=y_val,
                    cat_features=categorical_feats
                )
                model.fit(
                    train_pool,
                    eval_set=eval_pool,
                    plot=True,
                    verbose=True,
                    use_best_model=True,
                    early_stopping_rounds=1000,
                )
                model.save_model(model_path + '.cbm')
            else:
                model.fit(X.drop(columns=categorical_feats), y,)
                dump(model, model_path + '.joblib')
    else:
        X_t, X_val = train_test_split(
            X_train[feats + ["target",  "variantid1", "variantid2"]], 
            test_size=0.1, 
            random_state=GLOBAL_SEED, 
            stratify = X_train[["target", "cat3_grouped1"]]
        )
        y_t = X_t["target"]
        y_val = X_val["target"]

        X_t = X_t.drop(["target"], axis=1)
        X_val = X_val.drop(["target"], axis=1)

        if model_name == 'CatBoost':
            train_pool = Pool(
                data=X_t,
                label=y_t,
                cat_features=categorical_feats
            )
            eval_pool = Pool(
                data=X_val,
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
            model_path = f"../models/{model_name}_{RUN_NAME}"
            model.save_model(model_path + '.cbm')
        else:
            model.fit(X.drop(columns=categorical_feats), y)
            model_path = f"../models/{model_name}_{RUN_NAME}"
            dump(model, model_path + '.joblib')

    print('Model saved with file name', model_path)
    mlflow.log_param("model_path", model_path)
    if model_name == 'CatBoost':
        X_test["scores"] = model.predict_proba(X_test[feats + ["variantid1", "variantid2"]])[:, 1]
    else:
        X_test["scores"] = model.predict_proba(X_test[feats].drop(columns=categorical_feats))[:, 1]

    X_test["scores"] = X_test.apply(lambda x: postprocess(x['scores'], x['cat3_grouped1']), axis=1)

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