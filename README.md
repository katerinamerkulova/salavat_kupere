# Experiments notes
Можно использовать этот файл для записи всех гипотез
папка с данными лежит в корне /home/ailab_user/data
## Fast start
* activate venv 
`source /venv/bin/activate` 

# Предложения
1. [StratifiedGroupKFold](https://www.kaggle.com/code/realsid/stratifiedgroupkfold-using-sklearn)
1. Увеличение товаров для обучения использовать правило транзитивности, см ноутбук eda.ipynb
1. Препроцессы разные завернуть в pandas pipe
1. Попробовать разные encoders (tfidf, LABSE, rubert-tiny2), млм на текущих данных
1. Анализ влияния фичей (SHAP, feature importance, correlation matrix)
1. Optuna, RandomCVSearch для подбора гиперпараметров catboost [useful1](https://github.com/nyanp/nyaggle/tree/master/nyaggle/hyper_parameters)
1. stacking [kaggle notebook](https://www.kaggle.com/code/dailysergey/howtodata-stacking-explained)
1. Feature engineering
1. Ensemble
1. Отдельно загружать variantid, name_encoded, color_encoded, 

## Useful notebooks
* [rubert-tiny-largevoc](https://colab.research.google.com/drive/1mSWfIQ6PIlteLVZ9DKKpcorycgLIKZLf?usp=sharing#scrollTo=cFYNSS90QnFR)
* [jigsaw анализ токсичных выражений в тексте, в основном тащили линейные модели](https://www.kaggle.com/code/dailysergey/mega-b-ridge-to-the-top-0-84-roberta-0-fold)
* [how to create product match with xgboost](https://practicaldatascience.co.uk/machine-learning/how-to-create-a-product-matching-model-using-xgboost)