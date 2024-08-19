import optuna
import pandas as pd
import numpy as np
import logging
import pytz
import json

from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from optuna.samplers import TPESampler
from datetime import datetime

# 'svm': {
#     'kernel': trial.suggest_categorical('svm__kernel', ['rbf', 'sigmoid']),
#     'gamma': trial.suggest_float('svm__gamma', 1e-5, 1),
#     'C': trial.suggest_float('svm__C', 1, 1e2),
#     'epsilon': trial.suggest_float('svm__epsilon', 1e-1, 1e1),
#     'max_iter': 20000,
# },

# 'rf': {
#     'n_estimators': trial.suggest_int('rf__n_estimators', 100, 450),
#     'max_depth': trial.suggest_int('rf__max_depth', 4, 50),
#     'min_samples_split': trial.suggest_int('rf__min_samples_split', 2, 15),
#     'min_samples_leaf': trial.suggest_int('rf__min_samples_leaf', 2, 15),
#     'max_features': trial.suggest_categorical('rf__max_features', ['sqrt', 'log2', 1.0]),
#     'min_impurity_decrease': trial.suggest_float('rf__min_impurity_decrease', 0, 1),
#     'ccp_alpha': trial.suggest_float('rf__ccp_alpha', 0, 10),
#     'random_state': SEED
# },

def main():

    SEED = 100
    NCALLS = 1000

    train_df = pd.read_parquet("../data/sentence_train.pq")
    X, y = train_df.drop(columns=['target']).to_numpy(), train_df['target'].to_numpy().astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=train_df['target'])

    def objective(trial):

        def suggest_params():
            return {
                'n_estimators': trial.suggest_int('lgbm__n_estimators', 50, 100),
                'max_depth': trial.suggest_int('lgbm__max_depth', 4, 50),
                'min_child_weight': trial.suggest_int('lgbm__min_child_weight', 1, 6),
                'learning_rate': trial.suggest_float('lgbm__learning_rate', 1e-5, 1),
                'reg_alpha': trial.suggest_float('lgbm__reg_alpha', 0, 1e1),
                'reg_lambda': trial.suggest_float('lgbm__reg_lambda', 0, 1e1),
                'verbosity': -1,
                'random_state': SEED
            }
        
        def loss(y, y_pred):
            return -np.sum((y * np.log(y_pred)) + (1 - y) * np.log(1 - y_pred))

        params = suggest_params()

        lgbm = LGBMClassifier(
            **params
        )

        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict_proba(X_test)[:,1]
        
        return loss(y_test, y_pred)

    timezone = pytz.timezone("Australia/Melbourne")
    log_path = "./logs/"
    time = datetime.now(tz=timezone)
    file = f"optuna_run_{time}"
    fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, file))
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.addHandler(fileHandler)

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=NCALLS)

    return study.best_params, time

if __name__ == "__main__":
    params, time = main()

    with open(f"./params/params_{time}.json", "w") as fp:
        json.dump(params, fp)