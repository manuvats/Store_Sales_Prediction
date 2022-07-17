# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config):
    try:
        val_data_path = config["split_data"]["validation_path"]
        train_data_path = config["split_data"]["train_path"]
        random_state = config["base"]["random_state"]
        model_dir = config["model_dir"]

        max_depth = config["estimators"]["DecisionTree"]["params"]["max_depth"]
        min_samples_leaf = config["estimators"]["DecisionTree"]["params"]["min_samples_leaf"]

        target = [config["base"]["target_col"]]

        train = pd.read_csv(train_data_path, sep=",")
        val = pd.read_csv(val_data_path, sep=",")

        train_y = train[target]
        val_y = val[target]

        train_x = train.drop(target, axis=1)
        val_x = val.drop(target, axis=1)
    
        dtr = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state)
        #print(np.all(np.isfinite(train_x)))
        dtr.fit(train_x, train_y)
        predicted_sales = dtr.predict(val_x)

        (rmse, mae, r2) = eval_metrics(val_y, predicted_sales)

        print("Decision Tree model (max_depth=%f, min_samples_leaf=%f):" % (max_depth, min_samples_leaf))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]

        with open(scores_file, "w") as f:
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            json.dump(scores, f, indent=4)

        with open(params_file, "w") as f:
            params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
            json.dump(params, f, indent=4)
#####################################################


        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")

        joblib.dump(dtr, model_path)

    except Exception as e:
        raise Exception("(train_and_evaluate): " + str(e))
################### MLFLOW ###############################
    '''    mlflow_config = config["mlflow_config"]
        remote_server_uri = mlflow_config["remote_server_uri"]

        mlflow.set_tracking_uri(remote_server_uri)

        mlflow.set_experiment(mlflow_config["experiment_name"])
        print(remote_server_uri)
        print('artifact uri:', mlflow.get_artifact_uri())
        with mlflow.start_run (run_name=mlflow_config["run_name"]) as mlops_run:
            print('tracking uri:', mlflow.get_tracking_uri())
            print('artifact uri:', mlflow.get_artifact_uri())
            dtr = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state)
            dtr.fit(train_x, train_y)

            predicted_sales = dtr.predict(val_x)

            (rmse, mae, r2) = eval_metrics(val_y, predicted_sales)

            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R-sqr", r2)

            tracking_uri_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_uri_type_store!="file":
                mlflow.sklearn.log_model(
                    dtr, 
                    "model", 
                    registered_model_name = mlflow_config["registered_model_name"]
                    )
            else:
                mlflow.sklearn.load_model(dtr, "model")'''
    #####################################################
        

if __name__=="__main__":
    from get_data import read_params
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)
    train_and_evaluate(config)