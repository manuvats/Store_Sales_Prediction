# split the raw data
# save it in data/processed folder
import yaml
import os
import json
import joblib
import numpy as np
import pandas as pd
import argparse
from get_data import read_params

params_path = "params.yaml"

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict_testdata(config_path):
    try:
        config = read_params(config_path)
        model_path = config["webapp_model_dir"]
        test_df = config["split_data"]["test_path"]
        out_path = config["test_prediction"]["output_path"]
        model = joblib.load(model_path)
        prediction = model.predict(test_df)
        test_df['item_outlet_sales'] = prediction
        test_df.to_csv(out_path)

    except Exception as e:
        raise Exception("(predict_testdata): " + str(e))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    predict_testdata(config_path=parsed_args.config)