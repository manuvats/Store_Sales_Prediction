## read params
## process
## return dataframe

import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    #print(config)
    train_data_path = config["data_source"]["train_source"]
    test_data_path = config["data_source"]["test_source"]
    
    train_df = pd.read_csv(train_data_path, sep=",", encoding = "utf-8")
    test_df = pd.read_csv(test_data_path, sep=",", encoding = "utf-8")
    return train_df, test_df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    get_data(config_path = parsed_args.config)