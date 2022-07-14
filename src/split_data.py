# split the raw data
# save it in data/processed folder

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def split_and_saved_data(config_path):
    try:
        config = read_params(config_path)
         
        train_data_path = config["split_data"]["train_path"]
        test_data_path = config["split_data"]["test_path"]
        val_data_path = config["split_data"]["validation_path"]
        
        input_data_path = config["one_hot_encoding"]["data_path"]
        
        split_ratio = config["split_data"]["val_size"]
        random_state = config["base"]["random_state"]
        
        data = pd.read_csv(input_data_path, sep=",")
        
        #Divide into test and train:
        train = data.loc[data['source']=="train"].copy()
        test = data.loc[data['source']=="test"].copy()

        #Drop unnecessary columns:
        test.drop(['item_outlet_sales','source'],axis=1,inplace=True)
        train.drop(['source'],axis=1,inplace=True)

        train, val = train_test_split(
            train, 
            test_size=split_ratio, 
            random_state=random_state
        )
        train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
        test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
        val.to_csv(val_data_path, sep=",", index=False, encoding = "utf-8")

    except Exception as e:
        raise Exception("(split_and_saved_data): Something went wrong in the splitting of data\n" + str(e))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path = parsed_args.config)