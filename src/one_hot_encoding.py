# create dummy variables of categorical data
# save it in data/processed folder
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from get_data import read_params
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def one_hot_encode(config):
    try:
        #config = read_params(config_path)
        
        data = pd.read_csv(config["preprocess_data"]["processed_data"])
        
        id_columns = config["one_hot_encoding"]["id_columns"]
        target = config["one_hot_encoding"]["target"]
        source = config["one_hot_encoding"]["source"]
        dummy_cols = [x for x in data.columns if x not in [target]+id_columns+[source] if data[x].dtype=='O']

        #print("id_columns")
        #print(target)
        #print(dummy_cols)
        #print(data.dtypes)
        #enc = OneHotEncoder(sparse=False)
        #print(enc.fit_transform(data[[dummy_cols]]).toarray())
        data = pd.get_dummies(data, columns = dummy_cols)
        data.drop(['item_identifier','outlet_identifier'], axis=1, inplace=True)
        #col_list = data.columns.tolist()
        data_path = config["one_hot_encoding"]["data_path"]
        data.to_csv(data_path, sep=",", index=False, encoding = "utf-8")
        #data[data.isnull().any(axis=1)].to_csv("data/one_hot_encoded/null_data.csv", sep=",", index=False, encoding = "utf-8") 
    
    except Exception as e:
        raise Exception("(one_hot_encoding): Something went wrong in one_hot_encoding data\n" + str(e))

if __name__=="__main__":
    from get_data import read_params
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    config = read_params(parsed_args.config)
    one_hot_encode(config)


        