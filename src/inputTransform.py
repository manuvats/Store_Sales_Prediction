from datetime import date
import os
import argparse
import pandas as pd
from get_data import read_params
from one_hot_encoding import one_hot_encode

def getItemType(Item_Identifier):
    try:
        Item_Identifier = Item_Identifier[0:2]
        return Item_Identifier
    except Exception as e:
        raise Exception(f"(itemID): Something went wrong in transforming itemID\n" + str(e))

def getOutletYrs(outlet_established_year):
    try:
        outlet_yrs = date.today().year - outlet_established_year
        return outlet_yrs
    except Exception as e:
        raise Exception(f"(outletYrs): Something went wrong in transforming Outlet Established Year\n" + str(e))

def getTrainDummies(config_path, input_df):
    try:
        enc = one_hot_encode(config_path)        
        config = read_params(config_path)

        input_df['Item_Identifier'] = getItemType(input_df['Item_Identifier'])
        input_df['Outlet_Years'] = getOutletYrs(input_df['Outlet_Years'])
        id_columns = config["one_hot-encoded"]["id_columns"]
        target = config["one_hot-encoded"]["target"]
        dummy_cols = [x for x in input_df.columns if x not in [target]+id_columns if input_df[x].dtype=='O']
        
        input_df[dummy_cols] = enc.fit_transform(input_df[dummy_cols])

        return input_df
    except Exception as e:
        raise Exception("(getTrainDummies): Something went wrong in creating dummies fo input data\n" + str(e))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    getTrainDummies(config_path = parsed_args.config, input_df)