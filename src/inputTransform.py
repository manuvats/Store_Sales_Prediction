from datetime import date
import os
import argparse
import pandas as pd
#from get_data import read_params
#from one_hot_encoding import one_hot_encode

def getItemType(Item_Identifier):
    try:
        temp = Item_Identifier.str[:2]
        item_id = temp.map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
        return item_id
    except Exception as e:
        raise Exception(f"(itemID): Something went wrong in transforming itemID\n" + str(e))

def getOutletYrs(outlet_established_year):
    try:
        outlet_yrs = date.today().year - int(outlet_established_year)
        return outlet_yrs
    except Exception as e:
        raise Exception(f"(outletYrs): Something went wrong in transforming Outlet Established Year\n" + str(e))

def getTrainDummies(columns, config, input_df):
    try:
        #cols = one_hot_encode(config)["columns"]        
        #config = read_params(config_path)

        #input_df['item_type_combined'] = getItemType(input_df['item_identifier'])
        #input_df['outlet_years'] = getOutletYrs(input_df['outlet_year'])

        #input_df = input_df.drop(['item_identifier', 'outlet_year'], axis=1)
        input_df[['item_weight', 'item_visibility', 'item_mrp', ]] = input_df[['item_weight', 'item_visibility', 'item_mrp', ]].apply(pd.to_numeric)

        id_columns = config["one_hot_encoding"]["id_columns"]
        target = config["one_hot_encoding"]["target"]
        dummy_cols = [x for x in input_df.columns if x not in [target]+id_columns if input_df[x].dtype=='O']
        #print("Dummy columns: ")
        #print(dummy_cols)

        # Get missing columns in the training test
        input_df = pd.get_dummies(input_df, columns = dummy_cols)
        missing_cols = set(columns ) - set(input_df.columns)
        #print("Missing columns: ")
        #print(missing_cols)

        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            input_df[c] = 0

        # Ensure the order of column in the test set is in the same order than in train set
        input_df = input_df[columns]
        #input_df = input_df.reindex(columns=columns).fillna(0)
        #input_df.index = columns
        #input_df[dummy_cols] = enc.fit_transform(input_df[dummy_cols])

        return input_df
    except Exception as e:
        raise Exception("(getTrainDummies): Something went wrong in inputTransform\n" + str(e))

'''if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    getTrainDummies(config_path = parsed_args.config, input_df)'''