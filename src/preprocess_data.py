import yaml
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import argparse
from get_data import read_params
import cassandra
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)
    
def preprocess_data(config_path):
    try:
        config = read_params(config_path)
        cassandra_config = {'secure_connect_bundle': config["connect_cassandra"]["cloud_config"]}
        auth_provider = PlainTextAuthProvider(
                config["connect_cassandra"]["client_id"],
                config["connect_cassandra"]["client_secret"]
                )
        cluster = Cluster(cloud=cassandra_config, auth_provider=auth_provider)
        session = cluster.connect()
        session.set_keyspace(config["cassandra_db"]["keyspace"])
        session.row_factory = pandas_factory
        session.default_fetch_size = None

        count_query = f"SELECT COUNT(*) from {config['cassandra_db']['data_table']} LIMIT 20000"
        count_rslt = session.execute(count_query, timeout=None)
        print(count_rslt._current_rows)
        query = f"SELECT * from {config['cassandra_db']['data_table']}"
        simple_statement = SimpleStatement(query, consistency_level=ConsistencyLevel.ONE, fetch_size=None)
        execute_result = session.execute(simple_statement, timeout=None)
        data = execute_result._current_rows
        #rslt = session.execute(query, timeout=None)
        #data = result._current_rows
        #print(data.shape)
        data = data.replace('NA', np.nan)
        #Making 0 values as missing in Item_Visibility
        data["item_visibility"][data['item_visibility'] == 0] = np.nan

        #Create a broad category of Type of Item
        data['item_identifier'].value_counts()
        data['item_type_combined'] = data['item_identifier'].apply(lambda x: x[0:2])
        data['item_type_combined'] = data['item_type_combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
        
        # Determine the years of operation of a store
        data['outlet_years'] = date.today().year - pd.to_numeric(data['outlet_establishment_year'])

        # Modify categories of Item_Fat_Content
        data['item_fat_content'] = data['item_fat_content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
                                                        
        #Mark non-consumables as separate category in low_fat:
        data.loc[data['item_type_combined']=="Non-Consumable",'item_fat_content'] = "Non-Edible"

        #print(data[data['item_weight'] == 'NA'].shape[0])

        data[['item_mrp', 'item_outlet_sales', 'item_visibility', 'item_weight']] = data[['item_mrp', 'item_outlet_sales', 'item_visibility', 'item_weight']].astype(float)
        #Label Encoding to impute missing values For regression, SVC, etc.
        le = LabelEncoder()
        series = data['outlet_size']
        data['outlet_size'] = pd.Series(le.fit_transform(series[series.notnull()]), 
                                        index=series[series.notnull()].index)
        
        data[['item_mrp', 'item_outlet_sales', 'item_visibility', 'item_weight']] = data[['item_mrp', 'item_outlet_sales', 'item_visibility', 'item_weight']].astype(float)
        #data = data.astype(convert_dict)
        #print(data.dtypes)
        # Using Knn imputer to impute missing values instead of using mean, mode, median, etc.
        imputer = KNNImputer()
        data.loc[:,(data.dtypes=='float64').values] = imputer.fit_transform(data.loc[:,(data.dtypes=='float64').values])

        #Rounding off the imputed values
        data['outlet_size'] = np.round(data['outlet_size'])

        #Inverse transforming the encoded values
        data['outlet_size'] = le.inverse_transform(data['outlet_size'].astype(int))

        #Drop the columns which have been converted to different types:
        data.drop(['item_type','outlet_establishment_year'],axis=1,inplace=True)

        #Divide into test and train:
        #train = data.loc[data['source']=="train"]
        #test = data.loc[data['source']=="test"]

        #Drop unnecessary columns:
        #test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
        #train.drop(['source'],axis=1,inplace=True)

        data_processed_path = config["preprocess_data"]["processed_data"]
        #train_processed_path = config["preprocess_data"]["processed_train_data"]
        #test_processed_path = config["preprocess_data"]["processed_test_data"]

        data.to_csv(data_processed_path, sep=",", index=False, encoding = "utf-8")
        #train.to_csv(train_processed_path, sep=",", index=False, encoding = "utf-8")
        #test.to_csv(test_processed_path, sep=",", index=False, encoding = "utf-8")
        
    except Exception as e:
        raise Exception("(preprocessData): " + str(e))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    preprocess_data(config_path = parsed_args.config)
