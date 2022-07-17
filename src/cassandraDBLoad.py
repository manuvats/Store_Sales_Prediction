#from sqlite3 import connect
from flask import session
import random
import yaml
import pandas as pd
import argparse
from get_data import read_params, get_data
import cassandra
from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, BatchStatement
import sys
import time

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, u"█"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def cassandraDBLoad(config_path):
    try:
        config = read_params(config_path)

        execution_profile = ExecutionProfile(request_timeout=10)
        cassandra_config = {'secure_connect_bundle': config["connect_cassandra"]["cloud_config"]}
        auth_provider = PlainTextAuthProvider(
                config["connect_cassandra"]["client_id"],
                config["connect_cassandra"]["client_secret"]
                )
        cluster = Cluster(cloud=cassandra_config, auth_provider=auth_provider)
        session = cluster.connect()
        session.default_timeout = None
        connect_db = session.execute("select release_version from system.local")
        #print(connect_db)
        set_keyspace = session.set_keyspace(config["cassandra_db"]["keyspace"])
        #print(use_keyspace)
        #print(cluster.protocol_version)
        table_ = config["cassandra_db"]["data_table"]
        define_columns = config["cassandra_db"]["define_columns"]
        #drop_table = f"DROP TABLE IF EXISTS {table_}"
        #start_drop = time.process_time()
        #drop_result = session.execute(drop_table)
        #print("Time taken to drop the table - " + str(time.process_time() - start_drop))

        create_table = f"CREATE TABLE IF NOT EXISTS {table_}({define_columns});"
        start_create = time.process_time()
        table_result = session.execute(create_table)
        #print("Time taken to create the table - " + str(time.process_time() - start_create))
        #print(train_result)
        
        train = pd.read_csv(config["data_source"]["train_source"])
        test = pd.read_csv(config["data_source"]["test_source"])
        #train, test = get_data(config_path)
        
        #Combine test and train into one file
        train['source']='train'
        test['source']='test'
        df = pd.concat([train, test],ignore_index=True)
        df = df.fillna('NA')
        print(df.shape)
        columns = list(df)
        for col in columns:
            df[col] = df[col].map(str)
        #print(df.isnull().sum())
        columns = config["cassandra_db"]["columns"]
        insert_qry = f"INSERT INTO {table_}({columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) IF NOT EXISTS"
        statement = session.prepare(insert_qry)
        #print(insert_qry)
        #print(tr_prepared)
        #print(prepared)
        #print(create_table)
        start_insert = time.process_time()
        batch = BatchStatement()
        for i in progressbar(range(len(df)), "Inserting: ", 40):
            time.sleep(0.1)            
            session.execute_async(
                statement,
                    [
                        df.iat[i,0], df.iat[i,1], df.iat[i,2], df.iat[i,3], df.iat[i,4], df.iat[i,5], 
                        df.iat[i,6], df.iat[i,7], df.iat[i,8], df.iat[i,9], df.iat[i,10], df.iat[i,11], 
                        df.iat[i,12]
                    ]
                )
            #session.execute_async(batch)
        print("Time taken to insert df - " + str((time.process_time() - start_insert)/60) + " minutes")

        '''new_cols = [col.replace(" ", "_") for col in df.columns]
        raw_data_path = config["load_data"]["raw_data_path"]
        df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)'''

    except Exception as e:
        raise Exception("(cassandraDBLoad): Something went wrong in the CassandraDB Load operations\n" + str(e))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args, unknown = args.parse_known_args()
    cassandraDBLoad(config_path = parsed_args.config)