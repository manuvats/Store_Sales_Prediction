from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from prediction_service import prediction
import argparse
import pandas as pd
import logging
from src import inputTransform, get_data, split_data
#from get_data import read_params
logging.basicConfig(filename='StoreSales_record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
 
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

args = argparse.ArgumentParser()
args.add_argument("--config", default = "params.yaml")
parsed_args = args.parse_args()
config = get_data.read_params(parsed_args.config)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                #print("Initial dictionary")
                #print(dict_req)
                
                input_df = pd.DataFrame.from_dict([dict_req])
                #print("After converting to df")
                #print(input_df['item_identifier'][0:2])
                #print(input_df.columns)
                #print(input_df.dtypes)
                input_df['outlet_years'] = inputTransform.getOutletYrs(input_df['outlet_estd_year'])
                input_df['item_type_combined'] = inputTransform.getItemType(input_df['item_identifier'])
                
                #print("After getting outlet yrs and item type")
                #print(input_df)
                #print(input_df.columns)
                input_df.drop(['item_identifier', 'outlet_estd_year'], axis=1, inplace=True)
                #print("After dropping item id and outlet estd yr")
                #print(input_df)
                #print(input_df.columns)
                

                columns = split_data.split_and_saved_data(config)
                #print("Columns got from split_data.py")
                #print(columns)
                transformed_input = inputTransform.getTrainDummies(columns, config, input_df)
                '''print("Columns after creating dummies")
                print("Item MRP" + str(transformed_input['item_mrp'][0]))
                print("Item Visibility" + str(transformed_input['item_visibility'][0]))
                print("Item Weight" + str(transformed_input['item_weight'][0]))
                print("Outlet Years" + str(transformed_input['outlet_years'][0]))
                print("Item Fat Low Fat" + str(transformed_input['item_fat_content_Low Fat'][0]))
                print("Non-Edible" + str(transformed_input['item_fat_content_Non-Edible'][0]))
                print("Regular" + str(transformed_input['item_fat_content_Regular'][0]))
                print("Tier1" + str(transformed_input['outlet_location_type_Tier 1'][0]))
                print("Tier2" + str(transformed_input['outlet_location_type_Tier 2'][0]))
                print("Tier3" + str(transformed_input['outlet_location_type_Tier 3'][0]))
                print("High" + str(transformed_input['outlet_size_High'][0]))
                print("Medium" + str(transformed_input['outlet_size_Medium'][0]))
                print("Small" + str(transformed_input['outlet_size_Small'][0]))
                print("Grocery Store" + str(transformed_input['outlet_type_Grocery Store'][0]))
                print("Type1" + str(transformed_input['outlet_type_Supermarket Type1'][0]))
                print("Type2" + str(transformed_input['outlet_type_Supermarket Type2'][0]))
                print("Type3" + str(transformed_input['outlet_type_Supermarket Type3'][0]))
                print("Drinks" + str(transformed_input['item_type_combined_Drinks'][0]))
                print("Food" + str(transformed_input['item_type_combined_Food'][0]))
                print("Non-Consumable" + str(transformed_input['item_type_combined_Non-Consumable'][0]))

                print(transformed_input.columns)'''
                response = prediction.form_response(transformed_input)
                #print(response)

                return render_template("index.html", response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.logger.debug("debug log info")
    app.logger.info("Info log information")
    app.logger.warning("Warning log info")
    app.logger.error("Error log info")
    app.logger.critical("Critical log info")