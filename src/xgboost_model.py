# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import xgboost as xgb

from google.cloud import storage
from google.cloud import bigquery




def build_train_model(project_id,
                        temp_bucket,
                        bq_dataset,
                        bq_table,
                        bq_sql_extract,
                        model_output_bucket):

    logging.info("Start exec ...")    
    client = bigquery.Client()
    destination_uri = "gs://{}/{}.csv".format(temp_bucket, bq_table)                    
    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)

    extract_job = client.extract_table(table_ref,destination_uri,location="US")  
    extract_job.result()  
    
    CATEGORICAL_COLUMNS = (
    'c6',
    'c7',
    'c8',
    'c9',
    'c10',
    'c12',
    'c13',
    'c16',
    'c18')

    TARGET_VAR = 'c23'

    raw_training_data = pd.read_csv(destination_uri)
    train_features = raw_training_data[['c3',
    'c5',
    'c6',
    'c7',
    'c8',
    'c9',
    'c10',
    'c12',
    'c13',
    'c16',
    'c18',
    'c23']]
    train_labels = raw_training_data[TARGET_VAR]
    encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        train_features[col] = encoders[col].fit_transform(train_features[col])
    
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=1)
    

    clf = xgb.XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200)
    clf.fit(X_train, y_train)
    print(clf)
    print(accuracy_score(y_test, clf.predict(X_test)))

    
    model = 'xgboost_model.bst'
    clf.save_model(model)


    bucket = storage.Client().bucket(model_output_bucket)
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('xgboost_%Y%m%d_%H%M%S'),model))
    blob.upload_from_filename(model)
    print("Model Exported {}".format(model_output_bucket)
    
    


)



def main(params):
    build_train_model(params.project_id,
                        params.temp_bucket,
                        params.bq_dataset,
                        params.bq_table,
                        params.bq_sql_extract,
                        params.model_output_bucket)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model from BQ Table')
    parser.add_argument('--project_id', type=str)
    parser.add_argument('--temp_bucket', type=str)
    parser.add_argument('--bq_dataset', type=str)
    parser.add_argument('--bq_table', type=str)
    parser.add_argument('--bq_sql_extract', type=str)
    parser.add_argument('--model_output_bucket', type=str)
    params = parser.parse_args()
    main(params)
