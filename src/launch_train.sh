#!/usr/bin/env bash
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
python3 ./xgboost_model.py --project_id 'velascoluis-test' --temp_bucket 'master_bucket_us' --bq_dataset 'master_dataset_us' --bq_table 'input_data_nc' --bq_sql_extract '' --model_output_bucket 'master_bucket_us'