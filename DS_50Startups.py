#!/usr/bin/env python
# coding: utf-8

# ## Data Science Case Study 1
# ### 50 Startups - Investor's problem
#

# In[34]:


import os
import io
import boto3
import re
from sagemaker import get_execution_role

# Execution role will be used while writing configs into sagemaker

# In[35]:


role = get_execution_role()
bucket = 'fwdinsurance.poc'
prefix = 'meetup_datasets/50_Startups.csv'

# Importing necessary libraries
# * Data Wrangling : Pandas, Numpy
# * Data Preprocessing : Sklearn
# * Data Visualization : Matplotlib

# In[36]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sagemaker.amazon.common as smac
import statsmodels.formula.api as sm

# In[37]:


s3 = boto3.resource('s3')
# data_location = 's3://{}/{}'.format(bucket, prefix)
dataset = pd.read_csv("https://s3.ap-south-1.amazonaws.com/{}/{}".format(bucket, prefix))
print(dataset.head())
dataset.shape

# ### PREPROCESSING DATASET
# * Getting specific features in required format and in standardized (or normalized) form
# * Encoding Categorical varaibles

# In[38]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
experiment_X = dataset.iloc[:, :-1].values
experiment_y = dataset.iloc[:, 4].values

# In[39]:


# Describe dataset of Numerical values with core stats
df.describe()

# In[40]:


# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# In[41]:


# Removing Dummy Variable Trap
X = X[:, 1:]

# In[52]:


# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split

prng = np.random.randint(0, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=prng)
dont_use0, X_val, dont_use_1, y_val = train_test_split(X, y, test_size=0.10, random_state=prng)

# In[53]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set
y_pred = regressor.predict(X_test)

# In[54]:


# R Squared Score
regressor.score(X, y) * 100

# In[61]:


X_train.shape

# ## Prepare for Sagemaker Setup

# In[55]:


train_file = "class_train.data"

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, X_train.astype('float32'), y_train.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)

# In[56]:


val_file = "class_val.data"

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, X_val.astype('float32'), y_val.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', val_file)).upload_fileobj(f)

# ## TRAINING
# <p>
#     MODEL: <b>Linear Learner</b>   <br>
#     PREDICTOR: <b>Regressor</b>    <br>
#     ERROR MECH: <B>Logistic</B>
# </p>
#

# In[62]:


# See 'Algorithms Provided by Amazon SageMaker: Common Parameters' in the SageMaker documentation for an explanation of these values.
from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'linear-learner')
predictor = "regressor"

# In[67]:


linear_job = 'DS-usecase1-linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("JOB NAME:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m4.xlarge",
        "VolumeSizeInGB": 3
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }

    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
    },
    "HyperParameters": {
        "feature_dim": "5",
        "mini_batch_size": "10",
        "predictor_type": predictor,
        "epochs": "10",
        "num_models": "auto",
        "loss": "squared_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}

print("CONTAINER: ", container)
print("BUCKET: ", bucket)
print("PREFIX: ", prefix)

# In[68]:


get_ipython().run_cell_magic('time', '',
                             "region = boto3.Session().region_name\nsm = boto3.client('sagemaker')\n\nsm.create_training_job(**linear_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']\nprint(status)\n\nsm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)\n\nif status == 'Failed':\n    message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']\n    print('Training failed with the following error: {}'.format(message))\n    raise Exception('Training job failed')\n    \n\nprint(region)\nprint(sm)")

# In[71]:


linear_hosting_container = {
    'Image': container,
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=linear_job,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_hosting_container)

print(create_model_response['ModelArn'])

# ### ENDPOINT CREATION
# Service based API

# In[72]:


linear_endpoint_config = 'DS-usecase1-linear-EP-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': linear_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

# In[73]:


get_ipython().run_cell_magic('time', '',
                             '\nlinear_endpoint = \'DS-linear-sqloss-endpoint-uc1\'\nprint(linear_endpoint)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=linear_endpoint,\n    EndpointConfigName=linear_endpoint_config)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=linear_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nsm.get_waiter(\'endpoint_in_service\').wait(EndpointName=linear_endpoint)\n\nresp = sm.describe_endpoint(EndpointName=linear_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)\n\nif status != \'InService\':\n    raise Exception(\'Endpoint creation did not succeed\')')


# ### Checking Accuracy
# Method : Mean Absolute Error

# In[74]:


def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


# In[75]:


import json

runtime = boto3.client('runtime.sagemaker')

payload = np2csv(X_test)
response = runtime.invoke_endpoint(EndpointName=linear_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)
result = json.loads(response['Body'].read().decode())
test_pred = np.array([r['score'] for r in result['predictions']])

# In[76]:


test_pred_class = (test_pred > 0.5) + 0;
test_pred_baseline = np.repeat(np.median(y_train), len(y_test))

prediction_accuracy = np.mean((y_test == test_pred_class)) * 100
baseline_accuracy = np.mean((y_test == test_pred_baseline)) * 100

print("Prediction Accuracy:", round(prediction_accuracy, 1), "%")
print("Baseline Accuracy:", round(baseline_accuracy, 1), "%")

# In[78]:


test_pred_class

# ## Solution

# In[84]:


# Building Optimal Model for Backward Elimination Model
import statsmodels.formula.api as sm

# Add unit column for constant value to sustain the in records
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# In[85]:


X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# In[86]:


X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# In[83]:


y_pred1 = regressor_OLS.predict(X_opt)
y_pred1
