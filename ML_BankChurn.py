#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Case Study 1
# ### Bank Churning prediction
#

# In[30]:


import os
import io
import boto3
import re
from sagemaker import get_execution_role

# In[31]:


role = get_execution_role()
bucket = 'fwdinsurance.poc'
prefix = 'meetup_datasets/Churn_Modelling.csv'

# In[32]:


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

# In[36]:


print("https://s3.ap-south-1.amazonaws.com/{}/{}".format(bucket, prefix))
df = pd.read_csv("https://s3.ap-south-1.amazonaws.com/{}/{}".format(bucket, prefix))
dim = df.shape
print(dim)
dataset.head()

# In[37]:


# Splitting features IV Vs. Dependent Variable DV
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# In[38]:


# Describe dataset of Numerical values with core stats
df.describe()

# In[39]:


features = pd.get_dummies(features)
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)

# In[40]:


# One Hot Encoding for real !!!
features


# In[41]:


def normalization(data):
    mean, std = data.mean(), data.std()
    data = (data - mean) / std
    print(mean)
    print(std)
    return data


# In[43]:


import seaborn as sns

g = sns.PairGrid(df.iloc[:, :], hue='Exited')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# In[44]:


features = normalization(features)

# In[45]:


model = sm.OLS(labels, features)
results = model.fit()
print(results.summary())

# In[57]:


prng = np.random.randint(0, 200)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=prng)
dont_use0, X_val, dont_use_1, y_val = train_test_split(features, labels, test_size=0.10, random_state=prng)

# In[59]:


X_train = X_train.values
X_val = X_val.values

# In[61]:


train_file = "class_train.data"

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, X_train.astype('float32'), y_train.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)

# ## TRAINING
# <p>
#     MODEL: <b>Linear Learner</b>   <br>
#     PREDICTOR: <b>Classifier</b>   <br>
#     ERROR MECH: <B>Logistic</B>
# </p>
#

# In[ ]:


linear_job = 'ML-POC-T2-linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
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
        "InstanceType": "ml.p2.xlarge",
        "VolumeSizeInGB": 5
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
        "feature_dim": str(dim[0]),
        "mini_batch_size": "100",
        "predictor_type": "binary_classifier",
        "epochs": "10",
        "num_models": "32",
        "loss": "logistic"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}

print("CONTAINER: ", container)
print("BUCKET: ", bucket)
print("PREFIX: ", prefix)
