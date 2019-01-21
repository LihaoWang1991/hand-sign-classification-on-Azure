Like Google Cloud in Google and AWS in Amazon, Azure is the cloud computing platform provided by Microsoft. Its main services include computing, mobile services, storage services, data management, media services, machine learning, IoT, etc. 


[Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml) provides a cloud-based environment you can use to develop, train, test, deploy, manage, and track machine learning models. The process of a typical Azure machine learning is as below:

<img src="https://i.postimg.cc/wjsrnbjk/post-azure2.png" style="width:600px;">

Like most machine learning platforms, the supported language on Azure Machine Learning service is Python. It fully supports open-source technologies. That means you can use open-source Python packages such as TensorFlow and scikit-learn. If you are familiar with coding using Jupyter Notebook, then Azure Machine Learning service can be a good choice to you because it has the same programming interface which is called [Azure Notebooks](https://notebooks.azure.com/). Nevertheless, you can also code on your local Python IDE but you need to install [Azure Python SDK](https://docs.microsoft.com/zh-cn/python/api/overview/azure/ml/intro?view=azure-ml-py) packages at first. 

After learning the [official tutorials](https://docs.microsoft.com/en-us/azure/machine-learning/service/), I have migrated a previous project on Azure Machine Learning service. I will show you this project step by step.

## Creating an Azure machine learning workspace

The project begins by creating an Azure Machine Learning Workspace. 

The workspace is the top-level resource for Azure Machine Learning service. It provides a centralized place to work with all the artifacts you create when you use Azure Machine Learning service. The workspace keeps a list of compute targets that you can use to train your model. It also keeps a history of the training runs, including logs, metrics, output, and a snapshot of your scripts. You use this information to determine which training run produces the best model. 

The workspace is created on [Azure Portal](portal.azure.com) as below:

<img src="https://i.postimg.cc/wxtDfhnH/post-azure1.png" style="width:600px;">

**Resource group** means the container that holds related resources for an Azure solution.The machine learning workspace must be allocated to one resource group.

In the created workspace, we can see its main functions such as Experiments, Pipelines, Compute, Deployments, etc. After clicking the the button **Open Azure Notebooks**, the page [Azure Notebooks](https://notebooks.azure.com/) will open so that you can configure the workspace, create and run your machine learning model by coding in it.

<img src="https://i.postimg.cc/tCGxqT3P/post-azure3.png" style="width:600px;">

## Set up the development environment
Now we will set up the development environment Azure Notebook. You can find all the files used in this project in my [Azure Notebook](https://notebooks.azure.com/lihaowang/projects/handsignclassification) and [this](https://notebooks.azure.com/lihaowang/projects/handsignclassification/html/Model/hand-sign-classification.ipynb) is a direct link to the related notebook page. 

Setup includes the following actions:

* **Import Python packages.** 
* **Connect to a workspace, so that your local computer can communicate with remote resources.**
* **Create an experiment to track all your runs.**
* **Create a remote compute target to use for training.**

#### Import packages
Import Python packages we need in this session. 

```
%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import azureml
from azureml.core import Workspace, Run
```
#### Connect to a workspace
Before this step, we need to put a file called **config.json** in the current folder as below.  

![](https://i.postimg.cc/vTgfpB7Y/post-azure4.png)

**Config.json** stores the necessary information that Azure notebook needs to connect to the workspace that we have just created. In this project **config.json** contains only 3 lines.

```
 "subscription_id": "65c2cf5a-b718-4bd4-9031-b7a2f2881ff0",
 "resource_group": "docs-aml",
 "workspace_name": "docs-ws"
```

Then we can go back to Azure Notebook and create a workspace object from the existing workspace. `Workspace.from_config()` reads the file **config.json** and loads the details into an object named ws:

```
# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep = '\t')
```
#### Create an experiment
Create an experiment to track the runs in the workspace. A workspace can have multiple experiments:

```
experiment_name = 'CNN-handsign'
from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)
```

#### Create or attach an existing AMlCompute
By using Azure Machine Learning Compute (AmlCompute), a managed service, we can train machine learning models on clusters of Azure virtual machines. Examples include VMs with GPU support. In this project, we create AmlCompute as our training environment. This code creates the compute clusters for us if they don't already exist in our workspace.
Creation of the compute takes about five minutes. If the compute is already in the workspace, this code uses it and skips the creation process:

```
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

     # For a more detailed view of current AmlCompute status, use the 'status' property    
    print(compute_target.status.serialize())
```

We now have the necessary packages and compute resources to train a model in the cloud.

## Machine learning model and data preparation
In his project we will implement a ConvNet using TensorFlow to classify the hand sign images into 6 classes: number 0 to number 5. This was one of the programming assignments I have completed in Andrew Ng's course [Deep Learning](https://www.coursera.org/specializations/deep-learning).  

<img src="https://i.postimg.cc/437QtzfP/SIGNS.png" style="width:800px;">

[CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network), or ConvNet is a class of deep neural networks, most commonly applied to analyzing visual imagery. In this session we will do some preparation work such as loading and transforming data.
#### Data preparation
The following code load and normalize the data. 

```
import math
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalization and one hot conversion
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
```

The package cnn_utils contains the functions such as load data and one hot conversion. 

#### Upload data to the cloud
Now we will make the data accessible remotely by uploading that data from local machine into Azure. Then it can be accessed for remote training. The datastore is a convenient construct associated with our workspace for us to upload or download data. We can also interact with it from your remote compute targets. It's backed by an Azure Blob storage account.
The SIGNS files are uploaded into a directory named signs at the root of the datastore.

```
ds = ws.get_default_datastore()
ds.upload(src_dir='./datasets', target_path='signs', overwrite=True, show_progress=True)
```

## Train on a remote cluster
#### Create a directory
We create a directory to deliver the necessary code from local computer to the remote resource:

```
import os
script_folder = './CNN-handsign'
os.makedirs(script_folder, exist_ok=True)
```

#### Create a training script
To submit the job to the cluster, we need to create a training script. For the entire codes including tensorflow model functions please refer to my [Azure Notebook](https://notebooks.azure.com/lihaowang/projects/handsignclassification/html/Model/hand-sign-classification.ipynb). Here I will only show the codes related to Azure machine learning model. 

The following script lets user feed in 1 parameter, the location of the data files (from datastore):
```
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()
```

The following script saves the training metrics to the run log:
```
run.log('train_accuracy', np.float(train_accuracy))
run.log('test_accuracy', np.float(test_accuracy))
```

The following script saves our model into a directory named **outputs**: 
```
os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
f = open('outputs/hand-sign-classification.pkl','wb')
pickle.dump(final_param,f)
f.close()
```

Anything written in directory **outputs** is automatically uploaded into our workspace. We will access our model from this directory later in this project. The file cnn_utils.py is referenced from the training script to load the dataset correctly. Now we copy this script into the script folder, so that it can be accessed along with the training script on the remote resource.

```
import shutil
shutil.copy('cnn_utils.py', script_folder)
```

#### Create an estimator

An estimator object is used to submit the run. We create the estimator by running the following code.
```
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
}

est = Estimator(source_directory=script_folder,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['tensorflow','matplotlib'])
```

For detailed explanation please refer to my [Azure Notebook](https://handsignclassification-lihaowang.notebooks.azure.com/j/notebooks/Model/hand-sign-classification.ipynb#). 

#### Submit the job to the cluster
The following code will run the experiment by submitting the estimator object:
```
run = exp.submit(config=est)
run
```

#### Monitor a remote run
So the machine learning model is now running on Azure!
In total, the first run takes about 15 minutes. But for subsequent runs, as long as the script dependencies don't change, the same image is reused. So the container startup time is much faster.
We can check the progress of a running job using the following code:
```
from azureml.widgets import RunDetails
RunDetails(run).show()
```
In Azure Notebook we can see the running progress every 10 to 15 seconds until the job finishes.
<img src="https://i.postimg.cc/WzYQmkJ7/post-azure6.png" style="width:800px;">

We now have a model trained on a remote cluster. We can print the accuracy of the model:

<img src="https://i.postimg.cc/wxrfVHGF/post-azure7.png" style="width:600px;">

Maybe you find the training and test accuracy is not very high. This is because due to limited computing resource I have applied on Azure, I have not trained the model for too many epochs.

And in Azure Portal, we can see the experiment running history as following:

<img src="https://i.postimg.cc/zGNGym8X/post-azure8.png" style="width:600px;">

## Register model
Now we register the model in the workspace, so that we or other collaborators can later query, examine, and deploy this model:
```
# register model 
model = run.register_model(model_name='hand-sign-classification', model_path='outputs/hand-sign-classification.pkl')
```

And now if we go to the **Model** column of the Azure machine learning workspace, we will see the registered model:
<img src="https://i.postimg.cc/x1jqsr7p/post-azure9.png" style="width:800px;">

So finally we have successfully trained a hand signs classification model on Azure machine learning workspace! You can find all the codes and data used in this project in my [Azure Notebook](https://notebooks.azure.com/lihaowang/projects/handsignclassification). 
