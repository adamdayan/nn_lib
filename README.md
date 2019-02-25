# README

## How to train the PyTorch models on Azure

### Step 1 - create a virtualenv env on the lab machines so that you
    can install azureml

Create new environment (use site packages to get majority of reqs from
Nuric bitbucket):
```
virtualenv -p /usr/bin/python3.6 --system-site-packages venv_azure

```

Activate new environment:
```
venv_azure/bin/activate
```
**NB**: add this folder to .gitgnore immediately as it's huge. Also
add it to .amlignore (see below)

Install azureml:
```
pip install azureml
```
**NB**: this is a big package, so if this step fails it might be
because you've exceeded the file quota.

### Step 2 - running the training on azure GPU

While inside your azure venv run:
```
python3 azure_train.py
```

Notes:
* You need a config.json file in your root directory with details of
the resource group (configure this on the azure portal)
* You can change the script that is run on azure my modifying the
azure_train.py file
* Add any files that are not needed on azure (e.g. venv folder) to
.amlignore. File limit to transfer to azure is 2000
