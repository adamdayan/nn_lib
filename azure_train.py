import os

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.train.dnn import PyTorch
from azureml.core.compute import AmlCompute

ws = Workspace.from_config()
print("Workspace details:")
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')
script_folder = os.getcwd()

# Configure run configuration 
packages = ['numpy', 'scikit-learn', 'pytorch']

run_remote = RunConfiguration()
run_remote.environment.python.conda_dependencies = \
    CondaDependencies.create(conda_packages=packages)

# Create experiment
experiment_name = 'my_experiment'
exp = Experiment(workspace=ws, name=experiment_name)

# Create PyTorch experiment
compute_name = "gpu-nc6-1"

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
    else:
        print("compute target not found")

script_params = {}

pt_est = PyTorch(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target,
                 entry_script='hello_world.py',
                 use_gpu=True)

# Submit PyTorch experiment
run = exp.submit(pt_est)
run.wait_for_completion(show_output=True)
