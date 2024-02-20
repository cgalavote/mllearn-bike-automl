# mllearn-bike-automl

Projeto para automação de aluguel de bicicletas utilizando o Microsoft Azure.

Para a execução desse projeto, foi utilizado a documentação do próprio site (https://aka.ms/ai900-auto-ml)


```json

{
    "runId": "mllearn-bike-automl_2",
    "runUuid": "4b647e52-5acb-447f-acb4-dc0fad013991",
    "parentRunUuid": "14ffc5a5-b5c2-498c-b0de-93012547760e",
    "rootRunUuid": "14ffc5a5-b5c2-498c-b0de-93012547760e",
    "target": "Serverless",
    "status": "Completed",
    "parentRunId": "mllearn-bike-automl",
    "dataContainerId": "dcid.mllearn-bike-automl_2",
    "createdTimeUtc": "2024-02-19T17:29:40.6333607+00:00",
    "startTimeUtc": "2024-02-19T17:30:02.700Z",
    "endTimeUtc": "2024-02-19T17:30:52.744Z",
    "error": null,
    "warnings": null,
    "tags": {
        "_aml_system_azureml.automlComponent": "AutoML",
        "mlflow.source.type": "JOB",
        "mlflow.source.name": "automl_driver.py",
        "_aml_system_codegen": "completed",
        "_aml_system_automl_is_child_run_end_telemetry_event_logged": "True"
    },
    "properties": {
        "runTemplate": "automl_child",
        "pipeline_id": "__AutoML_Ensemble__",
        "pipeline_spec": "{\"pipeline_id\":\"__AutoML_Ensemble__\",\"objects\":[{\"module\":\"azureml.train.automl.ensemble\",\"class_name\":\"Ensemble\",\"spec_class\":\"sklearn\",\"param_args\":[],\"param_kwargs\":{\"automl_settings\":\"{'task_type':'regression','primary_metric':'normalized_root_mean_squared_error','verbosity':20,'ensemble_iterations':3,'ensemble_download_models_timeout_sec':300,'is_timeseries':False,'subscription_id':'8be34f78-b2ca-45ca-aac9-7d1077442dcf','time_column_name':None,'grain_column_names':None}\",\"ensemble_run_id\":\"mllearn-bike-automl_2\",\"experiment_name\":\"mllearn-bike-automl\",\"workspace_name\":\"Laboratorioai900\",\"subscription_id\":\"8be34f78-b2ca-45ca-aac9-7d1077442dcf\",\"resource_group_name\":\"LAB-DIO\"}}]}",
        "training_percent": "100",
        "predicted_cost": null,
        "iteration": "2",
        "_aml_system_scenario_identification": "Remote.Child",
        "_azureml.ComputeTargetType": "amlctrain",
        "_azureml.ClusterName": "Serverless",
        "ContentSnapshotId": "b19e3f8a-c6be-4a46-bca8-6fed0157cdca",
        "ProcessInfoFile": "azureml-logs/process_info.json",
        "ProcessStatusFile": "azureml-logs/process_status.json",
        "run_preprocessor": "",
        "run_algorithm": "VotingEnsemble",
        "ensembled_iterations": "[0, 1]",
        "ensembled_algorithms": "['LightGBM', 'RandomForest']",
        "ensembled_run_ids": "['mllearn-bike-automl_0', 'mllearn-bike-automl_1']",
        "ensemble_weights": "[0.6666666666666666, 0.3333333333333333]",
        "best_individual_pipeline_score": "0.09357675586470318",
        "best_individual_iteration": "0",
        "model_output_path": "outputs/model.pkl",
        "conda_env_data_location": "aml://artifact/ExperimentRun/dcid.mllearn-bike-automl_2/outputs/conda_env_v_1_0_0.yml",
        "model_data_location": "aml://artifact/ExperimentRun/dcid.mllearn-bike-automl_2/outputs/model.pkl",
        "model_size_on_disk": "266801",
        "scoring_data_location": "aml://artifact/ExperimentRun/dcid.mllearn-bike-automl_2/outputs/scoring_file_v_1_0_0.py",
        "scoring_data_location_v2": "aml://artifact/ExperimentRun/dcid.mllearn-bike-automl_2/outputs/scoring_file_v_2_0_0.py",
        "scoring_data_location_pbi": "aml://artifact/ExperimentRun/dcid.mllearn-bike-automl_2/outputs/scoring_file_pbi_v_1_0_0.py",
        "model_exp_support": "True",
        "pipeline_graph_version": "1.0.0",
        "model_name": "mllearnbikeauto2",
        "score": "0.09030125619896053",
        "score_table": "NaN",
        "run_properties": "estimators=[('0', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True",
        "pipeline_script": "{\"pipeline_id\":\"__AutoML_Ensemble__\",\"objects\":[{\"module\":\"azureml.train.automl.ensemble\",\"class_name\":\"Ensemble\",\"spec_class\":\"sklearn\",\"param_args\":[],\"param_kwargs\":{\"automl_settings\":\"{'task_type':'regression','primary_metric':'normalized_root_mean_squared_error','verbosity':20,'ensemble_iterations':3,'ensemble_download_models_timeout_sec':300,'is_timeseries':False,'subscription_id':'8be34f78-b2ca-45ca-aac9-7d1077442dcf','time_column_name':None,'grain_column_names':None}\",\"ensemble_run_id\":\"mllearn-bike-automl_2\",\"experiment_name\":\"mllearn-bike-automl\",\"workspace_name\":\"Laboratorioai900\",\"subscription_id\":\"8be34f78-b2ca-45ca-aac9-7d1077442dcf\",\"resource_group_name\":\"LAB-DIO\"}}]}",
        "training_type": "train_valid",
        "fit_time": "2",
        "goal": "normalized_root_mean_squared_error_min",
        "primary_metric": "normalized_root_mean_squared_error",
        "errors": "{}",
        "onnx_model_resource": "{}",
        "dependencies_versions": "{\"azureml-dataprep-native\": \"38.0.0\", \"azureml-dataprep\": \"4.12.8\", \"azureml-dataprep-rslex\": \"2.19.6\", \"azureml-mlflow\": \"1.54.0.post1\", \"azureml-core\": \"1.54.0.post1\", \"azureml-train-automl-client\": \"1.54.0.post1\", \"azureml-train-automl-runtime\": \"1.54.0.post1\", \"azureml-dataset-runtime\": \"1.54.0.post1\", \"azureml-pipeline-core\": \"1.54.0\", \"azureml-telemetry\": \"1.54.0\", \"azureml-responsibleai\": \"1.54.0\", \"azureml-interpret\": \"1.54.0\", \"azureml-training-tabular\": \"1.54.0\", \"azureml-automl-runtime\": \"1.54.0\", \"azureml-train-core\": \"1.54.0\", \"azureml-train-restclients-hyperdrive\": \"1.54.0\", \"azureml-automl-core\": \"1.54.0\", \"azureml-defaults\": \"1.54.0\", \"azureml-inference-server-http\": \"0.8.4.2\"}",
        "num_cores": "2",
        "num_logical_cores": "4",
        "peak_memory_usage": "1118592",
        "vm_configuration": "x86_64",
        "core_hours": "0.005580656111111111",
        "feature_skus": "automatedml_sdk_guardrails"
    },
    "parameters": {},
    "services": {},
    "inputDatasets": [],
    "outputDatasets": [],
    "runDefinition": {
        "script": "automl_driver.py",
        "useAbsolutePath": true,
        "arguments": [],
        "sourceDirectoryDataStore": null,
        "framework": "Python",
        "communicator": "None",
        "target": "Serverless",
        "autoClusterComputeSpecification": {
            "instanceSize": "Standard_D4s_v3",
            "instancePriority": "Dedicated",
            "osType": null,
            "location": null,
            "runtimeVersion": null
        },
        "dataReferences": {},
        "data": {},
        "outputData": {},
        "datacaches": [],
        "jobName": null,
        "maxRunDurationSeconds": null,
        "nodeCount": 1,
        "maxNodeCount": 3,
        "instanceTypes": [],
        "priority": null,
        "credentialPassthrough": false,
        "identity": null,
        "environment": {
            "name": "AzureML-AutoML",
            "version": "156",
            "assetId": "azureml://registries/azureml/environments/AzureML-AutoML/versions/156",
            "autoRebuild": true,
            "python": {
                "interpreterPath": "python",
                "userManagedDependencies": true,
                "condaDependencies": null,
                "baseCondaEnvironment": null
            },
            "environmentVariables": {
                "EXAMPLE_ENV_VAR": "EXAMPLE_VALUE"
            },
            "docker": {
                "baseImage": null,
                "platform": {
                    "os": "Linux",
                    "architecture": "amd64"
                },
                "baseDockerfile": "FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04\n\nENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl\nENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH\n\nCOPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py\n\nENV MLFLOW_MODEL_FOLDER=\"mlflow-model\"\n# ENV AML_APP_ROOT=\"/var/mlflow_resources\"\n# ENV AZUREML_ENTRY_SCRIPT=\"mlflow_score_script.py\"\n\nENV ENABLE_METADATA=true\n\n# begin conda create\n# Create conda environment\nRUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \\\n    python=3.8 \\\n    # begin conda dependencies\n    pip=22.1.2 \\\n    numpy~=1.22.3 \\\n    py-cpuinfo=5.0.0 \\\n    joblib=1.2.0 \\\n    cloudpickle=1.6.0 \\\n    scikit-learn=1.1.3 \\\n    pandas~=1.3.5 \\\n    py-xgboost=1.3.3 \\\n    holidays=0.29 \\\n    setuptools-git \\\n    setuptools=65.5.1 \\\n    wheel=0.38.1 \\\n    pyopenssl=23.2.0 \\\n    'psutil>5.0.0,<6.0.0' \\\n    # end conda dependencies\n    -c conda-forge -c pytorch -c anaconda && \\\n    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \\\n    conda clean -a -y\n# end conda create\n\n# begin pip install\n# Install pip dependencies\nRUN pip install  \\\n                # begin pypi dependencies\n                'azureml-core==1.54.0.post1' \\\n                'azureml-mlflow==1.54.0.post1' \\\n                'azureml-pipeline-core==1.54.0' \\\n                'azureml-telemetry==1.54.0' \\\n                'azureml-interpret==1.54.0' \\\n                'azureml-responsibleai==1.54.0' \\\n                'azureml-automl-core==1.54.0' \\\n                'azureml-automl-runtime==1.54.0' \\\n                'azureml-train-automl-client==1.54.0.post1' \\\n                'azureml-train-automl-runtime==1.54.0.post1' \\\n                'azureml-dataset-runtime==1.54.0.post1' \\\n                'azureml-defaults==1.54.0' \\\n                'cryptography>=41.0.4' \\\n                'inference-schema' \\\n                'prophet==1.1.4' \\\n                'mltable>=1.0.0'\n                # end pypi dependencies\n# end pip install\n\n# dummy number to change when needing to force rebuild without changing the definition: 3",
                "baseImageRegistry": {
                    "address": null,
                    "username": null,
                    "password": null
                },
                "enabled": false,
                "arguments": []
            },
            "spark": {
                "repositories": [],
                "packages": [],
                "precachePackages": true
            },
            "inferencingStackVersion": null
        },
        "history": {
            "outputCollection": true,
            "directoriesToWatch": [
                "logs"
            ],
            "enableMLflowTracking": true
        },
        "spark": {
            "configuration": {}
        },
        "parallelTask": {
            "maxRetriesPerWorker": 0,
            "workerCountPerNode": 1,
            "terminalExitCodes": null,
            "configuration": {}
        },
        "amlCompute": {
            "name": "Serverless",
            "vmSize": null,
            "retainCluster": false,
            "clusterMaxNodeCount": 3
        },
        "aiSuperComputer": {
            "instanceType": "D2",
            "imageVersion": null,
            "location": null,
            "aiSuperComputerStorageData": null,
            "interactive": false,
            "scalePolicy": null,
            "virtualClusterArmId": null,
            "tensorboardLogDirectory": null,
            "sshPublicKey": null,
            "sshPublicKeys": null,
            "enableAzmlInt": true,
            "priority": "Medium",
            "slaTier": "Standard",
            "userAlias": null
        },
        "kubernetesCompute": {
            "instanceType": null
        },
        "tensorflow": {
            "workerCount": 0,
            "parameterServerCount": 0
        },
        "mpi": {
            "processCountPerNode": 0
        },
        "pyTorch": {
            "communicationBackend": null,
            "processCount": null
        },
        "hdi": {
            "yarnDeployMode": "None"
        },
        "containerInstance": {
            "region": null,
            "cpuCores": 2,
            "memoryGb": 3.5
        },
        "exposedPorts": null,
        "docker": {
            "useDocker": true,
            "sharedVolumes": true,
            "shmSize": "2g",
            "arguments": []
        },
        "cmk8sCompute": {
            "configuration": {}
        },
        "commandReturnCodeConfig": {
            "returnCode": "Zero",
            "successfulReturnCodes": []
        },
        "environmentVariables": {
            "AUTOML_SDK_RESOURCE_URL": "https://aka.ms/automl-resources/"
        },
        "applicationEndpoints": {},
        "parameters": []
    },
    "logFiles": {
        "logs/azureml/azureml_automl-child.log": "https://laboratorioai96124544063.blob.core.windows.net/azureml/ExperimentRun/dcid.mllearn-bike-automl_2/logs/azureml/azureml_automl-child.log?sv=2019-07-07&sr=b&sig=ikH1qm6OC%2BJN6xnhfwpPQVOXN7iUEfVc6qHqLKuuLy0%3D&skoid=030b60e8-c44b-42c2-8b5b-bfbb7e341a12&sktid=483d5369-8f79-479c-bb98-1b542eccc4a8&skt=2024-02-20T13%3A37%3A21Z&ske=2024-02-21T21%3A47%3A21Z&sks=b&skv=2019-07-07&st=2024-02-20T13%3A37%3A21Z&se=2024-02-20T21%3A47%3A21Z&sp=r",
        "logs/azureml/azureml_automl.log": "https://laboratorioai96124544063.blob.core.windows.net/azureml/ExperimentRun/dcid.mllearn-bike-automl_2/logs/azureml/azureml_automl.log?sv=2019-07-07&sr=b&sig=FmF7LEyDsHbubCB%2BcrxY0Mqmw2tVTTx9mHlgVTBF4Tw%3D&skoid=030b60e8-c44b-42c2-8b5b-bfbb7e341a12&sktid=483d5369-8f79-479c-bb98-1b542eccc4a8&skt=2024-02-20T13%3A37%3A21Z&ske=2024-02-21T21%3A47%3A21Z&sks=b&skv=2019-07-07&st=2024-02-20T13%3A37%3A21Z&se=2024-02-20T21%3A47%3A21Z&sp=r"
    },
    "jobCost": {
        "chargedCpuCoreSeconds": 188,
        "chargedCpuMemoryMegabyteSeconds": null,
        "chargedGpuSeconds": null,
        "chargedNodeUtilizationSeconds": 5004.41382
    },
    "revision": 19,
    "runTypeV2": {
        "orchestrator": "Execution",
        "traits": [
            "automl",
            "Remote.Child",
            "scriptRun",
            "remote",
            "AlmostCommonRuntime",
            "CommonRuntime"
        ],
        "attribution": "AutoML",
        "computeType": "AmlcTrain"
    },
    "settings": {},
    "computeRequest": {
        "nodeCount": 1,
        "gpuCount": 0
    },
    "compute": {
        "target": "Serverless",
        "targetType": "amlcompute",
        "vmSize": null,
        "instanceType": null,
        "instanceCount": 1,
        "gpuCount": 0,
        "priority": null,
        "region": null,
        "armId": null,
        "properties": null
    },
    "createdBy": {
        "userObjectId": "57a8a404-8226-4dde-86ac-3f525b282e34",
        "userPuId": "10032003551ABF14",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:00067FFEAD36E0ED",
        "userIss": "https://sts.windows.net/483d5369-8f79-479c-bb98-1b542eccc4a8/",
        "userTenantId": "483d5369-8f79-479c-bb98-1b542eccc4a8",
        "userName": "Caroline Galavote",
        "upn": null
    },
    "computeDuration": "00:00:50.0441382",
    "effectiveStartTimeUtc": null,
    "runNumber": 1708363780,
    "rootRunId": "mllearn-bike-automl",
    "experimentId": "67c3d54a-a645-49fa-b2b0-2fe97e718e55",
    "userId": "57a8a404-8226-4dde-86ac-3f525b282e34",
    "statusRevision": 6,
    "currentComputeTime": null,
    "lastStartTimeUtc": null,
    "lastModifiedBy": {
        "userObjectId": "57a8a404-8226-4dde-86ac-3f525b282e34",
        "userPuId": "10032003551ABF14",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:00067FFEAD36E0ED",
        "userIss": "https://sts.windows.net/483d5369-8f79-479c-bb98-1b542eccc4a8/",
        "userTenantId": "483d5369-8f79-479c-bb98-1b542eccc4a8",
        "userName": "Caroline Galavote",
        "upn": null
    },
    "lastModifiedUtc": "2024-02-19T17:30:54.5483098+00:00",
    "duration": "00:00:50.0441382",
    "inputs": {
        "training_data": {
            "assetId": "azureml://locations/eastus/workspaces/ab7923e7-edcb-4704-a99e-0c0f3d6311b2/data/alugueldebicicletas/versions/1",
            "type": "MLTable"
        }
    },
    "outputs": {
        "mlflow_log_model_971140291": {
            "assetId": "azureml://locations/eastus/workspaces/ab7923e7-edcb-4704-a99e-0c0f3d6311b2/models/azureml_mllearn-bike-automl_2_output_mlflow_log_model_971140291/versions/1",
            "type": "MLFlowModel"
        }
    },
    "currentAttemptId": 1
}
