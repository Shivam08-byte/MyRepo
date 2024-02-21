"""
*
* =============================================================================
* COPYRIGHT NOTICE
* =============================================================================
*  @ Copyright HCL Technologies Ltd. 2021, 2022,2023
* Proprietary and confidential. All information contained herein is, and
* remains the property of HCL Technologies Limited. Copying or reproducing the
* contents of this file, via any medium is strictly prohibited unless prior
* written permission is obtained from HCL Technologies Limited.
*

AION Endpoints
1. training/problemtype/{problem_type_id}
2. problemtypes/training
3. retrainModel
"""
import json
import os
import warnings
from datetime import datetime
import logging
import re

from fastapi.responses import JSONResponse
from config.aion_config_reader import setup_config, get_key_value
import django
import httpx
from django.db import connection
from fastapi import APIRouter, Path, UploadFile, File
from starlette.background import BackgroundTasks
from aion import aion_training
from appbe.aion_config import settings
from appbe.pages import getversion
from utils.db_operations import getFileName, update_status
from django.core.paginator import Paginator
from retrain_reconfigure.reconfigure_endpoint_handler import reconfigure_call
from retrain_reconfigure.retrain_endpoint_handler import retrain_call
from utils.api_models import trainingResponse, notFound, unexpectedError, timeOutError

router = APIRouter()

ALLOWED_PROBLEM_TYPES = [
    "timeseries",
    "anomaly_detection",
    "timeseries_anomaly_detection",
    "clustering",
    "classification",
    "regression",
]


# training/problemtype/{problem_type_id} API Endpoint
@router.post(
    "/aion/training/problemtype/{problem_type_id}",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
    description="[README.md](/docs/preview_md?file_path=docs/training_readme.md)",
)
def train_only(
    tracking_id: int,
    problem_type_id: str = Path(
        ...,
        description=f"Type of the problem like {ALLOWED_PROBLEM_TYPES}",
        # title="Problem Type",
        # regex="|".join(ALLOWED_PROBLEM_TYPES),
    ),
    config_file: UploadFile = File(
        description="Upload config file for training",
    ),
):
    aion_config = setup_config()
    filename = getFileName()
    config_path = "/root/HCLT/data/config/" + filename.replace("csv", "json")
    data = json.load(config_file.file)
    if data["basic"]["modelName"] == "" or re.search(
        "[^A-Za-z0-9]", data["basic"]["modelName"]
    ):
        # Exit or handle the case where modelName is empty or contains special characters
        exit(["Usecase ID missing or contains special characters"])
    with open(config_path, "w") as json_file:
        json.dump(data, json_file)
    if problem_type_id.lower() == "timeseries":
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8001/aion/problemtype/timeseries/training?config_path={config_file_path}"

        # timeout_seconds = 10000
        timeseries_url = get_key_value(aion_config, "ProblemTypeURLs", "timeseries.url")
        url = f"{timeseries_url}={config_file_path}"
        print("url value is: ", url)

        with httpx.Client() as client:
            try:
                # ts_resp = client.post(url, timeout=timeout_seconds)
                ts_resp = client.post(url, timeout=None)
                if ts_resp.status_code == 200:
                    print(f"Received response: {ts_resp.text}")
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "SUCCESS", config_file_path
                    )
                    return ts_resp.text

                else:
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "FAILED", config_file_path
                    )
                    return f"Received a non-200 status code: {ts_resp.status_code}"
            except httpx.TimeoutException:
                return f"Request timed out after {timeout_seconds} seconds"
    # elif problem_type_id.lower() == "anomaly_detection":
    elif (
        problem_type_id.lower().replace(" ", "").replace("_", "").replace("-", "")
        == "anomalydetection"
        or problem_type_id.lower().replace(" ", "").replace("_", "").replace("-", "")
        == "timeseriesanomalydetection"
    ):
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8003/aion/problemtype/anomaly-detection/training?config_path={config_file_path}"
        anomalydetection_url = get_key_value(
            aion_config, "ProblemTypeURLs", "anomalydetection.url"
        )
        url = f"{anomalydetection_url}={config_file_path}"
        print("url value is: ", url)
        timeout_seconds = 10000

        with httpx.Client() as client:
            try:
                ad_resp = client.post(url, timeout=timeout_seconds)

                # Check the response status code to ensure it's successful (e.g., 200 OK)
                if ad_resp.status_code == 200:
                    # Handle successful response here
                    print(f"Received response: {ad_resp.text}")
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "SUCCESS", config_file_path
                    )
                    return ad_resp.text

                else:
                    # Handle non-200 status codes here
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "FAILED", config_file_path
                    )
                    return f"Received a non-200 status code: {ad_resp.status_code}"
            except httpx.TimeoutException:
                # Handle timeout error
                return f"Request timed out after {timeout_seconds} seconds"
    elif problem_type_id.lower() == "clustering":
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8004/aion/training/clustering?config_path={config_file_path}"
        clustering_url = get_key_value(aion_config, "ProblemTypeURLs", "clustering.url")
        url = f"{clustering_url}{config_file_path}"
        print("url value is: ", url)
        timeout_seconds = 10000

        with httpx.Client() as client:
            try:
                clustering_resp = client.post(url, timeout=timeout_seconds)

                # Check the response status code to ensure it's successful (e.g., 200 OK)
                if clustering_resp.status_code == 200:
                    # Handle successful response here
                    print(f"Received response: {clustering_resp.text}")
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "SUCCESS", config_file_path
                    )
                    return clustering_resp.text

                else:
                    # Handle non-200 status codes here
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "FAILED", config_file_path
                    )
                    return (
                        f"Received a non-200 status code: {clustering_resp.status_code}"
                    )
            except httpx.TimeoutException:
                # Handle timeout error
                return f"Request timed out after {timeout_seconds} seconds"
    elif problem_type_id.lower() == "classification":
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8005/aion/problemtype/classification/training?config_path={config_file_path}"

        # timeout_seconds = 10000
        classification_url = get_key_value(
            aion_config, "ProblemTypeURLs", "classification.url"
        )
        url = f"{classification_url}={config_file_path}"
        print("url value is: ", url)

        with httpx.Client() as client:
            try:
                classification_response = client.post(url, timeout=None)

                if classification_response.status_code == 200:
                    print(f"Received response: {classification_response.text}")
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "SUCCESS", config_file_path
                    )
                    return classification_response.text

                else:
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "FAILED", config_file_path
                    )
                    return f"Received a non-200 status code: {classification_response.status_code}"

            except httpx.TimeoutException:
                # Handle timeout error
                return f"Request timed out"
    elif problem_type_id.lower() == "regression":
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8006/aion/problemtype/regression/training?config_path={config_file_path}"

        # timeout_seconds = 10000
        regression_url = get_key_value(aion_config, "ProblemTypeURLs", "regression.url")
        url = f"{regression_url}={config_file_path}"
        print("url value is: ", url)

        with httpx.Client() as client:
            try:
                regression_response = client.post(url, timeout=None)

                if regression_response.status_code == 200:
                    print(f"Received response: {regression_response.text}")
                    update_status(
                        tracking_id, "PARTIAL_TRAINING", "SUCCESS", config_file_path
                    )
                    return regression_response.text

                else:
                    return f"Received a non-200 status code: {regression_response.status_code}"
            except httpx.TimeoutException:
                update_status(
                    tracking_id, "PARTIAL_TRAINING", "FAILED", config_file_path
                )
                # Handle timeout error
                return f"Request timed out"
    else:
        return "Invalid problem type"


# problemtypes/training API Endpoint
@router.post(
    "/aion/problemtypes/training",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
    description="[README.md](/docs/preview_md?file_path=docs/training_readme.md)",
)
def train_for_pipeline(
    async_train: bool,
    background_tasks: BackgroundTasks,
    tracking_id: int,
    confFile: UploadFile = File(description="Upload config file for training"),
):
    filename = getFileName()
    config_path = "/root/HCLT/data/config/" + filename.replace("csv", "json")
    data = json.load(confFile.file)
    with open(config_path, "w") as json_file:
        json.dump(data, json_file)
    if async_train == True:
        background_tasks.add_task(train_for_pipeline_main, tracking_id, config_path)
        update_status(tracking_id, "TRAINING", "InProgress", config_path)
        return JSONResponse(content={"message":"Traing in progres you can trace its progress using fetch status API"})
    else:
        # pass
        return train_for_pipeline_main(tracking_id,config_path)

def train_for_pipeline_main(
    tracking_id: int,
    # confFile: UploadFile = File(description="Upload config file for training"),
    config_path: str
):
    from aion import aion_training

    # filename = getFileName()
    # config_path = "/root/HCLT/data/config/" + filename.replace("csv", "json")
    # data = json.load(confFile.file)
    # with open(config_path, "w") as json_file:
    #     json.dump(data, json_file)

    result_string = aion_training(config_path)
    result_dict = json.loads(result_string)

    print(
        "##### create the OUTPUT file and save it in the config folder with timestamp #####"
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    trainOuputLocation = f"/root/HCLT/data/config/AION_OUTPUT_{timestamp}.json"
    with open(trainOuputLocation, "w") as json_file:
        json.dump(result_dict, json_file, indent=2)

    print("##### updating the table with the OUTPUT file path #####")
    # breakpoint()

    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "appfe.ux.settings")
        aion_version = getversion()
        usecase_tab = settings()

        # warnings.filterwarnings("ignore")
        # config_path = Path(config_path)
        with open(config_path, "r") as f:
            config = json.load(f)

        usecaseid = config["basic"]["modelName"]
        version = config["basic"]["modelVersion"]
        dataFilePath = config["basic"]["dataLocation"]
        deployPath = result_dict["data"]["deployLocation"]
        status = result_dict["status"]
        modelType = result_dict["data"]["ModelType"]
        print("value of usecaseid is : ", usecaseid)
        print("value of version is : ", version)
        print("value of dataFilePath is : ", dataFilePath)
        print("value of deployPath is : ", deployPath)
        print("value of status is : ", status)
        print("value of modelType is : ", modelType)

        django.setup()

        get_usecase_id_query = (
            f""" SELECT id from usecasedetails where usecasename = '{usecaseid}' """
        )
        with connection.cursor() as cursor:
            cursor.execute(get_usecase_id_query)
            usecase_id = cursor.fetchone()
        print("The usecase_id value is: ", usecase_id[0])

        ##### insert the usecase detials in existusecases table #####
        existusecases_insert_query = f"""
            INSERT INTO existusecases (id, ModelName_id, version, dataFilePath, ConfigPath, DeployPath, Status, TrainOuputLocation, ProblemType,driftStatus,portNo,publishStatus,publishPID,modelType,trainingPID)
            VALUES ({usecase_id[0]}, {usecase_id[0]}, '{version}', '{dataFilePath}', '{config_path}', '{deployPath}', '{status}', '{trainOuputLocation}', '', '', 0, '', 0, '{modelType}', 0)
        """
        print("existusecases_insert_query value is: ", existusecases_insert_query)
        with connection.cursor() as cursor:
            cursor.execute(existusecases_insert_query)
        update_status(tracking_id, "TRAINING", "SUCCESS", config_path)
    except django.core.exceptions.ImproperlyConfigured as e:
        update_status(tracking_id, "TRAINING", "FAILED", config_path)
        print(f"Django configuration error: {e}")
    except django.db.utils.DatabaseError as e:
        update_status(tracking_id, "TRAINING", "FAILED", config_path)
        print(f"Database error: {e}")
    except Exception as e:
        update_status(tracking_id, "TRAINING", "FAILED", config_path)
        print(f"An unexpected error occurred: {e}")

    return {"output": result_dict}


# retrainModel API Endpoint
@router.post(
    "/aion/retrainModel/",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
    description="[README.md](/docs/preview_md?file_path=docs/retrain.md)",
)
def retrain(use_case_name: str, model_version: int):
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        # Call the retrain endpoint
        result = retrain_call(use_case_name, model_version)

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        result = {"error": f"FileNotFoundError: {e}"}
    return result


@router.post(
    "/aion/reconfigure_train/",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
    description="[README.md](/docs/preview_md?file_path=docs/reconfigure_train.md)",
)
def reconfigure(
    confFile: UploadFile = File(description="Upload config file for training"),
):
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        # Call the retrain endpoint
        result = reconfigure_call(confFile)

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        result = {"error": f"FileNotFoundError: {e}"}
    return result
