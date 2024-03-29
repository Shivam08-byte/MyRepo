
import csv
from enum import Enum

from fastapi.openapi.docs import get_swagger_ui_html

import auth.login
import auth.auth_db_ops
from auth import models
from auth.database import engine
from utils.db_operations import *
from utils.api_models import *
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
import os
import asyncio
import pandas as pd
import requests
from aion import *
from utils import upload_file_in_docker
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Path,
    UploadFile,
    Query,
    HTTPException,
    Depends,
    File,
)
import logging
from pydantic import BaseModel
import httpx
import os
import re
import shutil
import subprocess
import sys
from django.core.paginator import Paginator
from appbe import compute
from io import StringIO
import markdown
from typing import Union, List, Annotated
from zipfile import ZipFile
from fastapi.staticfiles import StaticFiles
from config.aion_config_reader import setup_config, get_key_value
from auth.login import oauth2_scheme

version_v1 = APIRouter()

aionapp = FastAPI(docs_url=None,
    description="<div style='display: flex; align-items:center;'><img src='/static/AION_logo_blue.png'  alt='logo' width='100' height='60' style='horizontal-align: middle; margin-right:10px;'></div> ",
    title="AION REST Interface",
)

@aionapp.get("/docs",include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(openapi_url="/openapi.json",title="AION", swagger_favicon_url='/static/AION_logo_blue.png')

models.Base.metadata.create_all(engine)
aionapp.include_router(auth.auth_db_ops.router)
aionapp.include_router(auth.login.router)
aionapp.mount("/static", StaticFiles(directory="static"), name="static")
upload_status = UploadStatus()


@version_v1.get(
    "/aion/get_unique_tracking_id/",
    response_model=TrackingIdResponse,
    responses={
        500: {"model": unexpectedError},
    },
    tags=["Tracker"],
    description="[README.md](/docs/preview_md?file_path=docs/tracker.md)",
)
async def get_unique_tracking_id():
    try:
        id = Unique_id_gen()
        tracking_id = id.generated_id()
        if tracking_id is None:
            raise HTTPException(
                status_code=404,
            )  # details="Failed to generate a valid tracking id")
        message = "Tracking Id generated successfully"
        return {"tracking_id": tracking_id, "message": message}

    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# fetchStatus API ENDPOINTS
@version_v1.get(
    "/aion/fetchStatus",
    response_model=fetchStatusResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["API status"],
    description="[README.md](/docs/preview_md?file_path=docs/api_status.md)",
)
def fetch_status_api(
    API_name: API_List, tracking_id: Union[int, None] = None, usecaseid: str = None
):
    try:
        status = get_status(API_name, tracking_id, usecaseid)
        condition1 = f"tracking_id," if tracking_id else ""
        condition2 = f"usecaseid = '{usecaseid}'" if usecaseid else ""

        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"No Status exist corresponding {condition1} {condition2} and API ",
            )

        return {
            "tracking_id": tracking_id,
            "API": API_name,
            "api_status": status,
        }

    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# upload data API
@version_v1.post(
    "/aion/uploadData/browseFile/",
    response_model=uploadDataResponse,
    responses={
        500: {"model": unexpectedError},
    },
    tags=["Data Handler"],
    description="[README.md](/docs/preview_md?file_path=docs/dataupload_browse.md)",
)
async def upload_data_gen_usecase(
    background_tasks: BackgroundTasks, browse: Browse_File = Depends()
):
    print(f"file_size = {get_file_size(browse.file.file)} kb")
    deploy_path = deployment_loc + getFileName()
    description = ""
    if get_file_size(browse.file.file) > 30:  # need to be chnaged
        if not upload_status.upload_in_progress:
            upload_status.upload_in_progress = True
            upload_status.data_uploaded = False  # Reset data_uploaded status

            # update/generate usecase
            try:
                res = insert_data(
                    trackingid=browse.tracking_id,
                    datapathlocation=deploy_path,
                    description=description,
                )
                (
                    id,
                    description,
                    usecaseid,
                    usecaseid,
                    trackingid,
                    datapathlocation,
                ) = res
                # update api status
                model = ApiMasterModel(
                    api_name="DATA_UPLOAD",
                    api_status="In Progress",
                    model_version=1,
                    uuid=trackingid,
                    api_result_location=datapathlocation,
                    usecaseid=usecaseid,
                )
                await setapistatus(model)

            except Exception as e:
                print(f"An error occured while inserting data in db: {e}")

            background_tasks.add_task(upload_data, browse, deploy_path, usecaseid)
        return JSONResponse(
            content={
                "message": "Data upload in progress",
                "trackingid": trackingid,
                "dataLocation": datapathlocation,
                "usecasename": usecaseid,
            }
        )
    else:
        fp = await browse.file.read()
        s = str(fp, "utf-8")
        data = StringIO(s)
        df = pd.read_csv(data)
        # description = "raw data"
        # deploy_path = deployment_loc + getFileName()
        df.to_csv(deploy_path, index=False)
        try:
            res = insert_data(
                trackingid=browse.tracking_id,
                datapathlocation=deploy_path,
                description=description,
            )
            id, description, usecaseid, usecaseid, trackingid, datapathlocation = res
            # update api status
            model = ApiMasterModel(
                api_name="DATA_UPLOAD",
                api_status="Success",
                model_version=1,
                uuid=trackingid,
                api_result_location=datapathlocation,
                usecaseid=usecaseid,
            )
            setapistatus(model)
            return JSONResponse(
                content={
                    "trackingid": trackingid,
                    "dataLocation": datapathlocation,
                    "usecasename": usecaseid,
                }
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


@version_v1.post(
    "/aion/uploadData/nifi/",
    response_model=uploadDataResponse,
    responses={500: {"model": unexpectedError}, 400: {"model": badError}},
    tags=["Data Handler"],
    description="[README.md](/docs/preview_md?file_path=docs/dataupload_nifi.md)",
)
async def upload_data_from_nifi(nifi: NiFi = Depends()):
    response = requests.get(nifi.nifi_url)
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail="Failed to retrive file from the given URL"
        )
    docker_file_path = nifi.nifi_url.replace("localhost", "host.docker.internal")
    response = requests.get(docker_file_path)
    data = response.content.decode("utf-8")
    description = "data belongs to nifi"
    df = pd.read_csv(StringIO(data))
    deploy_path = deployment_loc + getFileName()
    df.to_csv(deploy_path, index=False)
    try:
        res = insert_data(
            trackingid=nifi.tracking_id,
            datapathlocation=deploy_path,
            description=description,
        )
        id, description, usecaseid, usecaseid, trackingid, datapathlocation = res
        # update api status
        model = ApiMasterModel(
            api_name="DATA_UPLOAD",
            api_status="Success",
            model_version=1,
            uuid=trackingid,
            api_result_location=datapathlocation,
            usecaseid=usecaseid,
        )
        setapistatus(model)
        return {
            "trackingid": trackingid,
            "deploymentloc": datapathlocation,
            "usecasename": usecaseid,
        }
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@version_v1.post(
    "/aion/uploadData/url/",
    response_model=uploadDataResponse,
    responses={500: {"model": unexpectedError}, 400: {"model": badError}},
    tags=["Data Handler"],
    description="[README.md](/docs/preview_md?file_path=docs/dataupload_url.md)",
)
async def upload_data_from_url(url: URL = Depends()):
    response = requests.get(url.path)
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, detail="Failed to retrive file from the given URL"
        )
    df = pd.read_csv(url.path)
    deploy_path = deployment_loc + getFileName() + ".csv"
    df.to_csv(deploy_path, index=False)
    description = "data from url"
    try:
        res = insert_data(
            trackingid=url.tracking_id,
            datapathlocation=deploy_path,
            description=description,
        )
        id, description, usecaseid, usecaseid, trackingid, datapathlocation = res
        # update api status
        model = ApiMasterModel(
            api_name="DATA_UPLOAD",
            api_status="Success",
            model_version=1,
            uuid=trackingid,
            api_result_location=datapathlocation,
            usecaseid=usecaseid,
        )
        setapistatus(model)
        return {
            "trackingid": trackingid,
            "deploymentloc": datapathlocation,
            "usecasename": usecaseid,
        }
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@version_v1.post(
    "/aion/uploadDataOnly",
    response_model=dataOnlyResponse,
    responses={
        500: {"model": unexpectedError},
        406: {"model": notAcceptable},
    },
    tags=["Data Handler"],
)
async def upload_data_to_mountpoint_storage(
    file_path: UploadFile,
):
    ############ (STARTS) uploading the prediction file (of any format) which user has provided into "/root/HCLT/data/storage/" location ######
    # Check if the file was uploaded successfully
    if file_path and file_path.filename:
        # Define the target save location
        save_location = f"/root/HCLT/data/storage/{file_path.filename}"

        # Save the file
        with open(save_location, "wb") as file:
            file.write(file_path.file.read())

        print(f"{file_path.filename} saved sucessfully at location {save_location}")
        return {
            "message": "File saved successfully",
            "file_name": f"/root/HCLT/data/storage/{file_path.filename}",
        }
    else:
        print(f"{file_path.filename} has failed to save at location {save_location}")
        return {
            "message": f"{file_path.filename} has failed to save at location {save_location}"
        }

    ############ (ENDS) uploading the prediction file which user has provided into "/root/HCLT/data/storage/" location ######


# Markdown file preview page
# call this in API endpoint
@aionapp.get(
    "/docs/preview_md",
    response_class=HTMLResponse,
    responses={
        500: {"model": unexpectedError},
    },
    include_in_schema=False,
)
async def preview_md(file_path: str):
    try:
        # Load and render the content of the specified markdown file
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content, extensions=["tables"])

        # HTML page with Markdown content
        content = f"""
        <html>
            <head>
                <title>Markdown Preview</title>
            </head>
            <body>
                <a href="javascript:window.close()">Close</a>
                <hr>
                {html_content}
            </body>
        </html>
        """
        return HTMLResponse(content=content)
    except Exception as e:
        # Handle exceptions if necessary
        raise HTTPException(status_code=500, detail=str(e))


@version_v1.post(
    "/aion/issue_spotter/",
    responses={
        200: {"model": issueSpotterResponse},
        404: {"model": notFound},
        500: {"model": issueSpotterError},
    },
    tags=["Data Handler"],
    description="[README.md](/docs/preview_md?file_path=docs/issue_spotter_explanations.md)",
)
async def issue_spotter(
    tracking_id: int, file_name: str, token: Annotated[str, Depends(oauth2_scheme)]
):
    file_location = "/root/HCLT/data/storage/"
    csv_file = os.path.join(file_location, file_name)
    if not os.path.isfile(csv_file):
        raise HTTPException(status_code=400, detail="File not found")
    try:
        response = api_response(file_location, file_name)
        # model = ApiMasterModel(
        #     "DATA_ISSUE_SPOTTER", "SUCCESS", 0, tracking_id, file_location+file_name,"none",
        # )
        # update api status
        model = ApiMasterModel(
            api_name="DATA_ISSUE_SPOTTER",
            api_status="SUCCESS",
            model_version=0,
            uuid=tracking_id,
            api_result_location=file_name,
            usecaseid="none",
        )
        setapistatus(model)
        return response
    except Exception as e:
        # Handle exception
        status = "FAIL"
        msg = "error"
        error = HTTPException(status_code=500, detail=str(e))
        response = {"status": status, "msg": msg, "error": error}
        model = ApiMasterModel(
            api_name="DATA_ISSUE_SPOTTER",
            api_status=status,
            model_version=0,
            uuid=tracking_id,
            api_result_location=file_name,
            usecaseid="none",
        )
        setapistatus(model)
        return response


@version_v1.post(
    "/aion/problemtype/eda/set_data_location",
    responses={
        200: {"model": edaDataLocationResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["EDA"],
)
async def set_data_location(
    data_location: str = Query(
        "/root/HCLT/data/storage/filename", description="Enter the data file path"
    )
):
    data_location_instance = DataLocation()
    data_location_instance.data_location = data_location
    await data_location_instance.select_feature()

    return {
        "message": "Data location set successfully",
        "data_location": data_location_instance.data_location,
        "selected_feature": data_location_instance.selected_feature,
    }


@version_v1.post(
    "/aion/problemtype/eda",
    responses={
        200: {"model": edaProblemTypeResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["EDA"],
    description="[README.md](/docs/preview_md?file_path=docs/eda.md)",
)
async def handler_eda(
    select_eda: EDA_TYPES,
    data_location_instance: DataLocation = Depends(DataLocation),
    selected_feature: str = Query('None', description="`None` if no feature is selected"),
):
    await data_location_instance.select_feature()
    model = ApiMasterModel(
                api_name="EDA_ALL",
                api_status="FAILED",
                model_version=0,
                uuid=0,
                api_result_location="none",
                usecaseid="none",
            )
    try:
        url = f"http://127.0.0.1:8008/aion/problemtype/eda/{select_eda}?data_location={data_location_instance.data_location}&selected_feature={selected_feature}"

        async with httpx.AsyncClient() as client:
            records_resp = await client.get(url, timeout=None)
            records_resp.raise_for_status()

            response_data = records_resp.json()["path"]
            pdf_file_path = os.path.join(response_data, f"{select_eda}.pdf")

        # return StreamingResponse(content=response_data, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="records_report.pdf"'})
        model.api_status = "SUCCESS"
        setapistatus(model)
        return FileResponse(
            path=pdf_file_path,
            filename=f"{select_eda}.pdf",
            media_type="application/pdf",
        )

    except httpx.TimeoutException:
        setapistatus(model)
        return JSONResponse(content={"error": f"Request timed out"}, status_code=500)

    except httpx.HTTPError as e:
        setapistatus(model)
        return JSONResponse(
            content={"error": f"HTTP error occurred: {e}", "details": str(e)},
            status_code=e.response.status_code,
        )

    except Exception as e:
        setapistatus(model)
        return JSONResponse(
            content={"error": f"An unexpected error occurred: {str(e)}"},
            status_code=500,
        )

ALLOWED_PROBLEM_TYPES = [
    "timeseries",
    "anomaly_detection",
    "timeseries_anomaly_detection",
    "clustering",
    "classification",
    "regression",
]


@version_v1.post(
    "/aion/training/problemtype/{problem_type_id}",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
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
    with open(config_path, "w") as json_file:
        json.dump(data, json_file)
    if problem_type_id.lower() == "timeseries":
        config_file_path = str(config_path)
        # url = f"http://127.0.0.1:8001/aion/problemtype/timeseries/training?config_path={config_file_path}"
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


# @aionapp.post("/aion/problemtype/timeseries/training", tags=["AION-TRAINING"])
# def aion_timeseries_endpoint(config_path: str):
#     config_file_path = str(config_path)
#     url = f"http://127.0.0.1:8001/aion/problemtype/timeseries/training?config_path={config_file_path}"

#     timeout_seconds = 600

#     with httpx.Client() as client:
#         try:
#             ts_resp = client.post(url, timeout=timeout_seconds)

#             # Check the response status code to ensure it's successful (e.g., 200 OK)
#             if ts_resp.status_code == 200:
#                 # Handle successful response here
#                 print(f"Received response: {ts_resp.text}")
#                 return ts_resp.text

#             else:
#                 # Handle non-200 status codes here
#                 print(f"Received a non-200 status code: {ts_resp.status_code}")
#         except httpx.TimeoutException:
#             # Handle timeout error
#             print(f"Request timed out after {timeout_seconds} seconds")


# @aionapp.post("/aion/problemtype/classification/training", tags=["AION-TRAINING"])
# def aion_classification_endpoint(config_path: str):
#     config_file_path = str(config_path)
#     url = f"http://127.0.0.1:8005/aion/problemtype/classification/training?config_path={config_file_path}"

#     # timeout_seconds = 800

#     with httpx.Client() as client:
#         try:
#             classification_response = client.post(url, timeout=None)

#             if classification_response.status_code == 200:
#                 print(f"Received response: {classification_response.text}")
#                 return classification_response.text

#             else:
#                 print(
#                     f"Received a non-200 status code: {classification_response.status_code}"
#                 )
#         except httpx.TimeoutException:
#             # Handle timeout error
#             print(f"Request timed out")


# @aionapp.post("/aion/problemtype/regression/training", tags=["AION-TRAINING"])
# def aion_regression_endpoint(config_path: str):
#     config_file_path = str(config_path)
#     url = f"http://127.0.0.1:8006/aion/problemtype/regression/training?config_path={config_file_path}"

#     # timeout_seconds = 800

#     with httpx.Client() as client:
#         try:
#             regression_response = client.post(url, timeout=None)

#             if regression_response.status_code == 200:
#                 print(f"Received response: {regression_response.text}")
#                 return regression_response.text

#             else:
#                 print(
#                     f"Received a non-200 status code: {regression_response.status_code}"
#                 )
#         except httpx.TimeoutException:
#             # Handle timeout error
#             print(f"Request timed out")


# @aionapp.post("/aion/problemtype/clustering/training", tags=["AION-TRAINING"])
# def aion_regression_endpoint(config_path: str):
#     config_file_path = str(config_path)
#     url = (
#         f"http://127.0.0.1:8004/aion/training/clustering?config_path={config_file_path}"
#     )

#     timeout_seconds = 600

#     with httpx.Client() as client:
#         try:
#             clustering_resp = client.post(url, timeout=timeout_seconds)

#             # Check the response status code to ensure it's successful (e.g., 200 OK)
#             if clustering_resp.status_code == 200:
#                 # Handle successful response here
#                 print(f"Received response: {clustering_resp.text}")
#                 return clustering_resp.text

#             else:
#                 # Handle non-200 status codes here
#                 print(f"Received a non-200 status code: {clustering_resp.status_code}")
#         except httpx.TimeoutException:
#             # Handle timeout error
#             print(f"Request timed out after {timeout_seconds} seconds")


# @aionapp.post("/aion/problemtype/anomaly-detection/training", tags=["AION-TRAINING"])
# def aion_regression_endpoint(config_path: str):
#     config_file_path = str(config_path)
#     url = f"http://127.0.0.1:8003/anomaly-detection/training?config_path={config_file_path}"

#     timeout_seconds = 600

#     with httpx.Client() as client:
#         try:
#             ad_resp = client.post(url, timeout=timeout_seconds)

#             # Check the response status code to ensure it's successful (e.g., 200 OK)
#             if ad_resp.status_code == 200:
#                 # Handle successful response here
#                 print(f"Received response: {ad_resp.text}")
#                 return ad_resp.text

#             else:
#                 # Handle non-200 status codes here
#                 print(f"Received a non-200 status code: {ad_resp.status_code}")
#         except httpx.TimeoutException:
#             # Handle timeout error
#             print(f"Request timed out after {timeout_seconds} seconds")


@version_v1.post(
    "/aion/problemtype/prediction/instance/",
    responses={
        200: {"model": singleInstanceresponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Prediction"],
)
async def handle_single_prediction(model: InstancePredictModel):
    usecaseid = model.usecaseid
    version = model.version
    DEPLOY_LOCATION="/root/HCLT/data/target"
    data=str(model.data).replace("'",'"')
    try:
        model_path = os.path.join(DEPLOY_LOCATION, usecaseid, str(version))
        isdir = os.path.isdir(model_path)
        if isdir:
            try:
                predict_path = os.path.join(model_path, "aion_predict.py")
                outputStr = subprocess.check_output(
                    [sys.executable, predict_path, data]
                )
                outputStr = outputStr.decode("utf-8")
                json_str=outputStr[outputStr.find("{"):]
                resp=json.loads(json_str)
                return f"{resp}"
            except Exception as e:
                print(e)
                return {
                    json.dumps({"status": "error", "msg": "Bad Request"})
                }
        else:
            return {
                json.dumps({"status": "error", "msg": "Wrong UseCaseID or Version"})
                }

    except Exception as e:
        return JSONResponse(content={f"An error occured while inserting data in db: {e}"})

@version_v1.post(
    "/aion/problemtype/prediction/singleinstance",
    responses={
        200: {"model": singleInstanceresponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Prediction"],
    description="[README.md](/docs/preview_md?file_path=docs/single_prediction.md)",
    include_in_schema=False
)
async def handle_single_prediction(input_data: Input_data):
    usecaseid = input_data.usecaseid
    version = input_data.version
    data = input_data.data

    url = (
        "http://localhost/api/predict?usecaseid="
        + usecaseid
        + "&version="
        + str(version)
    )

    timeout = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=data)

        if response.status_code == 200:
            result = response.json()

        else:
            result = f"Request failed with status code {response.status_code}"
    return result


@version_v1.get(
    "/aion/problemtype/prediction/batchprediction",
    responses={
        200: {"model": batchPredResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Prediction"],
    description="[README.md](/docs/preview_md?file_path=docs/batch_prediction.md)",
)
def handle_batch_prediction(
    response: str = Query(
        "view/download", description="You can download or view the output"
    ),
    tracking_id: int = Query("1234567", description="Enter the tracking id"),
    usecaseid: str = Query("AI0001", description="Enter the usecase id."),
    model_version: str = Query("1", description="Enter the version."),
    path_type: str = Query(
        "local/url", description="You can give moutpoint paht or URL"
    ),
    predict_file_path: str = Query(
        "/root/HCLT/data/storage/filename",
        description="Enter the predict file path",
    ),
):
    from prediction_package.batch_prediction_helper import batch_prediction

    result = batch_prediction(usecaseid, model_version, predict_file_path)

    if path_type == "url":
        check_df = pd.read_csv(predict_file_path)
        filetimestamp = str(int(time.time()))
        predict_file_path = os.path.join(
            DATA_FILE_PATH, "AION_" + filetimestamp + ".csv"
        )
        check_df.to_csv(predict_file_path, index=False)

    if result["tab"] == "predict":
        # set api status
        model = ApiMasterModel(
            api_name="BATCH_PREDICTION",
            api_status="SUCCESS",
            model_version=model_version,
            uuid=tracking_id,
            api_result_location="none",
            usecaseid=usecaseid,
        )
        setapistatus(model)

        if response == "view" or "":
            return JSONResponse(content={'top_prediction':result["predictionResults"][:10]})
        else:
            # save data as csv file
            csvfilename = getFileName()
            file_path = f"/root/HCLT/data/target/{usecaseid}/{model_version}/data/predcition/{csvfilename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            headers = (
                result["predictionResults"][0].keys() if result["predictionResults"] else []
            )

            with open(file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(result["predictionResults"])
            return FileResponse(
                path=file_path, filename=csvfilename, media_type="application/csv"
            )

    else:
        model = ApiMasterModel(
            api_name="BATCH_PREDICTION",
            api_status="FAILED",
            model_version=model_version,
            uuid=tracking_id,
            api_result_location="none",
            usecaseid=usecaseid,
        )
        setapistatus(model)
        return JSONResponse(content={"status": "FAILED", "msg": result["error"]})


@version_v1.get(
    "/aion/problemtype/download/pythonpackage",
    response_model=downloadModelResponse,
    responses={
        400: {"model": dowmloadPackageError},
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Download-model"],
)
def aion_download_python_package_endpoint(
    deployPath: str = Query(
        "/root/HCLT/data/target/AI0001/1", description="Enter the deploy path."
    ),
    usecasename: str = Query("AI0001", description="Enter the usecase id."),
    Version: int = Query("1", description="Enter the version."),
):
    try:
        from appbe.installPackage import generate_python_package

        zip_file_path = generate_python_package(deployPath, usecasename, Version)
        print("zip_file_path", zip_file_path)
        FileResponse(zip_file_path, media_type="application/zip")
        return {"zip_file_path": zip_file_path}
    except Exception as e:
        return FileResponse(
            "Error creating package", status_code=500, media_type="application/error"
        )


@version_v1.get(
    "/aion/problemtype/download/docker_container_package",
    response_model=dockerPackageResponse,
    responses={
        400: {"model": dowmloadPackageError},
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Download-model"],
)
def aion_download_docker_container_package_endpoint(
    deployPath: str = Query(
        "/root/HCLT/data/target/AI0001/1", description="Enter the deploy path."
    ),
    usecasename: str = Query("AI0001", description="Enter the usecase id."),
    Version: int = Query("1", description="Enter the version."),
):
    from appbe.installPackage import build_docker_image

    msg, status = build_docker_image(deployPath, usecasename, Version)
    return {"msg": msg, "status": status}


@version_v1.get(
    "/aion/problemtype/download/mlac",
    response_model=mlacResponse,
    responses={
        400: {"model": dowmloadPackageError},
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Download-model"],
)
def aion_download_mlac_endpoint(
    deployPath: str = Query(
        "/root/HCLT/data/target/AI0001/1", description="Enter the deploy path."
    ),
):
    codeconfig = os.path.join(deployPath, "etc", "code_config.json")
    if os.path.isfile(codeconfig):
        with open(codeconfig, "r") as f:
            cconfig = json.load(f)
        f.close()
        cconfig["prod_db_type"] = "sqlite"
        cconfig["db_config"] = {}
        cconfig["mlflow_config"] = {}
        with open(codeconfig, "w") as f:
            json.dump(cconfig, f)
        f.close()

        from bin.aion_mlac import generate_mlac_code

        print("codeconfig value is: ", codeconfig)
        try:
            outputStr = generate_mlac_code(codeconfig)
        except Exception as e:
            print("Exception occured while generating mlac code: ", e)
            return {"Status": "Failure", "Msg": "Unsupported model"}
        print("mlac_code generation status is: ", outputStr)
        output = json.loads(outputStr)
        if output["Status"] == "SUCCESS":
            return output
        else:
            return {"Status": "Failure", "Msg": "MLaC code failed to generate"}
    else:
        return {"Status": "Failure", "Msg": "Code Config Not Present"}


@version_v1.post(
    "/aion/getMonitoring/inputdrift",
    response_model=driftMonitering,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Monitoring"],
    description="[README.md](/docs/preview_md?file_path=docs/monitoring.md)",
)
def moddrift(
    usecaseid: str = Query("AI0001", description="Enter the usecaseid"),
    version: str = Query("1", description="Enter the version"),
    data_path: str = Query(
        "/root/HCLT/data/storage/filename",
        description="Enter the data file path",
    ),
):
    from monitoring.monitoring_inputdrift_endpoint_handler import modinputdrift

    if not os.path.isfile(data_path):
        raise HTTPException(status_code=400, detail="File not found")

    try:
        result = modinputdrift(usecaseid, version, data_path)
        if result["Drift_analysis"]["status"] == "SUCCESS":
           html_file_path = result["Drift_analysis"]["htmlPath"]
           pdf_file_path = result.get("New_data_distribution")
           zip_filename = f"/root/HCLT/data/target/{usecaseid}/{version}/monitoring/files.zip"
           with ZipFile(zip_filename, 'w') as zip_file:
                    zip_file.write(html_file_path, arcname="myreport.html")
                    zip_file.write(pdf_file_path, arcname="monitoring.pdf")
        return FileResponse(
                    zip_filename, media_type="application/zip", filename="files.zip"
                )
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@version_v1.post(
    "/aion/getMonitoring/performance",
    response_model=driftMonitering,
    responses={404: {"model": notFound}, 500: {"model": unexpectedError}},
    tags=["Monitoring"],
)
def moddriftperformance(
    usecaseid: str = Query("AI0001", description="Enter the usecaseid"),
    version: str = Query("1", description="Enter the version"),
    data_path: str = Query(
        "/root/HCLT/data/storage/filename",
        description="Enter the data file path",
    ),
):
    from monitoring.monitoring_performance_endpoint_handler import modperformance

    if not os.path.isfile(data_path):
        raise HTTPException(status_code=400, detail="File not found")

    try:
        result = modperformance(usecaseid, version, data_path)
        if result["status"] == "SUCCESS":
            file_path = result["htmlPath"]
        return FileResponse(file_path, media_type="application/html", filename= "performance.html")
    
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@version_v1.post(
    "/aion/xplain/xplainmodel",
    response_model=xplainResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["xplain"],
    description="[README.md](/docs/preview_md?file_path=docs/xplain.md)",
)
def xplainmodel(
    usecaseid: str = Query("AI0001", description="Enter the usecaseid"),
    version: str = Query("1", description="Enter the version"),
):
    from xplain.xplain_model_endpoint import xplaininputmodel

    result = xplaininputmodel(usecaseid, version)
    file_path = result
    return FileResponse(file_path, media_type="application/html", filename= "xplainmodel.html")


@version_v1.post(
    "/aion/xplain/xplainprediction",
    response_model=xplainResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["xplain"],
)
def xplainprediction(input_data: xplain_data):
    from xplain.xplain_prediction_endpoint import xplaininputprediction

    result = xplaininputprediction(input_data)
    file_path = result
    return FileResponse(file_path, media_type="application/html", filename= "xplainprediction.html")


from issue_spotter_api.issue_spotter_caller import api_response


# @aionapp.post("/issue_spotter/")
# async def issue_spotter(file_location: str, file_name: str):
#     response = api_response(file_location, file_name)

#     return response


@version_v1.post(
    "/aion/setApiStatus/",
    response_model=setStatusResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["API status"],
    description="[README.md](/docs/preview_md?file_path=docs/set_apistatus.md)",
)
async def set_api_status(model: ApiMasterModel = Depends()):
    return setapistatus(model)


@version_v1.post(
    "/aion/uploadfile/",
    response_model=miscellaneousResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Miscellaneous"],
)
def aion_upload_file_endpoint(
    config_file_path_in_localhost: list[UploadFile],
    data_file_path_in_localhost: list[UploadFile],
    config_file_path_in_container: str = Query(
        "/root/HCLT/data/config/", description="Container path for config files"
    ),
    data_file_path_in_container: str = Query(
        "/root/HCLT/data/storage/", description="Container path for data files"
    ),
):
    result = upload_file_in_docker.upload_file_to_container(
        config_file_path_in_localhost,
        data_file_path_in_localhost,
        config_file_path_in_container,
        data_file_path_in_container,
    )
    return result


import json
import os
from datetime import datetime
import django
from django.db import connection
import os
import sys
import re
from django.core.paginator import Paginator
from appbe import compute
from appbe.pages import getversion
from appbe.aion_config import settings
import warnings
from pathlib import Path


@version_v1.post(
    "/aion/problemtypes/training",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
)
def train_for_pipeline(
    tracking_id: int,
    confFile: UploadFile = File(description="Upload config file for training"),
):
    from aion import aion_training

    filename = getFileName()
    config_path = "/root/HCLT/data/config/" + filename.replace("csv", "json")
    data = json.load(confFile.file)
    with open(config_path, "w") as json_file:
        json.dump(data, json_file)

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

        warnings.filterwarnings("ignore")
        config_path = Path(config_path)
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

        # max_id_val_query = "SELECT max(id)+1 FROM usecasedetails"

        # # get the max id+1 value from the table
        # with connection.cursor() as cursor:
        #     cursor.execute(max_id_val_query)
        #     max_id_val = cursor.fetchone()

        # print("max_id_val is : ", max_id_val[0])
        # if len(max_id_val) != 1:
        #     raise Exception("Expected one row but found 0 or more than 1 rows")

        # usecasedetails_insert_query = f"""
        #     INSERT INTO usecasedetails (id, usecasename, description, usecaseid, trackingid, datapathlocation)
        #     VALUES ({max_id_val[0]}, '{usecaseid}', '', '{usecaseid}', NULL, NULL)
        # """
        # print("usecasedetails_insert_query value is: ", usecasedetails_insert_query)
        # with connection.cursor() as cursor:
        #     cursor.execute(usecasedetails_insert_query)

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


@version_v1.post(
    "/aion/RetrainModel/",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
)
def retrain(use_case_name: str, model_version: int):
    from retrain_reconfigure.retrain_endpoint_handler import (
        retrain_call as retrain_handler,
    )

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        # Call the retrain endpoint
        result = retrain_handler(use_case_name, model_version)

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        result = {"error": f"FileNotFoundError: {e}"}
    return result


@version_v1.post(
    "/aion/ReconfigureandTrain/",
    responses={
        200: {"model": trainingResponse},
        404: {"model": notFound},
        500: {"model": unexpectedError},
        408: {"model": timeOutError},
    },
    tags=["Training"],
)
def reconfigure(
    confFile: UploadFile = File(description="Upload config file for training"),
):
    from retrain_reconfigure.reconfigure_endpoint_handler import (
        reconfigure_call as reconfigure_handler,
    )

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        # Call the retrain endpoint
        result = reconfigure_handler(confFile)

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        result = {"error": f"FileNotFoundError: {e}"}
    return result


@version_v1.post(
    "/aion/getlogs/",
    response_model=getlogResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Usecases"],
)
def read_logs(usecaseid: str, version: str):
    try:
        # Assuming your log file has a ".log" extension
        log_file_path = (
            f"/root/HCLT/data/target/{usecaseid}/{version}/log/model_training_logs.log"
        )

        with open(log_file_path, "r") as log_file:
            logs_content = log_file.read()

        return {"logs_content": logs_content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading logs file: {str(e)}"
        )


@version_v1.delete(
    "/aion/deleteusecase",
    response_model=setStatusResponse,
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Usecases"],
    description="[README.md](/docs/preview_md?file_path=docs/delete_usecase.md)",
)
def delete_usecasedetails(usecaseid: str):
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "appfe.ux.settings")
        django.setup()

        sql_query1 = f"""
            SELECT id FROM usecasedetails WHERE usecaseid = '{usecaseid}'
        """
        with connection.cursor() as cursor:
            cursor.execute(sql_query1)
            result = cursor.fetchone()
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Usecase details not found for '{usecaseid}' "
            )

        model_id = result[0]
        sql_query2 = f"""
            DELETE FROM Existusecases WHERE ModelName_id = '{model_id}'
        """
        with connection.cursor() as cursor:
            cursor.execute(sql_query2)

        sql_query3 = f"""
            DELETE FROM usecasedetails WHERE id = '{model_id}'
        """
        with connection.cursor() as cursor:
            cursor.execute(sql_query3)

        return {"msg": f"'{usecaseid}' deleted successfully"}
    except django.db.utils.DatabaseError as e:
        print(f"Database error: {e}")


@version_v1.post(
    "/aion/uploadData/stop",
    response_model=Data,
    responses={
        500: {"model": unexpectedError},
    },
    tags=["Data Handler"],
    include_in_schema=False,
)
async def stop_data_upload():
    upload_status.upload_in_progress = False
    return JSONResponse(content={"message": "Data upload stopped"})


@version_v1.get(
    "/aion/uploadData/status",
    response_model=uploadStatusResponse,
    responses={
        500: {"model": unexpectedError},
    },
    tags=["Data Handler"],
    include_in_schema=False,
)
async def get_upload_status():
    return JSONResponse(
        content={
            "upload_in_progress": upload_status.upload_in_progress,
            "data_uploaded": upload_status.data_uploaded,
        }
    )


async def upload_data(browse, dataLocation, usecaseid):
    file = browse.file
    with open(dataLocation, "wb") as dataFile:
        # if want to upload chunk
        # while True:
        #     chunk = await file.read(
        #         1024
        #     )  # Reading smaller chunks for better memory usage
        #     if not chunk:
        #         break
        #     dataFile.write(chunk)
        content = await file.read()
        dataFile.write(content)
    # await process_uploaded_data(dataLocation)
    upload_status.upload_in_progress = False
    upload_status.data_uploaded = True
    # update api status
    model = ApiMasterModel(
        api_name="DATA_UPLOAD",
        api_status="Success",
        model_version=1,
        uuid=browse.tracking_id,
        api_result_location=dataLocation,
        usecaseid=usecaseid,
    )
    setapistatus(model)


def get_file_size(file):
    # move to end
    file.seek(0, 2)
    file_size = file.tell()
    file_size = round(file_size / 1024, 2)
    # reset file pointer to the top
    file.seek(0)
    return int(file_size)


@version_v1.get(
    "/aion/get-usecase_list",
    response_model=List[dict],
    responses={
        404: {"model": notFound},
        500: {"model": unexpectedError},
    },
    tags=["Usecases"],
    description="[README.md](/docs/preview_md?file_path=docs/getusecaselist.md)",
)
def get_usecasedetails():
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "appfe.ux.settings")
        django.setup()

        sql_query = f"""
            SELECT u.usecaseid, e.Status FROM usecasedetails as u LEFT JOIN Existusecases as e ON u.id=e.ModelName_id
        """

        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Usecase details not found")
        result = [
            {
                "usecaseid": row[0],
                "status": "Trained" if row[1] == "SUCCESS" else "Not Trained",
            }
            for row in result
        ]
        return result
    except django.db.utils.DatabaseError as e:
        print(f"Database error: {e}")
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)


aionapp.include_router(version_v1, prefix="/v1")
