import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = 0.2
NUMBER_OF_ROWS = 3406088
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".salesninja", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".salesninja", "training_outputs")
MODEL_TARGET = "local" #local, gcs
GCP_SALESNINJA = "le-wagon-data-bootcamp"
GCP_REGION = "europe-west1"
BQ_DATASET = "dashboard_data"
BQ_REGION = "EU"
BUCKET_NAME = "sales_ninja_bucket"
#INSTANCE = "<<<Salesninja Name of Compute Engine / Virtual Machine GoogleCloud>>>"

MLFLOW_TRACKING_URI = "https://mlflow.lewagon.ai"
MLFLOW_EXPERIMENT = "lewagon1998-DSAI-ninja"
MLFLOW_MODEL_NAME = "lewagon1998-DSAI-ninja-v0.1"

PREFECT_FLOW_NAME = "salesninja_lifecycle"
PREFECT_LOG_LEVEL = "WARNING"

#EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
#GAR_IMAGE = os.environ.get("GAR_IMAGE")
#GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
#LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "raw_data")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

# Docker port
GAR_IMAGE="salesninja"
GAR_MEMORY="2Gi"
PORT=8000

## API on Gcloud Run
SERVICE_URL=""

COLUMN_NAMES_DASHBOARD = ["SalesKey","DateKey","ChannelKey","StoreKey","ProductKey","PromotionKey",
                          "UnitCost","UnitPrice","SalesQuantity","ReturnQuantity","ReturnAmount",
                          "DiscountQuantity","DiscountAmount","TotalCost","SalesAmount","ChannelKey",
                          "ChannelName","PromotionName","PromotionType","CalendarYear","CalendarQuarterLabel",
                          "CalendarMonthLabel","CalendarDayOfWeekLabel","MonthNumber",
                          "CalendarDayOfWeekNumber","ProductName","ProductSubcategoryKey",
                          "ProductSubcategoryName","ProductCategoryKey","ProductCategoryName",
                          "GeographyKey","StoreType","StoreName","ContinentName"]

#DTYPES_RAW = {
#    "example": "float32",
#}

#DTYPES_PROCESSED = np.float32
