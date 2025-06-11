# Imports
import glob
import os
import time
import pickle

from colorama import Fore, Style
#from tensorflow import keras
from google.cloud import storage

from salesninja.params import *
from salesninja.ml.models import load_XGB_model



def save_results(params: dict, metrics: dict):
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        if not os.path.exists(os.path.dirname(params_path)):
            print(f"- Path does not exist, will create {os.path.dirname(params_path)}")
            os.makedirs(os.path.dirname(params_path))
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        if not os.path.exists(os.path.dirname(metrics_path)):
            print(f"- Path does not exist, will create {os.path.dirname(metrics_path)}")
            os.makedirs(os.path.dirname(metrics_path))
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("[Registry] Results saved locally")

    if MODEL_TARGET == "gcs":
        ##### TO DO
        """
        results_path =
        results_filename = model_path.split("/")[-1] # e.g. "20230208-161047.json" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("[Registry] Results saved to GCS")
        """
        pass

    return None


def save_model(model = None):
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.json"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.json"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.json")
    if not os.path.exists(os.path.dirname(model_path)):
        print(f"- Path does not exist, will create {os.path.dirname(model_path)}")
        os.makedirs(os.path.dirname(model_path))
    model.save_model(model_path)

    print("[Registry] Model saved locally")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.json" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("[Registry] Model saved to GCS")

    return None


def load_model(stage = "Production"):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\n- Load latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            print(Fore.BLUE + f"- No model found in local registry, initializing new one ..." + Style.RESET_ALL)
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"- Load latest model from disk..." + Style.RESET_ALL)

        ### Not sure if this is gonna work, since XGBoost model might not be a Keras model
        # latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        latest_model = load_XGB_model(most_recent_model_path_on_disk)

        print("[Registry] Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\n- Load latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        #blobs = list(client.get_bucket("/".join([BUCKET_NAME, "models"])).list_blobs())
        #blobs = list(client.get_bucket(BUCKET_NAME))
        bucket = client.bucket(BUCKET_NAME)
        ### Specific file override
        blob = bucket.blob("models/20250611-141823.json")
        ###
        print("------", blob, "-----")

        try:
            #latest_blob = max(blobs, key=lambda x: x.updated)
            latest_blob = blob
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)

            if not os.path.exists(os.path.dirname(latest_model_path_to_save)):
                os.makedirs(os.path.dirname(latest_model_path_to_save))

            latest_blob.download_to_filename(latest_model_path_to_save)

            #latest_model = keras.models.load_model(latest_model_path_to_save)

            latest_model = load_XGB_model(latest_model_path_to_save)

            print("[Registry] Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"[Registry] No model found in GCS bucket {BUCKET_NAME}")

            return None
    else:
        return None

def save_synth_model(model = None):
    pass

def load_synth_model(stage = "Production"):
    pass

def save_synth_metadata(metadata = None):
    pass

def load_synth_metadata(metadata = None):
    pass
