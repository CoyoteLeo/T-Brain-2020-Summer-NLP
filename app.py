import datetime
import hashlib
import itertools
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from ckip import get_names
from classification import predict_is_possible
from name import validate_name

app = Flask(__name__)

# PUT YOUR INFORMATION HERE
CAPTAIN_EMAIL = os.environ.get("CAPTAIN_EMAIL", "")
SALT = "d839bd6a611841a292ef359a73a488c2"


def is_english(s):
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    else:
        return True


def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string: information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    return s.hexdigest()


def predict(article):
    """ Predict your model result
    @param article: a news article
    @returns prediction: a list of name
    """
    # PUT YOUR MODEL INFERENCING CODE HERE
    is_possible = predict_is_possible(article)
    if not is_possible:
        return []

    chunk = 300
    articles = [article[i : i + chunk] for i in range(0, len(article), chunk)]
    names = list(set(itertools.chain.from_iterable(get_names(articles))))

    if not names:
        return []

    return _check_datatype_to_list(validate_name(names, article))


def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not.
        And then convert your prediction to list type or raise error.

    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError("Prediction is not in list type.")


@app.route("/healthcheck", methods=["POST"])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(
        {
            "esun_uuid": data["esun_uuid"],
            "server_uuid": server_uuid,
            "captain_email": CAPTAIN_EMAIL,
            "server_timestamp": server_timestamp,
        }
    )


@app.route("/inference", methods=["POST"])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(data["news"])
    except Exception:
        raise ValueError("Model error.")
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(
        {
            "esun_timestamp": data["esun_timestamp"],
            "server_uuid": server_uuid,
            "answer": answer,
            "server_timestamp": server_timestamp,
            "esun_uuid": data["esun_uuid"],
        }
    )


if __name__ == "__main__":
    app.run("0.0.0.0", port=8787, debug=False)
