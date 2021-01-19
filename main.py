
from flask_cors import CORS
from flask import Flask, render_template, request, json, jsonify
import os
import logging
import traceback
from detection import GMM_Detector, LSTMDetector

import config

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', level=logging.INFO)

template_dir = os.path.abspath('views')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

gmm_detector = GMM_Detector()
lstm_detector = LSTMDetector()


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        json_data = request.json
        if "token" in json_data:
            token = json_data["token"].strip()
            if token not in config.API_TOKENS:
                return jsonify(
                    success=False,
                    msg="Failed to authenticate token!")
        else:
            return jsonify(
                success=False,
                msg="Failed to authenticate!")

        if "alg" not in json_data:
            return jsonify(
                success=False,
                msg="Missing field: search_type"
            )

        if "audio_path" not in json_data:
            return jsonify(
                success=False,
                msg="Missing field: audio_path"
            )

        alg = json_data["alg"]
        audio_path = json_data["audio_path"]

        if alg == "gmm":
            segments = gmm_detector.predict(audio_path)
        else:
            segments = lstm_detector.predict(audio_path)

        result = []
        for start, end in segments:
            dur = end - start
            if dur <= 1:
                label = "short"
            else:
                label = "long"

            result.append({"start": start, "end": end, "label": label})

        with open(audio_path.replace(".wav", ".txt"), "w") as f:
            for start, end in segments:
                f.write("{}\t{}\n".format(start, end))

        if result is not None:
            return jsonify(
                success=True,
                result=result,
                msg="Success"
            )

        return jsonify(
            success=False,
            msg="Error! Time out"
        )

    except:
        logging.error(traceback.format_exc())
    return jsonify(
        success=False,
        msg="Error! Something wrong"
    )


if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)
