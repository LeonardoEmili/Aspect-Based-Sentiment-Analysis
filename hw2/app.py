from flask import Flask, request, jsonify

from stud.implementation import build_model_b, build_model_ab, build_model_cd

app = Flask(__name__)

model_b = build_model_b('cpu')
try:
    model_ab = build_model_ab('cpu')
except:
    model_ab = None
try:
    model_cd = build_model_cd('cpu')
except:
    model_cd = None

def prepare_data(data):
    data_a = []
    data_b = []
    for sample in data:
        data_a.append({"text": sample["text"]})
        data_b.append({"text": sample["text"], "targets": [term[0:2] for term in sample["targets"]]})
    return data_a, data_b


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):

    try:
        json_body = request.json
        inputs = json_body['samples']
        data_a, data_b  = prepare_data(inputs)

        predictions_b = model_b.predict(data_b)

        if model_ab:
            predictions_ab = model_ab.predict(data_a)
        else:
            predictions_ab = None

        if model_cd:
            predictions_cd = model_cd.predict(data_a)
        else:
            predictions_cd = None

    except Exception as e:

        app.logger.error(e, exc_info=True)
        return {'error': 'Bad request', 'message': 'There was an error processing the request. Please check logs/server.stderr'}, 400

    return jsonify(samples=inputs, predictions_b=predictions_b, predictions_ab=predictions_ab, predictions_cd=predictions_cd)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)
