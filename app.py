from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import threading

# Initialize Flask app
app = Flask(__name__)

# existing imports & setup...

@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Trigger a background job to:
      1. Reload the latest CSV (or new data source)
      2. Re-fit tokenizer + model (with same hyperparams)
      3. Save artifacts back to disk / persistent volume
      4. (Optionally) roll your Deployment to pick up the new model
    """
    def _job():
        from spam_classifier import main as train_main
        # point to your raw CSV, same flags as before:
        train_main(
            csv_path="spam.csv",
            model_out="best_model.keras",
            epochs=10,
            test_size=0.2,
            val_size=0.1,
            seed=42,
            # …etc
        )
        # you could copy the new model into a shared PVC,
        # or use the Kubernetes Python client to rollout your Deployment.
    threading.Thread(target=_job).start()
    return jsonify({"status": "retraining started"}), 202


# Load model and tokenizer
model = tf.keras.models.load_model("best_model.keras")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    'spamapi_predictions_total',
    'Total number of predictions',
    ['label']
)
LATENCY = Histogram(
    'spamapi_request_latency_seconds',
    'Request latency for /predict endpoint'
)

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests, record metrics."""
    with LATENCY.time():
        data = request.get_json(force=True)
        text = data.get('text', '')
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=100, padding='post')
        prob = float(model.predict(pad)[0][0])
        label = 'spam' if prob > 0.5 else 'ham'
        PREDICTION_COUNT.labels(label=label).inc()
    return jsonify({'label': label, 'probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
