# Spam-Mail-Classifier
An end to end ML Flow

# Spam Mail Classifier & Monitoring

This repository contains a complete end-to-end solution for training, serving, and monitoring a spam-mail classifier using:

* **Python** (data preparation & model training)
* **Flask** (REST API with `/predict` & `/metrics` endpoints)
* **Docker** (containerization)
* **Kubernetes** (Deployment, Service, CronJob)
* **Prometheus** (scraping custom & system metrics)
* **Grafana** (dashboarding)

---

## Repository Structure

```plain
📦 Spam-Mail-Classifier
├── data/                      # raw & processed datasets
│   └── ...
├── processed_data/            # train/val/test splits, tokenizer
├── best_model.keras           # trained Keras model
├── tokenizer.pickle           # fitted tokenizer
├── app.py                     # Flask application
├── Train.py                   # data prep & training CLI
├── Dockerfile                 # Docker image spec
├── deployment.yaml            # K8s Deployment & Service for Spam API
├── service.yaml               # (alternative) K8s Service manifest
├── retrain-cronjob.yaml       # K8s CronJob to retrain the model periodically
├── extra-scrape-configs.yaml  # Prometheus scrape configs for Kubernetes
├── Grafana-deployment.yaml    # Grafana Deployment & Service in K8s
├── requirements.txt           # Python dependencies
└── README.md                  # this file
```

---

## Quickstart

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/spam-mail-classifier.git
cd spam-mail-classifier
```

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data & Train Model

```bash
python Train.py \
  --csv spam.csv \
  --output-dir data \
  --num-words 10000 \
  --maxlen 100 \
  --test-size 0.2 \
  --val-size 0.1 \
  --seed 42
```

### 3. Build & Push Docker Image

```bash
docker build -t <dockerhub-username>/spam-api:latest .
docker push <dockerhub-username>/spam-api:latest
```

### 4. Deploy to Kubernetes

Make sure Docker Desktop’s Kubernetes is enabled (or point `kubectl` to your cluster).

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Check pods and service:

```bash
kubectl get pods -l app=spam-api
kubectl get svc spam-api-service
```

### 5. Test the API

Port-forward or use the NodePort:

```bash
# Port-forward locally:
kubectl port-forward svc/spam-api-service 5000:80
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Free tickets waiting for you"}'
```

### 6. Monitoring with Prometheus & Grafana

1. **Prometheus**

   ```bash
   helm install prometheus prometheus-community/prometheus \
     --set server.persistentVolume.enabled=false
   kubectl port-forward svc/prometheus-server 9090:80
   ```

2. **Grafana**

   ```bash
   helm install grafana grafana/grafana \
     --set service.type=NodePort \
     --set service.nodePort=30081 \
     --set adminPassword="letmein"
   kubectl port-forward svc/grafana 30082:80
   ```

   * Access Grafana at `http://localhost:30082`
   * Add Prometheus (URL: `http://prometheus-server.monitoring.svc.cluster.local:80`)
   * Import community dashboards (e.g. Node Exporter Full)
   * Build custom panels for `spamapi_predictions_total`

---

## Next Steps & Customization

* **Autoscaling**: Add a HorizontalPodAutoscaler on CPU or custom Prometheus metrics.
* **CI/CD**: Automate image builds & `kubectl apply` with GitHub Actions.
* **Alerting**: Define Prometheus alerting rules and hook Alertmanager.
* **Persistence**: Enable persistent volumes for Prometheus & Grafana.
* **Ingress**: Use an ingress controller (e.g., NGINX) for external access.

---

> *Happy monitoring and spam‑fighting!*
