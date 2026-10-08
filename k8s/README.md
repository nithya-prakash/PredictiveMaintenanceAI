# Kubernetes manifests

Kustomize base for the API + Postgres (the Streamlit dashboard is not included).

```bash
docker build -f docker/Dockerfile.api -t predictive-maintenance-api:local .
kind load docker-image predictive-maintenance-api:local      # or: minikube image load ...
kubectl apply -f k8s/namespace.yaml
kubectl -n predictive-maintenance create secret generic pdm-secrets \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)" --from-literal=API_TOKEN="$(openssl rand -hex 16)"
kubectl apply -k k8s
```

Status: schema-validated in CI (`kubeconform`, see `.github/workflows/ci.yml`). Not yet applied to a live
cluster, so probe timings and the HPA (needs metrics-server) are untested. Postgres is a single-replica
StatefulSet with a 1 GiB PVC (demo-grade, no backups). A Prometheus scrape config for the pod annotations is not included.
