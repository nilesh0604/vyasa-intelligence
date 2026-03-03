# Vyasa Intelligence - Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Vyasa Intelligence to a local Rancher Desktop cluster.

## Prerequisites

1. Rancher Desktop installed with Kubernetes enabled
2. Docker container built and available locally
3. kubectl configured to use Rancher Desktop cluster

## Quick Start

### 1. Build and Tag Docker Image

```bash
# Build the image
docker build -t vyasa-intelligence:latest .

# If using a local registry (optional)
docker tag vyasa-intelligence:latest localhost:5000/vyasa-intelligence:dev
docker push localhost:5000/vyasa-intelligence:dev
```

### 2. Create Kubernetes Secret

Create a secret from your `.env` file:

```bash
# Make sure you have .env file in the project root
kubectl create secret generic vyasa-secrets \
  --from-env-file=.env \
  --namespace=vyasa \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n vyasa
kubectl get services -n vyasa
kubectl get ingress -n vyasa

# Watch the deployment rollout
kubectl rollout status deployment/vyasa-api -n vyasa
```

### 4. Test the Deployment

#### Option A: Port Forward

```bash
# Port forward the API service
kubectl port-forward svc/vyasa-api-service 8000:80 -n vyasa

# Test in another terminal
curl http://localhost:8000/health
```

#### Option B: Ingress (requires host entry)

```bash
# Add to /etc/hosts
echo "127.0.0.1 vyasa.local" | sudo tee -a /etc/hosts

# Test via ingress
curl http://vyasa.local/health
```

## Monitoring

### Check Pod Logs

```bash
# Get all pods
kubectl get pods -n vyasa

# View logs for a specific pod
kubectl logs -f deployment/vyasa-api -n vyasa

# View logs for Redis
kubectl logs -f deployment/redis -n vyasa
```

### Check HPA Status

```bash
# Watch HPA in real-time
kubectl get hpa -n vyasa -w

# Describe HPA for detailed info
kubectl describe hpa vyasa-api-hpa -n vyasa
```

### Scale Manually

```bash
# Scale up
kubectl scale deployment vyasa-api --replicas=4 -n vyasa

# Scale down
kubectl scale deployment vyasa-api --replicas=2 -n vyasa
```

## Load Testing

Use the provided Locust script to test HPA:

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --headless -u 20 -r 5 --run-time 60s \
  --host http://localhost:8000

# Watch HPA scale in another terminal
kubectl get hpa -n vyasa -w
```

## Troubleshooting

### Common Issues

1. **Image Pull Error**: Make sure the image is built and available
   ```bash
   docker images | grep vyasa-intelligence
   ```

2. **Pods CrashLooping**: Check logs for errors
   ```bash
   kubectl logs -f deployment/vyasa-api -n vyasa
   ```

3. **Secret Not Found**: Ensure secrets are created
   ```bash
   kubectl get secrets -n vyasa
   ```

4. **Ingress Not Working**: Check ingress controller
   ```bash
   kubectl get pods -n ingress-nginx
   ```

### Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Delete namespace (will delete all resources in it)
kubectl delete namespace vyasa
```

## Architecture

```
Internet
    |
Ingress (nginx)
    |
Service (vyasa-api-service)
    |
Deployment (vyasa-api) - 2-5 replicas
    |
    ├── Redis (cache)
    └── Persistent Storage (emptyDir)
```

## Resource Limits

- **API Pod**: 500m-2 CPU, 1Gi-4Gi memory
- **Redis Pod**: 100m-500m CPU, 128Mi-512Mi memory
- **HPA**: Scales 2-5 replicas based on CPU (70%) and memory (80%)
