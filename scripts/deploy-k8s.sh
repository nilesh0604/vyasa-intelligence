#!/bin/bash
# Deployment Script for Kubernetes

set -e

# Configuration
NAMESPACE=${NAMESPACE:-vyasa}
ENVIRONMENT=${ENVIRONMENT:-staging}
IMAGE_TAG=${IMAGE_TAG:-latest}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-ghcr.io}
IMAGE_NAME=${IMAGE_NAME:-vyasa-intelligence}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_warn "Helm is not installed. Using kubectl only."
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace $NAMESPACE environment=$ENVIRONMENT --overwrite
}

# Deploy application
deploy_application() {
    log_info "Deploying application to $ENVIRONMENT environment..."
    
    # Update image tag in deployment
    sed -i.bak "s|image: .*vyasa-intelligence.*|image: $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml
    
    # Apply configurations
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/configmap.yaml -n $NAMESPACE
    kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE
    
    # Restore original deployment.yaml
    mv k8s/deployment.yaml.bak k8s/deployment.yaml
    
    log_info "Deployment manifests applied"
}

# Wait for rollout
wait_for_rollout() {
    log_info "Waiting for deployment rollout..."
    
    kubectl rollout status deployment/vyasa-api -n $NAMESPACE --timeout=300s
    
    if [ $? -eq 0 ]; then
        log_info "Rollout completed successfully"
    else
        log_error "Rollout failed"
        exit 1
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get service vyasa-api -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL=$(kubectl get service vyasa-api -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ -z "$SERVICE_URL" ]; then
        log_warn "No external IP found. Using port-forward for testing..."
        kubectl port-forward service/vyasa-api 8080:8000 -n $NAMESPACE &
        PF_PID=$!
        sleep 5
        SERVICE_URL="http://localhost:8080"
    fi
    
    # Health check
    if curl -f "$SERVICE_URL/health" &> /dev/null; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        if [ ! -z "$PF_PID" ]; then
            kill $PF_PID
        fi
        exit 1
    fi
    
    # Simple query test
    if curl -f -X POST "$SERVICE_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "Who is the father of the Pandavas?"}' &> /dev/null; then
        log_info "Query test passed"
    else
        log_warn "Query test failed (might be expected if data not loaded)"
    fi
    
    # Clean up port-forward
    if [ ! -z "$PF_PID" ]; then
        kill $PF_PID
    fi
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    
    kubectl get pods -n $NAMESPACE -l app=vyasa-api
    echo ""
    
    kubectl get services -n $NAMESPACE -l app=vyasa-api
    echo ""
    
    kubectl get hpa -n $NAMESPACE -l app=vyasa-api
    echo ""
    
    # Show recent logs
    log_info "Recent logs:"
    kubectl logs -n $NAMESPACE -l app=vyasa-api --tail=20
}

# Cleanup on exit
cleanup() {
    if [ ! -z "$PF_PID" ]; then
        kill $PF_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    
    check_prerequisites
    create_namespace
    deploy_application
    wait_for_rollout
    run_smoke_tests
    show_status
    
    log_info "Deployment completed successfully! 🎉"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --namespace    Kubernetes namespace (default: vyasa)"
            echo "  -e, --environment  Environment (staging|production, default: staging)"
            echo "  -t, --tag          Docker image tag (default: latest)"
            echo "  -r, --registry     Docker registry (default: ghcr.io)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
