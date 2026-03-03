#!/bin/bash

# Vyasa Intelligence - Kubernetes Deployment Script
# This script automates deployment to Rancher Desktop Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="vyasa"
DOCKER_IMAGE="vyasa-intelligence:latest"
LOCAL_REGISTRY="localhost:5000/vyasa-intelligence:dev"

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
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_error ".env file not found. Please create it from .env.example"
        exit 1
    fi
    
    # Check if Kubernetes cluster is reachable
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Is Rancher Desktop running?"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully ✓"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Optional: Push to local registry
push_to_registry() {
    if command -v nerdctl &> /dev/null; then
        log_info "Pushing to local registry (nerdctl)..."
        nerdctl tag $DOCKER_IMAGE $LOCAL_REGISTRY
        nerdctl push $LOCAL_REGISTRY
        log_info "Image pushed to local registry ✓"
    elif docker info | grep -q "Registry: localhost:5000"; then
        log_info "Pushing to local registry (docker)..."
        docker tag $DOCKER_IMAGE $LOCAL_REGISTRY
        docker push $LOCAL_REGISTRY
        log_info "Image pushed to local registry ✓"
    else
        log_warn "Local registry not found. Skipping registry push."
    fi
}

# Create Kubernetes secret
create_secret() {
    log_info "Creating Kubernetes secret from .env file..."
    
    kubectl create secret generic vyasa-secrets \
        --from-env-file=.env \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    if [ $? -eq 0 ]; then
        log_info "Secret created/updated successfully ✓"
    else
        log_error "Failed to create secret"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_to_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Apply all manifests
    kubectl apply -f k8s/
    
    if [ $? -eq 0 ]; then
        log_info "Manifests applied successfully ✓"
    else
        log_error "Failed to apply manifests"
        exit 1
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for Redis
    kubectl wait --for=condition=available deployment/redis -n $NAMESPACE --timeout=60s
    
    # Wait for API
    kubectl wait --for=condition=available deployment/vyasa-api -n $NAMESPACE --timeout=120s
    
    if [ $? -eq 0 ]; then
        log_info "Deployment is ready ✓"
    else
        log_error "Deployment failed to become ready"
        exit 1
    fi
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo "----------------------------------------"
    
    echo "Namespace: $NAMESPACE"
    echo ""
    
    echo "Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    echo ""
    
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo ""
    
    echo "Ingress:"
    kubectl get ingress -n $NAMESPACE
    echo ""
    
    echo "HPA:"
    kubectl get hpa -n $NAMESPACE
    echo ""
    
    echo "----------------------------------------"
    log_info "Deployment completed successfully! 🎉"
    echo ""
    echo "To test the deployment:"
    echo "1. Port forward: kubectl port-forward svc/vyasa-api-service 8000:80 -n $NAMESPACE"
    echo "2. Then test: curl http://localhost:8000/health"
    echo ""
    echo "Or add '127.0.0.1 vyasa.local' to /etc/hosts and use:"
    echo "curl http://vyasa.local/health"
}

# Main execution
main() {
    log_info "Starting Vyasa Intelligence Kubernetes deployment..."
    echo ""
    
    check_prerequisites
    echo ""
    
    build_image
    echo ""
    
    push_to_registry
    echo ""
    
    create_secret
    echo ""
    
    deploy_to_k8s
    echo ""
    
    wait_for_deployment
    echo ""
    
    show_status
}

# Handle script interruption
trap 'log_error "Deployment interrupted!"; exit 1' INT

# Run main function
main "$@"
