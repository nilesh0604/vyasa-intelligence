#!/bin/bash

# M6 Verification Script - Local Kubernetes with Rancher Desktop
# This script verifies all M6 requirements are met

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verification checklist
checklist=(
    "Kubernetes cluster reachable"
    "kubectl configured"
    "Docker image built"
    "K8s manifests exist"
    "Deployment script executable"
    "Load test script ready"
)

echo -e "${BLUE}=== M6 Verification Checklist ===${NC}"
echo ""

# Check 1: Kubernetes cluster
echo -n "✓ Checking Kubernetes cluster... "
if kubectl cluster-info &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} - Is Rancher Desktop running?"
    exit 1
fi

# Check 2: kubectl configuration
echo -n "✓ Checking kubectl configuration... "
if kubectl get nodes &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Check 3: Docker image
echo -n "✓ Checking Docker image... "
if docker images | grep -q "vyasa-intelligence"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC} - Image not built yet. Run: docker build -t vyasa-intelligence:latest ."
fi

# Check 4: K8s manifests
echo -n "✓ Checking K8s manifests... "
manifests=("namespace.yaml" "configmap.yaml" "redis-deployment.yaml" "deployment.yaml" "service.yaml" "ingress.yaml" "hpa.yaml")
all_exist=true
for manifest in "${manifests[@]}"; do
    if [ ! -f "k8s/$manifest" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = true ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} - Missing manifests"
    exit 1
fi

# Check 5: Deployment script
echo -n "✓ Checking deployment script... "
if [ -x "deploy-k8s.sh" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} - Script not executable"
    exit 1
fi

# Check 6: Load test script
echo -n "✓ Checking load test script... "
if [ -f "tests/load/locustfile.py" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} - Load test script missing"
    exit 1
fi

echo ""
echo -e "${GREEN}=== All M6 Prerequisites Met! ===${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Deploy to Kubernetes: ./deploy-k8s.sh"
echo "2. Verify deployment: kubectl get pods -n vyasa"
echo "3. Test with: curl http://localhost:8000/health"
echo "4. Test HPA: locust -f tests/load/locustfile.py --headless -u 20 -r 5 --run-time 60s --host http://localhost:8000"
echo "5. Watch HPA: kubectl get hpa -n vyasa -w"
echo ""
echo -e "${YELLOW}M6 Exit Criteria:${NC}"
echo "□ 2 replicas Running"
echo "□ /health returns 200 from both pods"
echo "□ HPA scales under load"
echo "□ Rolling restart works: kubectl rollout restart deployment/vyasa-api -n vyasa"
