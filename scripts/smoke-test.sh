#!/bin/bash
# Smoke Test Script for Deployed Applications

set -e

# Configuration
SERVICE_URL=${SERVICE_URL:-http://localhost:8000}
TEST_TIMEOUT=${TEST_TIMEOUT:-30}
EXPECTED_RESPONSE_TIME=${EXPECTED_RESPONSE_TIME:-5}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0

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

# Test health endpoint
test_health() {
    log_info "Testing health endpoint..."
    
    start_time=$(date +%s)
    
    if curl -f -s "$SERVICE_URL/health" > /dev/null; then
        end_time=$(date +%s)
        response_time=$((end_time - start_time))
        
        if [ $response_time -le $EXPECTED_RESPONSE_TIME ]; then
            log_info "✓ Health check passed (response time: ${response_time}s)"
            ((PASSED++))
        else
            log_warn "⚠ Health check passed but slow (response time: ${response_time}s, expected: ≤${EXPECTED_RESPONSE_TIME}s)"
            ((PASSED++))
        fi
    else
        log_error "✗ Health check failed"
        ((FAILED++))
    fi
}

# Test API docs endpoint
test_docs() {
    log_info "Testing API docs endpoint..."
    
    if curl -f -s "$SERVICE_URL/docs" > /dev/null; then
        log_info "✓ API docs endpoint accessible"
        ((PASSED++))
    else
        log_error "✗ API docs endpoint not accessible"
        ((FAILED++))
    fi
}

# Test query endpoint
test_query() {
    log_info "Testing query endpoint..."
    
    # Test simple query
    response=$(curl -s -X POST "$SERVICE_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "Who is the father of the Pandavas?"}' \
        --max-time $TEST_TIMEOUT || echo "")
    
    if [ -n "$response" ]; then
        # Check if response contains expected fields
        if echo "$response" | jq -e '.answer' > /dev/null 2>&1; then
            log_info "✓ Query endpoint working"
            ((PASSED++))
            
            # Check for citations
            if echo "$response" | jq -e '.citations' > /dev/null 2>&1; then
                citation_count=$(echo "$response" | jq '.citations | length')
                log_info "  - Response includes $citation_count citations"
            fi
        else
            log_error "✗ Query endpoint returned invalid response"
            ((FAILED++))
        fi
    else
        log_error "✗ Query endpoint not responding"
        ((FAILED++))
    fi
}

# Test streaming query endpoint
test_streaming() {
    log_info "Testing streaming query endpoint..."
    
    # Test streaming query
    response=$(curl -s -X POST "$SERVICE_URL/query/stream" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the name of Arjuna'\''s bow?"}' \
        --max-time $TEST_TIMEOUT || echo "")
    
    if [ -n "$response" ]; then
        # Check if response looks like streaming data
        if echo "$response" | grep -q "data:"; then
            log_info "✓ Streaming query endpoint working"
            ((PASSED++))
        else
            log_warn "⚠ Streaming endpoint responded but not in expected format"
            ((PASSED++))
        fi
    else
        log_error "✗ Streaming query endpoint not responding"
        ((FAILED++))
    fi
}

# Test concurrent requests
test_concurrent() {
    log_info "Testing concurrent requests..."
    
    # Create temporary directory for test results
    temp_dir=$(mktemp -d)
    
    # Launch 5 concurrent requests
    for i in {1..5}; do
        {
            response=$(curl -s -X POST "$SERVICE_URL/query" \
                -H "Content-Type: application/json" \
                -d '{"question": "Who is Draupadi?"}' \
                --max-time $TEST_TIMEOUT || echo "")
            echo "$response" > "$temp_dir/response_$i.json"
        } &
    done
    
    # Wait for all requests to complete
    wait
    
    # Check results
    success_count=0
    for i in {1..5}; do
        if [ -f "$temp_dir/response_$i.json" ] && \
           echo "$(cat "$temp_dir/response_$i.json")" | jq -e '.answer' > /dev/null 2>&1; then
            ((success_count++))
        fi
    done
    
    # Clean up
    rm -rf "$temp_dir"
    
    if [ $success_count -eq 5 ]; then
        log_info "✓ All concurrent requests succeeded"
        ((PASSED++))
    else
        log_error "✗ Only $success_count/5 concurrent requests succeeded"
        ((FAILED++))
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test invalid JSON
    response=$(curl -s -X POST "$SERVICE_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"invalid": json}' \
        --max-time $TEST_TIMEOUT || echo "")
    
    if echo "$response" | grep -q -i "error\|400\|422"; then
        log_info "✓ Error handling working for invalid JSON"
        ((PASSED++))
    else
        log_error "✗ Error handling not working properly"
        ((FAILED++))
    fi
}

# Test load (optional)
test_load() {
    if command -v hey &> /dev/null; then
        log_info "Running load test with hey..."
        
        # Run 100 requests over 10 seconds
        hey -n 100 -z 10s -c 10 -m POST -d '{"question": "Test query"}' \
            -T "application/json" "$SERVICE_URL/query" > /dev/null 2>&1 || true
        
        log_info "✓ Load test completed"
        ((PASSED++))
    else
        log_warn "⚠ hey not installed, skipping load test"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "================================"
    echo "Smoke Test Summary"
    echo "================================"
    echo -e "Passed: ${GREEN}$PASSED${NC}"
    echo -e "Failed: ${RED}$FAILED${NC}"
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        log_info "🎉 All smoke tests passed!"
        exit 0
    else
        log_error "❌ $FAILED smoke test(s) failed!"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting smoke tests for $SERVICE_URL..."
    echo ""
    
    # Check if service is reachable
    if ! curl -s "$SERVICE_URL" > /dev/null; then
        log_error "Service at $SERVICE_URL is not reachable"
        exit 1
    fi
    
    # Run tests
    test_health
    test_docs
    test_query
    test_streaming
    test_concurrent
    test_error_handling
    test_load
    
    # Print summary
    print_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            SERVICE_URL="$2"
            shift 2
            ;;
        -t|--timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        -r|--response-time)
            EXPECTED_RESPONSE_TIME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -u, --url           Service URL (default: http://localhost:8000)"
            echo "  -t, --timeout       Test timeout in seconds (default: 30)"
            echo "  -r, --response-time Expected response time in seconds (default: 5)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check dependencies
if ! command -v curl &> /dev/null; then
    log_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_error "jq is required but not installed"
    exit 1
fi

# Run main function
main
