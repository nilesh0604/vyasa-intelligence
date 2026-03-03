#!/bin/bash
# Quality Gate Evaluation Script for CI/CD

set -e

echo "🚀 Starting Quality Gate Evaluation..."

# Check if required environment variables are set
if [ -z "$EVALUATION_RESULTS_FILE" ]; then
    EVALUATION_RESULTS_FILE="data/evaluation/ci_results.json"
fi

# Create evaluation directory if it doesn't exist
mkdir -p "$(dirname "$EVALUATION_RESULTS_FILE")"

# Run evaluation pipeline if results file doesn't exist
if [ ! -f "$EVALUATION_RESULTS_FILE" ]; then
    echo "📊 Running evaluation pipeline..."
    uv run python -m src.evaluation.evaluator \
        --output "$EVALUATION_RESULTS_FILE" \
        --sample-size 10
fi

# Check if results file exists
if [ ! -f "$EVALUATION_RESULTS_FILE" ]; then
    echo "❌ Evaluation results file not found: $EVALUATION_RESULTS_FILE"
    exit 1
fi

# Run quality gate evaluation
echo "🔍 Evaluating quality gates..."
uv run python -c "
import json
import sys
from src.evaluation.quality_gates import QualityGateEvaluator

# Load evaluation results
with open('$EVALUATION_RESULTS_FILE') as f:
    results = json.load(f)

# Extract scores (handle both single result and batch results)
if isinstance(results, list):
    # Use average scores from batch results
    scores = {}
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_similarity']:
        values = [r.get(metric) for r in results if metric in r and r.get(metric) is not None]
        scores[metric] = sum(values) / len(values) if values else 0.0
else:
    # Single result
    scores = results

# Initialize evaluator
evaluator = QualityGateEvaluator()

# Evaluate quality gates
result = evaluator.evaluate(scores)

# Print results
print('=' * 60)
print('QUALITY GATE EVALUATION RESULTS')
print('=' * 60)
print(f'Overall Status: {\"✓ PASSED\" if result[\"overall_passed\"] else \"✗ FAILED\"}')
print(f'Overall Score: {result[\"overall_score\"]:.2%}')
print(f'Pass Rate: {result[\"summary\"][\"pass_rate\"]:.1%}')
print()

# Print individual gate results
print('GATE RESULTS:')
for gate in result['all_results']:
    status = '✓' if gate['passed'] else '✗'
    if gate['value'] is not None:
        print(f'  {status} {gate[\"name\"]}: {gate[\"value\"]:.3f} (threshold: {gate[\"threshold\"]:.2f})')
    else:
        print(f'  ✗ {gate[\"name\"]}: NOT EVALUATED')

print()

# Print improvement suggestions if any gates failed
if not result['overall_passed']:
    suggestions = evaluator.get_improvement_suggestions(result)
    if suggestions:
        print('IMPROVEMENT SUGGESTIONS:')
        for i, suggestion in enumerate(suggestions, 1):
            print(f'  {i}. {suggestion}')
        print()

# Save detailed report
report = evaluator.generate_report(result)
with open('data/evaluation/quality_gate_report.txt', 'w') as f:
    f.write(report)

print('Detailed report saved to: data/evaluation/quality_gate_report.txt')

# Exit with error code if quality gates failed
if not result['overall_passed']:
    print('❌ Quality gates failed!')
    sys.exit(1)
elif result['overall_score'] < 0.8:
    print(f'❌ Overall score {result[\"overall_score\"]:.2%} is below 80% threshold')
    sys.exit(1)
else:
    print('✅ All quality gates passed!')
"

echo "✅ Quality gate evaluation completed successfully!"
