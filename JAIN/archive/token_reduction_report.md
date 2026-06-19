# Throughput Experiment 2: Token Reduction

Testing reduction of the 'reasoning' field to save output tokens and reduce latency/cost.

## Token Savings (Batch Size 5)
- **Baseline Avg Output Tokens**: 637
- **Reduced Avg Output Tokens**: 476
- **Token Savings**: 162 tokens/request (25.4% reduction)

## Reasoning Word Count
- **Baseline Avg Words**: 32.0
- **Reduced Avg Words**: 3.6

## JSON Validity
- **Parse Success Rate**: 100%

## Sample Outputs
1. "No relevant AI experience"
2. "No relevant AI experience"
3. "Some data engineering experience"

## Recommendation
The token-reduced prompt successfully forces the model to produce much shorter reasoning fields, saving significant output tokens without breaking JSON structure. This change should be merged into the production system prompt.
