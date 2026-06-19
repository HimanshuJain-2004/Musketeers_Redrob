# Throughput Experiment 1: Multi-Candidate Requests

Testing evaluating multiple candidates per API request to maximize throughput per API key (tested via Groq API).

| Batch Size | Avg Latency (s) | Avg Input Tokens | Avg Output Tokens | JSON Parse Success | Candidates Matching N |
|------------|-----------------|------------------|-------------------|--------------------|-----------------------|
| 5 | 27.28 | 8430 | 638 | 100% | 100% |
| 10 | 0.00 | 0 | 0 | 0% | 0% |
| 15 | 0.00 | 0 | 0 | 0% | 0% |

## Throughput Projections
Assuming an API rate limit of 100,000 tokens/day per key:
- **Batch Size 5**: ~55 candidates/day/key
- **Batch Size 10**: ~0 candidates/day/key
- **Batch Size 15**: ~0 candidates/day/key

## Recommendation
Based on the results, a batch size of **5** is recommended for production rollout as it maintains 100% validity while multiplying throughput.
