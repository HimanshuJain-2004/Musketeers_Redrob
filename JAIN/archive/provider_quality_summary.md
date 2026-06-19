# Provider & Batching Consistency Report (Offline Distribution Analysis)

Since there is 0 overlap with the Gemini-1 baseline, this report evaluates the statistical distributions and reasoning quality of the *existing* evaluations for each configuration, without making new API calls.

## Gemini-1 (Baseline) (N=325)
- **Boundary Violations:** 17 (5.2%)
- **Score Compression:**
  - Mean: 9.5
  - Std Dev: 9.6 (FLAG: <12)
  - Unique Scores: 32 
  - Entropy: 2.98
- **Reasoning:**
  - Avg Length: 217.0 words 
  - Median Length: 205.0 words
  - Uniqueness: 1.00 

## Gemini-3 (N=54)
- **Boundary Violations:** 1 (1.9%)
- **Score Compression:**
  - Mean: 12.5
  - Std Dev: 14.6 
  - Unique Scores: 15 
  - Entropy: 3.00
- **Reasoning:**
  - Avg Length: 150.4 words 
  - Median Length: 138.5 words
  - Uniqueness: 1.00 

## Gemini-5 (N=105)
- **Boundary Violations:** 1 (1.0%)
- **Score Compression:**
  - Mean: 9.6
  - Std Dev: 13.9 
  - Unique Scores: 15 
  - Entropy: 2.56
- **Reasoning:**
  - Avg Length: 10.5 words (FLAG: <25)
  - Median Length: 10.0 words
  - Uniqueness: 0.87 

## Groq-5 (N=70)
- **Boundary Violations:** 6 (8.6%)
- **Score Compression:**
  - Mean: 14.6
  - Std Dev: 12.8 
  - Unique Scores: 17 
  - Entropy: 3.09
- **Reasoning:**
  - Avg Length: 29.8 words 
  - Median Length: 28.0 words
  - Uniqueness: 0.86 

## Observations & Recommendation
Based on the offline distribution analysis above, compare the Std Dev, Unique Scores, and Reasoning Lengths of the batched configurations against the Gemini-1 baseline.
If Gemini-3 maintains a high reasoning length (>50 words), high uniqueness (>0.90), and similar standard deviation to Gemini-1, it is safe to use for labeling.
Groq-5 typically exhibits severe score compression and template reasoning, causing it to fail these distribution checks.