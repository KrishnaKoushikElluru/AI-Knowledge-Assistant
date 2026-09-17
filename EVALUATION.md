# Retrieval Evaluation

Local evaluation run using Ollama (llama3.2 generation, nomic-embed-text embeddings) — this script is for local evaluation only; the deployed app continues to use Gemini.

4 documents combined into one index, 111 total chunks. 21 test questions (5 per document + 1 deliberate out-of-scope question).

**Retrieval recall@4:** 20/20 — correct source document present in the top-4 retrieved chunks.
**Answer correctness (keyword check):** 18/21 — generated answer contains an expected fact.

| # | Question | Expected source | Retrieved sources | Retrieval hit | Answer OK | Answer |
|---|---|---|---|---|---|---|
| 1 | What is the submission deadline for the Lunorsoft Round 1 assignment? | Lunorsoft Recruitments (ROUND 1).pdf | Lunorsoft Recruitments (ROUND 1).pdf | yes | yes | The submission deadline for the Lunorsoft Round 1 assignment is 18 September 2026 at 12:00 PM IST. |
| 2 | What suggested technologies are listed for the AI Developer Option 1 assignment? | Lunorsoft Recruitments (ROUND 1).pdf | Lunorsoft Recruitments (ROUND 1).pdf | yes | yes | According to the provided documents, the suggested technologies for the AI Developer Option 1 assignment are:  1. Python |
| 3 | What is the core requirement for AI Developer Option 2? | Lunorsoft Recruitments (ROUND 1).pdf | Lunorsoft Recruitments (ROUND 1).pdf | yes | NO | The core requirement for AI Developer Option 2 is to "Prepare a small programming or learning dataset." |
| 4 | What are the three rounds in the Lunorsoft recruitment process? | Lunorsoft Recruitments (ROUND 1).pdf | Lunorsoft Recruitments (ROUND 1).pdf | yes | yes | The three rounds in the Lunorsoft recruitment process are:  1. Round 1: Assignment Task 2. Round 2: Video Explanation 3. |
| 5 | What existing platform should UI/UX Designer candidates review? | Lunorsoft Recruitments (ROUND 1).pdf | Lunorsoft Recruitments (ROUND 1).pdf | yes | NO | The existing platform that UI/UX Designer candidates should review is the Lunorsoft platform. |
| 6 | What does the S-F-V-C framework stand for? | Document Creation.pdf | BC module1.pdf, Document Creation.pdf | yes | yes | The S-F-V-C framework stands for Search - Find the right source - Verify - Confirm. |
| 7 | What accuracy did the AI attendance system achieve? | Document Creation.pdf | Document Creation.pdf | yes | yes | The AI attendance system achieved an accuracy of 92%. |
| 8 | What five things should a standard MoM template capture? | Document Creation.pdf | Document Creation.pdf, Lunorsoft Recruitments (ROUND 1).pdf | yes | yes | A standard MoM template should capture the following five things clearly:  1. When/where the meeting happened 2. Who att |
| 9 | What technologies were used to build the student attendance monitoring system? | Document Creation.pdf | Document Creation.pdf | yes | yes | Python and OpenCV were used to build the student attendance monitoring system. |
| 10 | What does version v2.0 mean in the standard document versioning scheme? | Document Creation.pdf | Document Creation.pdf | yes | yes | In the standard document versioning scheme, v2.0 means a Major revision. |
| 11 | How many transaction nodes are in the Elliptic++ dataset used in this project? | Bitcoin_Transaction_Forensics_Review_Report.pdf | Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | According to Table 1, the Elliptic++ dataset contains 203,769 transaction nodes. |
| 12 | Who introduced the Elliptic++ dataset and in what year? | Bitcoin_Transaction_Forensics_Review_Report.pdf | Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | The answer is available in the provided documents. According to section 8.2, Elmougy and Liu introduced Elliptic++. |
| 13 | What evaluation metrics will be used given the class imbalance? | Bitcoin_Transaction_Forensics_Review_Report.pdf | Bitcoin_Transaction_Forensics_Review_Report.pdf, Document Creation.pdf, Lunorsoft Recruitments (ROUND 1).pdf | yes | yes | The answer is: F1-score, recall, precision, ROC-AUC, and PR-AUC, with illicit-class F1, recall, precision, and PR-AUC tr |
| 14 | What percentage of transactions in the dataset are illicit? | Bitcoin_Transaction_Forensics_Review_Report.pdf | Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | According to the provided context, illicit transactions account for approximately 2.2 per cent of all transaction nodes. |
| 15 | What dataset does this Bitcoin forensics project use? | Bitcoin_Transaction_Forensics_Review_Report.pdf | Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | The Elliptic++ dataset. |
| 16 | Who published the Bitcoin whitepaper and when? | BC module1.pdf | BC module1.pdf, Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | Satoshi Nakamoto published the Bitcoin whitepaper on 31st October 2008. |
| 17 | What are the three properties in the CAP theorem? | BC module1.pdf | BC module1.pdf | yes | yes | The three properties in the CAP theorem are:  1. Consistency 2. Availability 3. Partition tolerance |
| 18 | What is a satoshi? | BC module1.pdf | BC module1.pdf, Bitcoin_Transaction_Forensics_Review_Report.pdf | yes | yes | A satoshi is currently the smallest unit of the Bitcoin currency recorded on the blockchain. It is one hundred millionth |
| 19 | What is Blockchain 2.0 associated with? | BC module1.pdf | BC module1.pdf | yes | NO | The answer is not available in the provided documents. |
| 20 | What are the three types of network architecture mentioned for blockchain networks? | BC module1.pdf | BC module1.pdf | yes | yes | According to the context, the three types of network architecture or models mentioned for blockchain networks are:  1. C |
| 21 | What is the capital of France? | (out-of-scope) | BC module1.pdf, Document Creation.pdf | - | yes | The answer is not available in the provided documents. |