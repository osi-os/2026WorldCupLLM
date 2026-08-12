# 2026WorldCupLLM
Ask this LLM anything about player stats from the 2026 Summer World Cup!



*Retrieval evaluation:

I created an evaluation folder to compare different retrieval methods - text, vector, and hybrid.

After looking at the below table, I decided to go with the hybrid approach as my default search type. This 
is because the RAG pipeline passes the top 5 documents to Claude as context — the higher hit rate means the correct document lands in Claude's context more often, which is what actually drives answer quality. MRR matters most if only the top 1-2 results are being used, where rank position is critical. Since Claude is reading all 5 documents passed through, hybrid is better.


Loaded 900 ground-truth questions

Loading cached embeddings (1416 docs) from embeddings.npy
method        hit_rate         mrr
----------------------------------
text             0.904       0.876                                                                                                    
vector           0.949       0.917                                                                                                    
hybrid           0.962       0.912                                                                                                    

Best method by MRR: vector (hit_rate=0.949, mrr=0.917)


*LLM Evaluation:

Using the llm_judge.py script, I compared 3 prompt styles - concise, detailed, and baseline - using LLM as a judge.

Evaluating 3 prompt variants on 50 questions:
                                                                                                            
variant       mean_score  RELEVANT  PARTLY   NON
------------------------------------------------
concise            0.960        48       0     2
detailed           0.960        48       0     2
baseline           0.940        47       0     3

Based off of the results, I decided to use the detailed variant in as my default in the INSTRUCTIONS block
in my rag.py script. 

Before I made this decision, I had to make an edit to my llm_judge.py script. When I first ran it, I got the
below results, which weren't very accurate:
                                                                                                            
variant       mean_score  RELEVANT  PARTLY   NON
------------------------------------------------
baseline           0.720        32       8    10
detailed           0.630        30       3    17
concise            0.570        23      11    16

This is because the judge's model training cut off is before the dates of the 2026 World Cup. There were 
answers it was classifying as non-relevant (like Messi scoring goals), because 1. It believed he retired 
from international play in 2021 and 2. said the 2026 World Cup hadn't been played yet.

To fix this, I told the JUDGE_PROMPT to judge answers ONLY against the CONTEXT provided (the tournament data), and to treat the CONTEXT as the sole source of truth.