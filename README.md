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