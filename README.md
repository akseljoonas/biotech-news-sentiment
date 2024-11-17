# Press Release Sentiment Analysis


## Braindump


### Huggingface models I have tried (bolded is the best):
I looked at datasets tuned on `financial phrasebank` and tried them out on our dataset. The performance was mediocre and there was no clear winner. Bolded are the best performing models from what I remember. I fine-tuned the current best performing model (f1 0.685) on the first model in this list.

- **mrm8488/deberta-v3-ft-financial-news-sentiment-analysis**
- **mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis**
- nickmuchi/distilroberta-finetuned-financial-text-classification
- marcev/financebert
- ahmedrachid/FinancialBERT-Sentiment-Analysis
- yiyanghkust/finbert-tone
- ProsusAI/finbert
- Narsil/finbert2
- StephanAkkerman/FinTwitBERT-sentiment
- nickmuchi/sec-bert-finetuned-finance-classification

## Papers I have read

- FINBERT (good paper, easy to read and gives good overview): https://arxiv.org/abs/1908.10063
- Classification of different fin data but very solid methodology: https://pdfs.semanticscholar.org/95e7/29cb065b9e274c609478376b8ad93982b123.pdf
- IJCAI 2024 FinLLM Challenge: https://huggingface.co/spaces/TheFinAI/IJCAI-2024-FinLLM-Learderboard 
- PIXIU financial tailtored LLMs, instruction tuning datasets, and evaluation benchmarks: https://github.com/The-FinAI/PIXIU
- FinBen is a benchmark for financial LLMs. https://arxiv.org/abs/2402.12659 
- Id look at FINBERT cites for papers https://www.semanticscholar.org/paper/FinBERT-%3A-A-Large-Language-Model-for-Extracting-Huang-Wang/8798b3a01c29fe0ce45a271bedd934787343dfb5?sort=relevance 
- There is an interesting idea of taking the last token representation of a Decoder LLM and training a classifier on top of it so you can take advantage of the big and good llms we have. https://arxiv.org/abs/2311.01239


## BERT based models we could try
- roBERTa: https://arxiv.org/abs/1907.11692
- DEBERTA: https://arxiv.org/abs/2006.03654
- DistilBERT: https://arxiv.org/abs/1910.01108


## Interesting stuff
- Different use cases of LLMs in finance: https://github.com/hananedupouy/LLMs-in-Finance
- NLP Papers applcibale to finance: https://github.com/maximedb/nlp_papers 
