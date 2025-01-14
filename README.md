# Press Release Sentiment Analysis
In the US, 62% of adults own stocks, whether treasury bonds, stocks, commodities, or any other. The information about public companies such as earnings releases, product announcements, or financial metrics is easily accessible through the internet.

While most of those press releases consist of objective information, a clear sentiment can be extracted as the analysts almost always give an opinion on said news. This project aims to utilize automated sentiment analysis with Large Language Models to try to outpace the market and be the first in the queue to react to new information, trying to predict the market response and profit from shorting/longing the company mentioned in the news.


## 🤗 Huggingface models used in this research:

- **Best f1 score: mrm8488/deberta-v3-ft-financial-news-sentiment-analysis**
- **Best Gross Profit: ProsusAI/finbert**
- **Base model: google-bert/bert-base-uncased**
- mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- nickmuchi/distilroberta-finetuned-financial-text-classification
- ncbi/MedCPT-Article-Encoder
- dmis-lab/biobert-v1.1
- microsoft/deberta-v3-base
- marcev/financebert
- ahmedrachid/FinancialBERT-Sentiment-Analysis
- yiyanghkust/finbert-tone
- Narsil/finbert2
- StephanAkkerman/FinTwitBERT-sentiment
- nickmuchi/sec-bert-finetuned-finance-classification
- FacebookAI/roberta-base

## 🌳Project Structure

```plaintext
.
├── data                          <- Folder with the datasets developed by the team
│   ├── processed                 <- Ready-to-use-datasets
│   |    ├── finetuning_3_labels..<- Dataset with 3 labels used for training
│   |    ├── finetuning_5_labels..<- Dataset with 5 labels used for training
│   |    ├── news_prices-new-2    <- Dataset with labels and prices used for evaluation and label creation
│   |    └── rest                 <- legacy version of the datasets
│   └── raw                       <- Raw data (not recommended to use)
├── notebooks                     <- Jupyter notebooks for analysis
│   ├── OLD-3-LABELS              <- Legacy file with discontinued functionality, not recommended to use
│   ├── fine_tuning_3_labels      <- File for full fine-tuning on 3 labels
│   ├── fine_tuning_3_labels      <- File for full fine-tuning on 5 labels (not used in research due to poor performance)
│   └── lora_tuning               <- PEFT tuning on 5 labels (not used in research due to low full tuning times)
├── papers                        <- Main papers we build upon + our own
│   └── THIS-RESEARCH-PAPER       <- the paper we wrote while working on the project
├── src/data_pipeline             <- Code used to process the datasets
├── README.md                     <- Project documentation
└── requirements.txt              <- Dependency list
```

## 📚 Papers we recommend to read for the curious project viewer

- FINBERT (good paper, easy to read and gives good overview): https://arxiv.org/abs/1908.10063
- Classification of different fin data but very solid methodology: https://pdfs.semanticscholar.org/95e7/29cb065b9e274c609478376b8ad93982b123.pdf
- IJCAI 2024 FinLLM Challenge: https://huggingface.co/spaces/TheFinAI/IJCAI-2024-FinLLM-Learderboard 
- PIXIU financial tailtored LLMs, instruction tuning datasets, and evaluation benchmarks: https://github.com/The-FinAI/PIXIU
- FinBen is a benchmark for financial LLMs. https://arxiv.org/abs/2402.12659 
- Id look at FINBERT cites for papers https://www.semanticscholar.org/paper/FinBERT-%3A-A-Large-Language-Model-for-Extracting-Huang-Wang/8798b3a01c29fe0ce45a271bedd934787343dfb5?sort=relevance 
- There is an interesting idea of taking the last token representation of a Decoder LLM and training a classifier on top of it so you can take advantage of the big and good llms we have. https://arxiv.org/abs/2311.01239



## 💡 Interesting stuff
- Different use cases of LLMs in finance: https://github.com/hananedupouy/LLMs-in-Finance
- NLP Papers applcibale to finance: https://github.com/maximedb/nlp_papers
- Read our future research section :)
