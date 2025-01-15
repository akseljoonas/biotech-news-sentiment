# Press Release Sentiment Analysis
In the U.S., 62% of adults own various investments like stocks, treasury bonds, or commodities. Information about public companies—such as earnings reports, product launches, and financial metrics—is widely accessible online.

Although these updates are often objective, they frequently include analysts' opinions, allowing for clear sentiment analysis. This project leverages Large Language Models to perform automated sentiment analysis, aiming to anticipate market reactions faster than competitors such as investment banks or private investors. Predicting market responses to news seeks to financially benefit the users through strategic buying or shorting of stocks.

## 🏃‍♂️ Running Source Code
### 🛠️ Set-Up

**Clone the Repository**: 
Start by cloning the repository to your local machine.
   ```bash
   git clone https://github.com/akseljoonas/news-sentiment.git
   cd news-sentiment
   ```
> [!TIP]
> Before downloading the requirements as seen in the next step,
> we recommend creating a virtual environment and setting it up
> as there are a lot of dependencies in this project.

  
   Make sure all dependencies are installed by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

### 🏋️‍♂️ Training the Models

To train the models, run the desired notebook from the notebooks folder. See the next section for more details.

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
├── README.md                     <- Repository documentation
└── requirements.txt              <- Dependency list
```

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


## 📈 Areas for improvement of the project


## ⚠️ Disclosure of copyright
> [!IMPORTANT]
> This research project has been conducted during the Language Technology Practical course at the University of Groningen but has been developed
> independently by the research team and hence should be referenced using the link of this GitHub repo. The team only allows for **non-commercial** use
> of the models and methodologies assuming standard citation practices. Furthermore, the users of our code take full responsibility for model outputs.



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
