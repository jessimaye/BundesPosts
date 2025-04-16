# Project Title

## Descriptive Statistics

This folder contains metadata on members of the German Parliament such as Instagram follower count, year of birth, party etc. (Alle_Abgeordneten.xlsx), as well as the jupyter-notebook, that was used to analyze this data (analysis.ipynb)

## Sentiment Analysis 

### Comparison of sentiment models

We tested three sentiment models on the publicly available “German politicians twitter Sentiment”-dataset (https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment). For this evaluation we used the test split with 357 examples and acquired the following models via the Hugging Face platform: Multilingual Sentiment Classification Model, German Sentiment Bert, XLM RoBERTa German Sentiment. We employed the different models and their tokenizer to classify the tweets of the test split and then compared the results to the true labels. The code for this process as well as the overall accuracy-values and the classification reports for all three models can be found in "Sentiment_models_comparison_twitter.ipynb".

### Finetuning GBERT-base

The jupyter notebook "Finetuning_gbert1_twitter_dataset" contains the pipeline for fine-tuning the German language model GBERT-base by deepset to sentiment classification (three classes) with the “German politicians twitter Sentiment”-dataset (https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment). This fine-tuned model is referred to as GBERT1. 

The jupyter notebook "Finetuning_gbert2_twitter&germeval17" contains the pipeline for fine-tuning GBERT-base to sentiment classification with the “German politicians twitter Sentiment”-dataset and the publicly available “Germeval Task 2017”-dataset (https://huggingface.co/datasets/akash418/germeval_2017). This fine-tuned model is referred to as GBERT2. 

To ensure compatibility with our model, we mapped the sentiment labels to numerical values. Additionally, unnecessary columns were removed to retain only the tweet text and corresponding labels. For the training we used the train-split of the datasets and the test-split for evaluation. Each post was tokenized with truncation and padding to a maximum sequence length of 512 for the twitter-dataset and 128 for the combined dataset. Afterwards the tokenized datasets were converted into PyTorch tensors so that the model can process them. Each model was trained for 4 epochs and with a learning rate of 2e-5. The batch size of GBERT1 was 8 and of GBERT2 16. Moreover, the trainer was configured with an evaluation metric function that computes accuracy, precision, recall, and F1-score. 

### Evaluation of fine-tuned models using gold standard

To ensure that the fine-tuned models are robust and not overfitted to the training data, we used an subset of 2,000 posts of our collected data, which were choosen randomly through the "Evaluationsdatensatz_Erstellung"-Notebook, that we have annotated ourselves ("Evaluationsdatensatz-3.xlsx), for evaluation and to determine which model is most suitable for our task. The jupyter notebook "Classification_of_evaluation_texts" contains the pipeline to assign a sentiment to each post of this subset using GBERT1 and GBERT2, the results can be seen in  "Evaluationstexte_mit_Modellergebnissen.csv". We then compared the model results with our gold standard. This evaluation process can be found in "Evaluation_with_Goldstandard.ipynb", which also contains a confusion matrix and a classification report for GBERT1 and GBERT2 respectively. Morover, this notebook contains the calculation of Krippendorffs Alpha for the Inter Annotator Agreement. 

### Running the model

The jupyter notebook "Model_Run" contains the run of the model with which the data was annotated. During the scraping some data had to be retrieved at a later point, which is also processed within this notebook.

### Analysing and visualizing the data
The two jupyter notebooks called "Sentiment_pro_Jahr" and "Sentiment_pro_Partei" contain the code to analyse and visualise the data. 
"Sentiment_pro_Jahr" focuses on absolute post numbers and their visualisation per year, per month and in long graphs spanning several years, regardless of parties.
"Sentiment_pro_Partei" on the other hand does almost the same, except with focus on each individual party. In addition to absolute numbers, this notebook concentrates also on a mean-sentiment value for eacht party. This mean is calculated by subtracting the number of negative posts from the number of positive posts and dividing the result through the total number of posts. The resulting graphics can be found in Analyse/Sentiment_Graphics.zip of this repository.
These visualisations have been the foundation of the analysis.


# Topic Modeling

For the Topic Modeling, we used BERTopic, specifically, we implemented a configuration optimized for large-scale data processing. Following standard preprocessing procedures, including the removal of stopwords,
the documents (i.e., the social media posts) are transformed into dense vector representations using Sentence-BERT embeddings. We applied the pretrained embedding model paraphrase-multilingual-MiniLM-L12-v2, which
is well-suited for multilingual content. The resulting embeddings were cached temporarily and subsequently passed to a multilingual BERTopic model for unsupervised clustering and topic extraction. The initial number of generated topics was reduced to 150 using the model’s builtin topic reduction functionality. Furthermore, we applied the reduce_outliers method to reassign these documents to the most semantically appropriate topics.
To assess whether the topic model generated meaningful and contextually appropriate themes, we analyzed the ten most frequent topics separately for negative, positive, and neutral posts, and compared them with corresponding word clouds for each sentiment category. The training of the model can be found in the jupyter notebook "bert_topic_training".

The file "Model_Info_in_df.csv" contains the information about the model regarding which topics there are, associated words and example documents.

In the jupyter notebook "Topic_Analysis" the mapping of the topic names to the ids can be found as well as the analysis for whicht the dataframe was filtered by party, year and month in different versions.

# Analyse

This folder contains the results of the analysis. There are currently one folder called "Parteien_Top_Ten" and a zip-file called "Sentiment_Graphics" in this folder. The zip-file contains every graphic that can be produced with the code from Sentiment-Analysis/Sentiment_pro_Jahr.ipynb and Sentiment-Analysis/Sentiment_pro_Partei.ipynb, which are over 100 indiviual graphics.

### Parteien_Top_Ten

This folder contains 40 files with five files for each party. 

The five files contain:
- The ten most frequent topics for all positive posts of the party
- The ten most frequent topics for all neutral posts of the party
- The ten most frequent topics for all negative posts of the party
- The ten most frequent topics in general of the party
- An overview combining the prior four files









