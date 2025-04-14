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
