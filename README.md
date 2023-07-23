# Sentiment Analysis of Covid-19 Tweets
## Description
The Covid-19 pandemic has led to a surge in social media engagement, particularly on Twitter, where users express their thoughts and opinions. Analyzing the sentiment of these tweets can provide valuable insights into public perception, which can assist policymakers and stakeholders in their decision-making process. This project investigates the effectiveness of various feature representation methods for sentiment analysis of Covid-19 tweets.

The project compares the performance of a Support Vector Classifier (Linear SVC) using different feature representation methods with a fine-tuned RoBERTa model incorporating contextual embeddings. The analysis is conducted on a dataset of 44,926 manually labeled tweets. The sentiment labels are divided into three categories: Positive, Neutral, and Negative.

## Methodology
The project is divided into several stages, each corresponding to a Jupyter notebook:

**Data Cleaning (lemmatized).ipynb**: This notebook contains the preprocessing and cleaning of the tweet data. The sentiment categories are encoded into three classes: Positive, Neutral, and Negative. The tweets are cleaned, lemmatized, and stopwords are removed. Irrelevant columns and rows in the data are removed.

**Feature Representation and Model Training**: These are the notebooks with titles like 'BoW LinearSVC (RQ1).ipynb', 'Tf-Idf LinearSVC (RQ1).ipynb', 'GloVe LinearSVC (RQ2).ipynb', 'Word2Vec LinearSVC (RQ2).ipynb', and 'FastText LinearSVC (RQ2).ipynb'. Each notebook trains a Linear SVC model using a different feature representation method (Bag of Words, Tf-Idf, GloVe, Word2Vec, FastText). Hyperparameters are tuned using grid search with cross-validation.

**Fine-tuned RoBERTa Model**: The 'RoBERTa_fine_tuned_(RQ3) GC.ipynb' notebook trains a fine-tuned RoBERTa model incorporating contextual embeddings.

**Exploratory Data Analysis**: The 'EDA (RQ4).ipynb' notebook contains exploratory data analysis of the tweet data.

## Results
The RoBERTa model emerges as the superior performer, achieving an F1 score of 0.86 and a PR AUC score of 0.91, which underscores the power of contextual embeddings in capturing the nuances of social media language. Interestingly, the simple count-based Linear SVC model performs better than the models using static word embeddings. These results highlight the importance of selecting appropriate feature representation techniques for sentiment analysis tasks, especially when analyzing pandemic-related social media data.

## Data
The project uses a dataset of 44,926 manually labeled tweets ([COVID-19 NLP Text Classification Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)).

## Requirements
This project requires Python and the following Python libraries installed:

- numpy
- pandas
- nltk
- spacy
- sklearn
- json
- matplotlib
- seaborn
- torch (for RoBERTa)
- transformers (for RoBERTa)

## Execution
To run the project, execute each Jupyter notebook in the order mentioned above. Make sure to update the paths to the data files as per your directory structure.
