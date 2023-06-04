## Business_Data_Analytics_Transaction_Clustering_Virtual_Training

### table of contents
   * Overview
      * Notebook
      * Requirements
      * Trained models
      * Test script
      * Dataset results
   * How to run and test locally 
   * Project Flow



## Overview
   This project was about to categorize customer narrations at the time of transactions and analysis the behaviour of customers based on the categorization. It was done using unsupervised learning algorithms and NLP.

### Notebook
   The whole work is located at "Training_Task.ipynb"

### Requirements
   The requirements to run this project locally and test accordingly at "requirements.txt"

### Trained Models
   Trained model for this project.

   * KMeans Clustering Model

### Test Script
   The test script to test the models "test_script.py"

### Dataset Results
   The results on the given labeled small dataset added in the "dataset_results"


## How to run and test locally

 * clone or download the zip of the repo and unzip it
 * inside the project folder, open terminal and run -

       pip install -r requirements.txt

 * now we will be able to run the test script to predict new data. To run the test script, we need to provide a <b>filePath (.csv)</b> indicating the test data, <b>column name(narrations or something like that)</b> for prediction  and <b>the model name</b> for trained model.
 So, the command will be -

       python3 test_script.py "filepath" "columnName" "modelName"


   * for <b>kmeans</b> model on the "test.csv" dataset for the "narrations" column -

         python3 test_script.py test.csv narrations kmeans

## Project Flow

   Some basic preprocessing techniques of NLP like tokenization, stop word removal, punctuation removal, digit removal, stemming, and lemmatization.

NLP Pipeline:

    * Text Preprocessing:
    * Text Cleaning:
    * Lowercasing
    * Special character removal
    * Punctuation removal etc. 

    * Tokenization on cleaned text
    * Normalization on Tokens
        * Stemming
        * Lemmatization (Please try to find out their difference like which is better in which case)
    * Stop words removal (Unimportant words in our text like a, an, the, preposition etc.)
    * Keyword Extraction
        * TF-IDF
        * Gensim
        * Spacy, NLTK, YAKE, RAKE, KeyBert
    * Featured Engineering/ Feature Extraction (Converting text data to numerical):

