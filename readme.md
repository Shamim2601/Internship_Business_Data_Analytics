## Business_Data_Analytics_Transaction_Clustering_Virtual_Training

## Overview
   This project was about to categorize customer narrations at the time of transactions and analysis the behaviour of customers based on the categorization. 


## How to run

 * clone or download the zip of the repo and unzip it
 * inside the project folder, open terminal and run -

      python3 main.py "fileName" "columnName"

For example:

      python3 main.py test.csv narration

This will output Categories of narrations in "test.csv" file.

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
    * Categorization
    * Most important word finding using TF-IDF
