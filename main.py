import numpy as np
import pandas as pd
import nltk
import re
import sys
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('words')

from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')

words = set(nltk.corpus.words.words())
from nltk.corpus import stopwords
st = set(stopwords.words("english"))

lemmatizer=WordNetLemmatizer()


def preprocess_data(df, fieldName):
    narrations = df[fieldName]
    # print(narrations)
    preprocessed_data = []
    
    for entry in narrations:
        entry_string = str(entry)
        entry_string = entry_string.lower()
        entry_string = re.sub('[^A-Za-z ]+', '', entry_string)
        entry_string = ' '.join(word.lower() for word in entry_string.split() if word not in st)
        entry_string = nltk.word_tokenize(entry_string)
        entry_string = [lemmatizer.lemmatize(word, 'v') for word in entry_string]
        preprocessed_data.append(entry_string)            

    return preprocessed_data


def categorize(preprocessed_data):
    categorized_data = []
    for narration in preprocessed_data:
        for word in narration:
            if 'withdraw' in word:
                category = 'withdraw'
                break
            if 'deposit' in word:
                category = 'deposit'
                break
            if 'payment' in word:
                category = 'payment'
                break
            if 'transfer' in word:
                category = 'transfer'
                break
            if 'commission' in word:
                category = 'commission'
                break
            if 'collection' in word:
                category = 'collection'
                break
            if 'card' in word:
                category = 'card'
                break
            if 'charge' in word:
                category = 'charge'
                break
            category = 'uncategorized'
        categorized_data.append(category)
    
    return categorized_data


def most_important_word(preprocessed_data):
    vectorizer = TfidfVectorizer()

    # Convert the list of word lists into strings
    sentence_strings = [' '.join(sentence) for sentence in preprocessed_data]

    # Compute the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentence_strings)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    miw = []

    # Iterate over each sentence and find the most important word
    for i, sentence in enumerate(preprocessed_data):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        max_tfidf_index = tfidf_scores.argmax()
        most_important_word = feature_names[max_tfidf_index]
        miw.append(most_important_word)
        # print(f"Most important word in sentence {i + 1}: {most_important_word}")
    
    return miw


def processData(filePath, fieldName):
    df = pd.read_csv(filePath)
    # print(df[fieldName])

    preprocessed_data = preprocess_data(df, fieldName)

    most_imp_words = most_important_word(preprocessed_data)

    categorized_data = categorize(preprocessed_data)

    print('\n', '#Tx', '\t', 'Category', '\t', 'most_imp_word', '\t', 'Processed Narration')
    for idx in range(len(preprocessed_data)):
        print(df['index'][idx], '\t', categorized_data[idx], '\t',  most_imp_words[idx], '\t',preprocessed_data[idx])


if __name__ == "__main__":
    # input file path, field name(narrations)
    filePath = sys.argv[1]
    fieldName = sys.argv[2]

    processData(filePath, fieldName)