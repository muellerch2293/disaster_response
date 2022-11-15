import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np
import pandas as pd
nltk.download(['punkt', 'wordnet','stopwords','omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
# List stop words from NLTK
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
from nltk.corpus import stopwords

def load_data(database_filepath):
    engine = create_engine('sqlite:///data/disaster_response.db')
    df = pd.read_sql_table('disaster_messages',engine)
    X = df.message
    Y = df[df.columns.difference(["id","message","original","genre"])]
    return X,Y,Y.columns


def tokenize(text):
    """
    Returns a normalized, tokenized and lemmed (process of mapping  words back to their root) version of the given text
    Parameters: 
    text : str
        text that should be tokenized
    Return:
    list of str: list containing all lemmatized tokens for the given text
    
    """
    #Clean
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #Normalize text: case normalization, punctuation removal
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    #tokenize
    words = word_tokenize(text)
    #remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]   
    return lemmed_tokens


def build_model():
    """
    Creates a RandomForestClassifier with the required preprocessing steps in a pipeline 
    Return:
   GridSearchCv: a GridSearchCV over a pipeline consisting of a CountVectorizer,TfidfTransformer and a RandomForestClassifier with multiple output    classes
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [10, 25, 50, 100, 200,500]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,  verbose=10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):   
    """
    Evaluates the prediction performance of the model on every category
    Parameters: 
    model : 
        model that should be evaluated
    X_test: 
        test input that should be used for evaluation
    Y_test:
        the correct classification of the test input
    category_names:
        the name of all possible classification categories 
    """
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    


def save_model(model, model_filepath):
    """
    Persists the model in a pickle file
    Parameters: 
    model : 
        model that should be persistest
    model_filepath: 
        location where the pickle file should be written to
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
 
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_params_)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        y_pred = model.predict(X_test)
        np.savetxt("models/prediction.csv", y_pred, delimiter=",")
        np.savetxt("models/true_categories.csv",Y_test,delimiter=",")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()