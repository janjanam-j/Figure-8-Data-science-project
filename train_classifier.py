import sys
import pandas as pd
import numpy as np
import pickle
import nltk
import re


from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])


def load_data(database_filepath):
    
    """
        Loads the sql databse and reads in the cleaned data
        Input: SQL database name and the table name
    
        Output: Outputs the X, Y, and column names for futher analysis
        
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_data', engine)
    
    X = df['message']    
    Y = df.iloc[:, 4:]
    
    category_names = list(df.columns[4:])
    print(category_names)
    return X, Y, category_names


def tokenize(text):
    
    """
        Messages are toknized by replacing the unwanted characters, removing too common word, 
        stop words, punctiations etc
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    
    """
        Model pipeline is created using RandomForestClassifer and grid search is used for 
        optimizing the parameters.
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
   

    parameters = {
                    'tfidf__use_idf': (True, False),
                    'clf__estimator__n_estimators': [10,20],
                    'clf__estimator__min_samples_split': [2, 3]
                }
                         
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters, n_jobs = -1)
    
    return cv
                         
                         
def evaluate_model(model, X_test, Y_test, category_names):
    
    """
        Model is evaluated for overall accuracy as well as by category. 
    
    """
    
    Y_pred = model.predict(X_test)
    print('Overall accuracy is {0:.2f}%'.format(((Y_pred == Y_test).mean().mean())*100))
    
    for i, category_names in enumerate(Y_test):
        print(category_names)
        print(classification_report(Y_test.iloc[:,i].values, Y_pred[:, i]))  
    

def save_model(model, model_filepath):
    
    """
        Pickle file is saved
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))                    


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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()