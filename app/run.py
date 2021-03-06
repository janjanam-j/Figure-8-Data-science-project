import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.externals import joblib
import pickle


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

"""
Loading the SQL database
Loading the celanded data

"""
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)


# load model
"""
Loading pickel file saved in the ML Pipeline

"""
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    """
    Loading data required for the visualisations

    """
    categories = df.iloc[:,4:]
    
    #Top 10 categories
    categories_distribution = categories.sum().sort_values(ascending=False)[0:10]
    categories_names = list(categories_distribution.index)
    
    #Distribution of top categeory identified in plot 1 category
    related =categories.related
    related_distribution = related.value_counts()/len(related)*100
    related_names = list(related_distribution.index)
    
    
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_distribution
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Message Catgories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Catergories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_distribution
                )
            ],

            'layout': {
                'title': 'Distribution of Message Relavence (Related)',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Message Relavence"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
