import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    """
    Input files are: 
        mesages_filepath- File with messages will be loaded
        categories_filepath: File with categoreis will be loaded
    
    Process: 
        Both meassages and catergories files will be merged using the 'id' variables
    
    Output: 
        Megred file will be save in the SQL  database.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on ='id')
    #print(df.head())
    return df

def clean_data(df):
    
    """
    categories are cleaned by seperating stirngs and numreic values.
    1. Numbers associated/attached with the strings (numbers attached to the strings: related-1, related-0) are seperated and converted to numeric values
    2. Strings/categorie names are updated as column headers.
    3. Cofindence should usually be zero or one however, related category and 2 as well. I replaced the values of 2 with 1.
    4. Messages and catergories concatenated together
    5. Duplicate values are dropped and final clean dataset isprovided for further analysis 
    
    """
        
    categories  = df.categories.str.split(pat =';', expand =True)
    row = categories.iloc[0]
    category_colnames =  row.apply(lambda x:x[:-2]).tolist()
    categories.columns = category_colnames
    

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    categories['related'].replace({2:1}, inplace=True)
       
    df.drop('categories', axis=1, inplace=True)
    
    
    df = pd.concat([df, categories], axis =1)
    
    df.drop_duplicates(inplace =True)
    print(df.head())
    return df



def save_data(df, database_filename):
    
    """
    A SQL database is created and the cleaned data is saved for ML Pipeline
    
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('clean_data',engine, index=False, if_exists ='replace')
    
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()