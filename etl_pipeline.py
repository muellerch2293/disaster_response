# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine



import sys


def load_data(messages_filepath, categories_filepath):
    """
    Correlates and loads raw data into a pandas dataframe 
    messages_filepath : str
        path pointing to the file containing the messages
    categories_filepath : str
        path pointing to the file containing the categorization of the messages
    Return:
    pandas dataframe containing the messages with their respective categories   
    """
    # load messages dataset
    messages = pd.read_csv('data/disaster_messages.csv')

    # load categories dataset
    categories = pd.read_csv('data/disaster_categories.csv')

    # merge datasets
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    """
    Cleans categories of messages st. every category has its own column that could be either 0 or 1
    df : pandas dataframe
        dataframe containing messages with their categories (in a single column, seperated by semicolons)
    Return:
    pandas dataframe with messages and their cleaned up classification  
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda col: col[:len(col)-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string and convert to numeric
        categories[column] =  categories[column].astype(str).apply(lambda col: int(col[len(col)-1:len(col)]))
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join = 'inner')
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    """
    Saves dataframe to a database with the given name
    df : pandas dataframe
        dataframe that should be persisted in the database
    database_filename: str
        location of the database file in which the dataframe should be stored
    """
    engine = create_engine(('sqlite:///%s' % database_filename))
    #drop table if exists 
    sql = 'DROP TABLE IF EXISTS disaster_messages;'
    result = engine.execute(sql)
    df.to_sql('disaster_messages', engine, index=False)
    print("Cleaned data and saved into disaster_messages table!")


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
