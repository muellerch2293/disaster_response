# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def main():
    # load messages dataset
    messages = pd.read_csv('data/disaster_messages.csv')

    # load categories dataset
    categories = pd.read_csv('data/disaster_categories.csv')

    # merge datasets
    df = messages.merge(categories,on='id')
    
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
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join = 'inner')
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    engine = create_engine('sqlite:///data/disaster_response.db')
    #drop table if exists 
    sql = 'DROP TABLE IF EXISTS disaster_messages;'
    result = engine.execute(sql)
    df.to_sql('disaster_messages', engine, index=False)
    print("Cleaned data and saved into disaster_messages table!")
    
if __name__ == '__main__':
    main()