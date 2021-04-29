import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load data files and merge them
    Args:
        messages_filepath: path to messages csv
        categories_filepath: path to categories csv
    Returns:
        df: dataframe with merged datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on="id")

    return df


def clean_data(df):
    """ Transform the category lists into dummy variables,
    drop duplicated rows and columns with zeros and make all
    columns binary
    Args:
        df: dataframe with data of messages and categories
    Returns:
        df: cleaned dataframe
    """
    # split categories into separate columns, use helper df categories
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract list of new column names with row: use word, slice last two chars
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values into binary values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')

    # drop original categories col
    df.drop("categories", axis=1, inplace=True)

    # join original df and categories
    df = df.join(categories)

    # drop duplicates
    df = df.drop_duplicates()

    # while the other columns look like binary categories, "related"
    # includes 2.0 as value --> replace with 1
    df["related"] = df["related"].replace(to_replace=2, value=1)

    # drop all columns that only contain 0
    df = df.loc[:, (df != 0).any(axis=0)]

    return df


def save_data(df, database_filename):
    """ Save data
    Args:
        df: dataframe to save
        database_filename: filename for SQLite database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_data', engine, index=False, if_exists='replace')


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
