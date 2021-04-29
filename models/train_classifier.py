import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """ Load message data with categories
    Args:
        database_filepath: path of database
    Returns:
        X: dataframe with features
        Y: dataframe with labels
        category_names: list with category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_data', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """ Tokenize, lemmatize and normalize text
    Args:
        text: text to tokenize
    Returns:
        clean_tokens: cleaned tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ Build model pipeline and parameters
    for Grid Search
    Returns:
        cv: Grid Search Optimizer
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'vect__max_df': [0.75, 1.0],
        'clf__estimator__n_estimators': [25, 50, 100],
        'clf__estimator__learning_rate': [0.5, 1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Calculate predictions from test data
    and print classification report
    Args:
        model: trained model
        X_test: test data
        Y_test: test labels
        category_names: names of categories
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred,
                                target_names=category_names))
    # accuracy


def save_model(model, model_filepath):
    """ Save model
    Args:
        model: trained model
        model_filepath: path where to save the model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

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
