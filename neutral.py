import numpy as np
import pandas as pd

from glassdoorconverter import convert
from gridvect import grid_vect

pd.set_option('display.max_colwidth', -1)

from cleantext import CleanText

from time import time
import re
import string
import os
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


import warnings
warnings.filterwarnings('ignore')


def word_frequency(clean_text):
    cv = CountVectorizer()
    bow = cv.fit_transform(clean_text)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
    fig, ax = plt.subplots(figsize=(22, 20))
    sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    plt.show()


np.random.seed(37)

df = pd.read_csv('datagt/glassdoor_v3.csv', nrows=1000)
df = df.reindex(np.random.permutation(df.index))
df_model = convert(df)

x_train, x_test, y_train, y_test = train_test_split(df_model.drop('rating', axis=1), df_model.rating, test_size=0.2, random_state=37)

# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'mv__max_df': (0.25, 0.5, 0.75),
    'mv__ngram_range': ((1, 1), (1, 2)),
    'mv__min_df': (0.01, 0.02)
}

# Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}

# Parameter grid settings for LogisticRegression
parameters_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
}

mnb = MultinomialNB()
logreg = LogisticRegression()

tfidfvect = TfidfVectorizer(max_df=0.75, min_df=1, ngram_range=(1,1))
# MultinomialNB
best_mnb_countvect = grid_vect(mnb, parameters_mnb, x_train, x_test, y_train, y_test, parameters_text=parameters_vect, vect=tfidfvect)
joblib.dump(best_mnb_countvect, './output/best_mnb_tfidfvect.pkl')
# LogisticRegression
best_logreg_countvect = grid_vect(logreg, parameters_logreg, x_train, x_test, y_train, y_test, parameters_text=parameters_vect, vect=countvect)
joblib.dump(best_logreg_countvect, './output/best_logreg_countvect.pkl')