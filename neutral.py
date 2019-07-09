import numpy as np
import pandas as pd
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

import warnings
warnings.filterwarnings('ignore')

np.random.seed(37)

df = pd.read_csv('datagt/glassdoor_v3.csv', nrows=10000)

positive_pros = df.loc[df.Rating_overall.isin([4, 5])].Pros.to_frame(name='text')
positive_pros['sentiment'] = 1
neutral_pros = df.loc[df.Rating_overall.isin([3])].Pros.to_frame(name='text')
neutral_pros['sentiment'] = 0
neutral_cons = df.loc[df.Rating_overall.isin([3])].Cons.to_frame(name='text')
neutral_cons['sentiment'] = 0
negative_cons = df.loc[df.Rating_overall.isin([1, 2])].Cons.to_frame(name='text')
negative_cons['sentiment'] = -1

data = pd.concat([positive_pros, neutral_pros, neutral_cons, negative_cons])
data.reset_index(drop=True, inplace=True)
data = data.reindex(np.random.permutation(data.index))

# Clean up text data
ct = CleanText()
sr_clean = ct.fit_transform(data.text)
print(sr_clean.sample(5))

# Make sure there are no empty entries left
empty_clean = sr_clean == ''
print('{} records have no words left after text cleaning'.format(sr_clean[empty_clean].count()))
sr_clean.loc[empty_clean] = '[no_text]'

cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(22, 20))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show()