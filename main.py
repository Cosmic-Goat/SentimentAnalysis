import math
import random
from collections import defaultdict
from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE

sm = SMOTE()
nb = MultinomialNB()

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16, 9)})

df = pd.read_csv('datagt/glassdoor_v3.csv', nrows=50000)

positive_pros = df.loc[df.Rating_overall.isin([4, 5])].Pros.to_frame(name='text')
positive_pros['sentiment'] = 1
neutral_pros = df.loc[df.Rating_overall.isin([3])].Pros.to_frame(name='text')
neutral_pros['sentiment'] = 1
neutral_cons = df.loc[df.Rating_overall.isin([3])].Cons.to_frame(name='text')
neutral_cons['sentiment'] = -1
negative_cons = df.loc[df.Rating_overall.isin([1, 2])].Cons.to_frame(name='text')
negative_cons['sentiment'] = -1

data = pd.concat([positive_pros, neutral_pros, neutral_cons, negative_cons])

text_train, text_test, sentiment_train, sentiment_test = train_test_split(data.text, data.sentiment, test_size=0.2)

vect = CountVectorizer(max_features=1000, binary=True)
text_train_vect = vect.fit_transform(text_train)

text_train_res, sentiment_train_res = sm.fit_sample(text_train_vect, sentiment_train)

nb.fit(text_train_res, sentiment_train_res)
nb.score(text_train_res, sentiment_train_res)
text_test_vect = vect.transform(text_test)
sentiment_pred = nb.predict(text_test_vect)

print("Accuracy: {:.2f}%".format(accuracy_score(sentiment_test, sentiment_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(sentiment_test, sentiment_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(sentiment_test, sentiment_pred))

