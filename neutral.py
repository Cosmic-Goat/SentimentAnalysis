import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer

from cleantext import CleanText
from gridvect import grid_vect

sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)


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
text_cols = ['Review_starttext', 'Cons', 'Pros']
y_axis = 'Rating_overall'

df = pd.read_csv('datagt/indeed_v2.csv', nrows=100)
df = df.reindex(np.random.permutation(df.index))
df_data = df[[*text_cols, y_axis]]

# Preprocessing, this shouldn't be done in the grid search
preprocess = make_column_transformer((CleanText(), text_cols), remainder='passthrough')
df_processed = preprocess.fit_transform(df_data)
df[y_axis] = df[y_axis].astype()

# Column transformer converts the data to an ndarray, so this converts it back to a dataframe
df_processed = pd.DataFrame(data=df_processed,  # values
                       index=df_data.index,  # 1st column as index
                       columns=df_data.columns)  # 1st row as the column names

x_train, x_test, y_train, y_test = train_test_split(df_processed.drop(y_axis, axis=1), df_processed[y_axis], test_size=0.2, random_state=37)

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
best_mnb_countvect = grid_vect(mnb, parameters_mnb, x_train, x_test, y_train, y_test, text_cols, parameters_text=parameters_vect, vect=tfidfvect)
#joblib.dump(best_mnb_countvect, './output/best_mnb_tfidfvect.pkl')
# LogisticRegression
best_logreg_countvect = grid_vect(logreg, parameters_logreg, x_train, x_test, y_train, y_test, text_cols, parameters_text=parameters_vect, vect=countvect)
#joblib.dump(best_logreg_countvect, './output/best_logreg_countvect.pkl')