import collections
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix

from .cleantext import CleanText
from .gridvect import grid_vect
from .confusion_matrix_pretty_print import pretty_plot_confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)


# sns.set(style="darkgrid")
# sns.set(font_scale=1.3)


def word_frequency(clean_text):
	cv = CountVectorizer()
	bow = cv.fit_transform(clean_text)
	word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
	word_counter = collections.Counter(word_freq)
	word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
	fig, ax = plt.subplots(figsize=(22, 20))
	sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
	plt.show()


def show_features(estimator, class_names=None):
	# TODO: Use transformer names or 'final estimator' to make this more decoupled.
	if class_names is None:
		class_names = ['Negative', 'Neutral', 'Positive']
	best_mk_ct = estimator.best_estimator_.steps[0][1].mk_ct
	best_clf = estimator.best_estimator_.steps[2][1]

	feature_names = [re.sub('^.{0,16}', '', name) for name in best_mk_ct.get_feature_names()]
	feature_count = best_clf.feature_count_
	features_df = pd.DataFrame(list(zip(feature_names, *feature_count)), columns=['Name', *class_names])
	#features_df = features_df.sort_values('Positive')
	#sns.heatmap(features_df.drop('Name', axis=1)[-10:], cbar=False, yticklabels=features_df.Name[-10:])
	#plt.show()

	for clazz in class_names:
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
		class_features = features_df.sort_values(clazz)[::-1]
		plt.figure(figsize=(12, 5))
		sns.barplot(x=class_features.Name[:10], y=class_features[clazz][:10])
		plt.show()


np.random.seed(37)
text_cols = ['Review_starttext', 'Pros', 'Cons']
y_axis = 'Rating_overall'

df = pd.read_csv('data/indeed_v2.csv')  # , nrows=30000
df_data = df[[*text_cols, y_axis]]  # Remove unused data
df_data = df_data.dropna()  # Remove rows with empty text
df_data = df_data.reindex(np.random.permutation(df_data.index))
df_data[y_axis] = df_data[y_axis].replace([1, 2, 3, 4, 5], [-1, -1, 0, 1, 1])  # Replace ratings with classifier classes
df_data = df_data[:100000]  # Limit to first N entries, only for quick testing.

# Preprocessing, this shouldn't be done in the grid search
preprocess = make_column_transformer((CleanText(), text_cols), remainder='passthrough')
df_processed = preprocess.fit_transform(df_data)

# Column transformer converts the data to an ndarray, so this converts it back to a dataframe
df_processed = pd.DataFrame(data=df_processed,  # values
							index=df_data.index,  # 1st column as index
							columns=df_data.columns)  # 1st row as the column names

x_train, x_test, y_train, y_test = train_test_split(df_processed.drop(y_axis, axis=1), df_processed[y_axis],
													test_size=0.2, random_state=37)

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

vect = TfidfVectorizer()

# MultinomialNB
best_mnb_countvect = grid_vect(mnb, parameters_mnb, x_train, x_test, y_train, y_test, text_cols,
							   parameters_text=parameters_vect, vect=vect)
# joblib.dump(best_mnb_countvect, './output/best_mnb_tfidfvect.pkl')
# LogisticRegression
# best_logreg_countvect = grid_vect(logreg, parameters_logreg, x_train, x_test, y_train, y_test, text_cols, parameters_text=parameters_vect, vect=vect)
# joblib.dump(best_logreg_countvect, './output/best_logreg_countvect.pkl')

con_matrix = confusion_matrix(np.asarray(y_test, dtype='int64'), best_mnb_countvect.predict(x_test))
pretty_plot_confusion_matrix(pd.DataFrame(con_matrix, columns=['negative', 'neutral', 'positive']))

show_features(best_mnb_countvect)
