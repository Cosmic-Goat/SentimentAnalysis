import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16, 9)})

ss = ShuffleSplit(n_splits=10, test_size=0.2)
vect = CountVectorizer(max_features=1000, binary=True)
sm = SMOTE()
nb = MultinomialNB()

accs = []
f1s = []
cms = []


def predict_sentiment(text):
    pred = nb.predict(vect.transform([text]))
    return pred[0]


def print_accuracy():
    print("\nAverage accuracy across folds: {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("\nAverage F1 score across folds: {:.2f}%".format(sum(f1s) / len(f1s) * 100))
    print("\nAverage Confusion Matrix across folds: \n {}".format(sum(cms) / len(cms)))


df = pd.read_csv('datagt/glassdoor_v3.csv', nrows=10000)

positive_pros = df.loc[df.Rating_overall.isin([4, 5])].Pros.to_frame(name='text')
positive_pros['sentiment'] = 1
neutral_pros = df.loc[df.Rating_overall.isin([3])].Pros.to_frame(name='text')
neutral_pros['sentiment'] = 1
neutral_cons = df.loc[df.Rating_overall.isin([3])].Cons.to_frame(name='text')
neutral_cons['sentiment'] = -1
negative_cons = df.loc[df.Rating_overall.isin([1, 2])].Cons.to_frame(name='text')
negative_cons['sentiment'] = -1

data = pd.concat([positive_pros, neutral_pros, neutral_cons, negative_cons])
data.reset_index(drop=True, inplace=True)

current_fold = 1

for train_index, test_index in ss.split(data.text):
    print(f"On fold: {current_fold}")
    current_fold += 1
    text_train, text_test = data.text.iloc[train_index], data.text.iloc[test_index]
    sentiment_train, sentiment_test = data.sentiment.iloc[train_index], data.sentiment.iloc[test_index]

    # Fit vectorizer and transform text train, then transform text test
    text_train_vect = vect.fit_transform(text_train)
    text_test_vect = vect.transform(text_test)

    # Oversample
    text_train_res, sentiment_train_res = sm.fit_sample(text_train_vect, sentiment_train)

    # Fit Naive Bayes on the vectorized text with sentiment train labels,
    # then predict new sentiment labels using text test
    nb.fit(text_train_res, sentiment_train_res)
    sentiment_pred = nb.predict(text_test_vect)

    # Determine test set accuracy and f1 score on this fold using the true sentiment labels and predicted sentiment labels
    accs.append(accuracy_score(sentiment_test, sentiment_pred))
    f1s.append(f1_score(sentiment_test, sentiment_pred))
    cms.append(confusion_matrix(sentiment_test, sentiment_pred))

print_accuracy()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,9))

acc_scores = [round(a * 100, 1) for a in accs]
f1_scores = [round(f * 100, 2) for f in f1s]

x1 = np.arange(len(acc_scores))
x2 = np.arange(len(f1_scores))

ax1.bar(x1, acc_scores)
ax2.bar(x2, f1_scores, color='#559ebf')

# Place values on top of bars
for i, v in enumerate(list(zip(acc_scores, f1_scores))):
    ax1.text(i - 0.25, v[0] + 2, str(v[0]) + '%')
    ax2.text(i - 0.25, v[1] + 2, str(v[1]))

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Naive Bayes')
ax1.set_ylim([0, 100])

ax2.set_ylabel('F1 Score')
ax2.set_xlabel('Runs')
ax2.set_ylim([0, 100])

sns.despine(bottom=True, left=True)  # Remove the ticks on axes for cleaner presentation

plt.show()