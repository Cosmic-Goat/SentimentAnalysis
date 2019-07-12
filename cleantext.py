import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin


class CleanText(BaseEstimator, TransformerMixin):
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        blacklist = stopwords.words('english')
        blacklist.extend([])
        words = input_text.split()
        clean_words = [word for word in words if (word not in blacklist or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)

        # Make sure there are no empty entries left
        empty_clean = clean_X == ''
        print('{} records have no words left after text cleaning'.format(clean_X[empty_clean].count()))
        clean_X.loc[empty_clean] = '[no_text]'

        print(clean_X.sample(5))

        return clean_X
