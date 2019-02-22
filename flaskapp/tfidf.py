import pandas as pd
import numpy as np
import nltk
import string
import os
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    text = str(text).lower().translate(None, string.punctuation)  # remove punctuations
    tokens = nltk.word_tokenize(text)
    # filtered = [w for w in tokens if (w not in stopwords.words('english')) and (w.isalnum()) and (not w.isdigit())]
    filtered = [w for w in tokens if (w not in stopwords.words('english')) and (not w.replace('.', '').isdigit())]
    # |return filtered
    stems = []
    for item in filtered:
        stems.append(PorterStemmer().stem(item))
    return stems


def tokenize_skill(text):
    tokens = re.split('\s{2,}|\t', text)  # split more than 2 whitespaces
    # tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if (w not in stopwords.words('english')) and (not w.replace('.', '').isdigit())]
    filtered = [w.strip() for w in filtered]
    return filtered


# return tfidf, tfs given a list of docs
def get_tfidf(text, tokenize, max_features=1000):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', lowercase=True, max_features=max_features)
    tfs = tfidf.fit_transform(text)
    return tfidf, tfs


def get_top_k_matches(project, tfidf, tfs, K=10):
    """
    project - string describing the project
    tfidf  - precomputed tfidf on users
    K - the number of best matches
    """
    project_vec = tfidf.transform([project])
    scores = cosine_similarity(project_vec, tfs)
    scores = scores.flatten()
    top_index = (np.argsort(-scores))[:K]
    # return [(i, scores[i]) for i in top_index]
    return top_index, scores


def get_top_k_nonoverlapping_matches(project, tfidf, tfs, K=10, non_negative=True):
    """
    project - string describing the project
    tfidf  - precomputed tfidf on users
    K - the number of best matches
    """

    project_vec = tfidf.transform([project])
    out = []
    for k in range(K):
        if k > 0:
            project_vec -= tfs[top_index, :]
            if non_negative:  # reset negative components to zero
                project_vec[project_vec < 0] = 0

        scores = cosine_similarity(project_vec, tfs)
        scores = scores.flatten()
        for j, s in out:  # excluding already used users
            scores[j] = 0

        top_index = scores.argmax()
        if scores[top_index] < 0.01: continue
        out.append((top_index, scores[top_index]))

    return out

