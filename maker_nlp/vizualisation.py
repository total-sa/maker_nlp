import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from maker_nlp.config import *


def get_top_n_words(corpus, k: int =None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [[word, sum_words[0, idx]] for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_freq = pd.DataFrame(words_freq, columns=['words', 'importance'])
    return words_freq[:k]


def plot_top_k_words_per_sentiment(corpus: pd.Series, labels: pd.Series, k: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    for axe_index, sentiment in enumerate(CLASS_NAMES):
        sentiment_related_phrases = corpus[labels == CLASS_DICT[sentiment]]

        top_k_related_sentiment = get_top_n_words(sentiment_related_phrases, k=k)

        # Ploting figures
        top_k_words = top_k_related_sentiment['words'].values
        top_k_words_position = top_k_related_sentiment.index.to_list()
        top_k_importance = top_k_related_sentiment['importance'].values

        axes[axe_index].barh(top_k_words_position,
                             top_k_importance,
                             align='center')
        axes[axe_index].set_yticks(top_k_words_position)
        axes[axe_index].set_yticklabels(top_k_words)
        axes[axe_index].invert_yaxis()
        axes[axe_index].set_xlabel('Importance')
        axes[axe_index].set_title(f'Most important words related to sentiment : {sentiment.upper()}')


def plot_top_k_words_in_corpus(corpus: [str], k: int = 1):
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))

    top_k_related_sentiment = get_top_n_words(corpus, k=k)
    # Ploting figures
    top_k_words = top_k_related_sentiment['words'].values
    top_k_words_position = top_k_related_sentiment.index.to_list()
    top_k_importance = top_k_related_sentiment['importance'].values
    axes.barh(top_k_words_position,
                         top_k_importance,
                         align='center')
    axes.set_yticks(top_k_words_position)
    axes.set_yticklabels(top_k_words)
    axes.invert_yaxis()
    axes.set_xlabel('Importance')
    axes.set_title(f'Most important words')


def plot_top_k_words_per_sentiment_tfidf(corpus: pd.Series, labels: pd.Series, k: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    vectorizer = TfidfVectorizer().fit(corpus)

    for axe_index, sentiment in enumerate(CLASS_NAMES):
        sentiment_related_phrases = corpus[labels == CLASS_DICT[sentiment]]

        top_k_related_sentiment = get_top_k_words_tfidf(sentiment_related_phrases, vectorizer, k=k)

        # Ploting figures
        top_k_words = top_k_related_sentiment['words'].values
        top_k_words_position = top_k_related_sentiment.index.to_list()
        top_k_importance = top_k_related_sentiment['importance'].values

        axes[axe_index].barh(top_k_words_position,
                             top_k_importance,
                             align='center')
        axes[axe_index].set_yticks(top_k_words_position)
        axes[axe_index].set_yticklabels(top_k_words)
        axes[axe_index].invert_yaxis()
        axes[axe_index].set_xlabel('Importance')
        axes[axe_index].set_title(f'Most important words related to sentiment : {sentiment.upper()}')


def get_top_k_words_tfidf(dataframe: pd.DataFrame, vectorizer, k: int):
    inverse_dict = {val: key for key, val in vectorizer.vocabulary_.items()}
    data = vectorizer.transform(dataframe)
    max_tfidf_list = np.zeros(data.shape[1])

    for i in range(data.shape[0]):
        row = data.getrow(i).toarray()[0].ravel()
        max_tfidf_list[[x for x in row.argsort()[-5:] if row[x] > 0.2]] += 1

    top_10_indices = np.argsort(max_tfidf_list)[-k:][::-1]
    top_10_values = max_tfidf_list[top_10_indices]

    top_10_words = [inverse_dict[j] for j in top_10_indices]
    return pd.DataFrame({'words': top_10_words, 'importance': top_10_values})


## These functions plot top k words with the help of idf vector
def plot_top_k_explicative_words_per_sentiment(corpus: pd.DataFrame, labels: pd.Series, k: int = 20):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    vectorizer = TfidfVectorizer().fit(corpus)

    for axe_index, sentiment in enumerate(CLASS_NAMES):
        sentiment_related_phrases = corpus[labels == CLASS_DICT[sentiment]]

        #top_k_related_sentiment = get_top_k_words_tfidf(sentiment_related_phrases, vectorizer, k=k)
        top_k_related_sentiment = _get_top_k_words(sentiment_related_phrases, vectorizer, k)

        # Ploting figures
        top_k_words = top_k_related_sentiment['words'].values
        top_k_words_position = top_k_related_sentiment.index.to_list()
        top_k_importance = top_k_related_sentiment['importance'].values

        axes[axe_index].barh(top_k_words_position,
                             top_k_importance,
                             align='center')
        axes[axe_index].set_yticks(top_k_words_position)
        axes[axe_index].set_yticklabels(top_k_words)
        axes[axe_index].invert_yaxis()
        axes[axe_index].set_xlabel('Importance')
        axes[axe_index].set_title(f'Most important words related to sentiment : {sentiment.upper()}')


def _get_sentiment_idf_dataframe(corpus: pd.DataFrame, vectorizer):
    vect_neg = TfidfVectorizer()
    vect_neg.fit(corpus)
    neg_inverse_dict = {val: key for key, val in vect_neg.vocabulary_.items()}

    inverse_dict = {val: key for key, val in vectorizer.vocabulary_.items()}
    idf_df = pd.DataFrame([[inverse_dict[i], x] for i, x in enumerate(vectorizer.idf_)], columns=['word', 'idf_score'])
    idf_df = idf_df.sort_values(by='idf_score', ascending=False).reset_index().drop(columns=['index'])

    return idf_df, neg_inverse_dict


def _get_top_k_words(corpus: pd.DataFrame, vectorizer, k: int):
    count_neg = CountVectorizer()
    counts = count_neg.fit_transform(corpus)
    counts = counts.sum(axis=0).reshape((-1, 1))

    top_neg_words = []
    top_importance = []

    idf_df, neg_inverse_dict = _get_sentiment_idf_dataframe(corpus, vectorizer)
    counts_df = pd.DataFrame([[neg_inverse_dict[i], x[0, 0]] for i, x in enumerate(counts)], columns=['word', 'count'])
    counts_df = counts_df[counts_df['count'] > 4]

    for index, (word, score) in idf_df.iterrows():
        if word in counts_df.word.values:
            top_neg_words.append(word)
            top_importance.append(score)
        if len(top_neg_words) == k:
            break
    return pd.DataFrame({'words': top_neg_words, 'importance': top_importance})

