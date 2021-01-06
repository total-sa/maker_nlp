import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from maker_nlp.config import *


def get_top_n_words(corpus, k=None):
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
