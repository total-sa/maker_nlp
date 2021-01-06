from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

CLASS_NAMES = ['negative', 'neutral', 'positive']
CLASS_DICT = {'negative': 0, 'somewhat negative': 1, 'neutral': 2, 'somewhat positive': 3, 'positive': 4}

STOP_WORDS_LIST = list(fr_stop) + list(en_stop) + ['lrb', 'rrb', 'makes', 'make', 'zzzzzzzzz', 'zwick', 'zucker', 'zone']
