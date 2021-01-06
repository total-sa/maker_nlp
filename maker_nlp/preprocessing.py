import unidecode
import string
import re
import numpy as np

import en_core_web_sm

from maker_nlp.config import STOP_WORDS_LIST

global NLP
NLP = en_core_web_sm.load()


def convert_to_lowercase(text: str) -> str:
    return text.lower()


def remove_accents(text: str) -> str:
    return unidecode.unidecode(text)


def remove_punctuation_and_digits(text: str) -> str:
    exclist = string.punctuation + string.digits
    table_ = str.maketrans(exclist, ' ' * len(exclist))
    text = text.translate(table_)
    return text


def remove_spaces_and_new_lines(text: str) -> str:
    return re.sub('\s+', ' ', text).strip()


def normalize_text(text: str) -> str:
    text = convert_to_lowercase(text)
    text = remove_accents(text)
    text = remove_punctuation_and_digits(text)
    text = remove_spaces_and_new_lines(text)
    return text


def remove_stop_words(text: str) -> str:
    useful_words_list = ' '.join([token for token in text.split(' ') if not token in STOP_WORDS_LIST])
    return useful_words_list


def lemmatize(text: str) -> str:
    text = ' '.join([token.lemma_ for token in NLP(text)])
    return text


def remove_stop_words_and_lemmatize(text):
    new_text = ' '.join(token.lemma_ for token in text if not token.text in STOP_WORDS_LIST)
    return new_text


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = remove_stop_words_and_lemmatize(NLP(text))
    if len(text.split(' ')) < 2:
        text = np.nan
    return text
