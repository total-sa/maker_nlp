import unidecode
import string
import re

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

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
    stop_words_list = list(fr_stop) + list(en_stop)
    useful_words = ' '.join([token for token in text.split(' ') if not token in stop_words_list])
    return useful_words


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = remove_stop_words(text)
    return text
