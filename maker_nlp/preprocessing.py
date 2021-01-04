import unidecode
import string
import re

def convert_to_lowercase(text: str) -> str:
    return text.lower()

def remove_accents(text: str) -> str:
    return unidecode.unidecode(text)

def remove_ponctuation_and_digits(text: str) -> str:
    exclist = string.punctuation + string.digits
    table_ = str.maketrans(exclist, ' ' * len(exclist))
    text = text.translate(table_)
    return text

def remove_spaces_and_new_lines(text: str) -> str:
    return re.sub('\s+',' ', text).strip()
    

def clean_text(text: str) -> str:
    text = convert_to_lowercase(text)
    text = remove_accents(text)
    text = remove_ponctuation_and_digits(text)
    text = remove_spaces_and_new_lines(text)
    return text
