# -*- coding: utf-8 -*-
import re
import string
from typing import Dict, List

import ssg

ARABIC_RX = re.compile(r"[A-Za-z]+")
CAMEL_CASE_RX = re.compile(r"([a-z])([A-Z])([a-z])")
EMAIL_RX = re.compile(r"^\w+\@\w+\.\w+$")
NUMBER_RX = re.compile(r"[0-9,]+")
TRAILING_SPACE_RX = re.compile(r"\n$")
URL_RX = re.compile(r"(https?:\/\/)?(\w+\.)?\w+\.\w+")


def syllable2token(syllable: str) -> str:
    if ARABIC_RX.match(syllable):
        return "<ENGLISH>"
    elif NUMBER_RX.match(syllable):
        return "<NUMBER>"
    else:
        return syllable


def syllable2ix(sy2ix: Dict[str, int], syllable: str) -> int:
    token = syllable2token(syllable)

    return sy2ix.get(token, sy2ix.get("<UNK>"))


def character2ix(ch2ix: Dict[str, int], character: str) -> int:
    if character == "":
        return ch2ix.get("<PAD>")
    elif character in string.punctuation:
        return ch2ix.get("<PUNC>")

    return ch2ix.get(character, ch2ix.get("<UNK>"))

def find_words_from_preds(tokens, preds) -> List[str]:
    # Construct words from prediction labels {0, 1}
    curr_word = tokens[0]
    words = []
    for s, p in zip(tokens[1:], preds[1:]):
        if p == 0:
            curr_word = curr_word + s
        else:
            words.append(curr_word)
            curr_word = s

    words.append(curr_word)

    return words


def syllable_tokenize(txt: str) -> List[str]:
    # Proxy function for syllable tokenization, in case we want to try
    # a different syllable tokenizer.
    seps = txt.split(" ")

    new_tokens = []

    for i, s in enumerate(seps):
        tokens = ssg.syllable_tokenize(s)
        new_tokens.extend(tokens)

        if i < len(seps) - 1:
            new_tokens.append(" ")

    return new_tokens
