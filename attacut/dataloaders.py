import torch
from torch.utils.data import Dataset
import json

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

def syllable2ix(sy2ix: Dict[str, int], syllable: str) -> int:
    if ARABIC_RX.match(syllable):
        token = "<ENGLISH>"
    elif NUMBER_RX.match(syllable):
        token = "<NUMBER>"
    else:
        token = syllable

    return sy2ix.get(token, sy2ix.get("<UNK>"))

def character2ix(ch2ix: Dict[str, int], character: str) -> int:
    if character == "":
        return ch2ix.get("<PAD>")
    elif character in string.punctuation:
        return ch2ix.get("<PUNC>")

    return ch2ix.get(character, ch2ix.get("<UNK>"))


class SyllableCharacterSeqDataset():

    def make_feature(self, txt, ch_dict, sy_dict):
        seps = txt.split(" ")

        new_tokens = []

        for i, s in enumerate(seps):
            tokens = ssg.syllable_tokenize(s)
            new_tokens.extend(tokens)

            if i < len(seps) - 1:
                new_tokens.append(" ")

        sy2ix, ch2ix = sy_dict, ch_dict

        ch_ix, syllable_ix = [], []

        for syllable in new_tokens:
            six = syllable2ix(sy2ix, syllable)
            chs = list(
                map(
                    lambda ch: character2ix(ch2ix, ch),
                    list(syllable)
                )
            )
            ch_ix.extend(chs)
            syllable_ix.extend([six]*len(chs))

        # Convert Python lists to PyTorch tensors
        ch_ix_tensor = torch.tensor(ch_ix, dtype=torch.int64)
        syllable_ix_tensor = torch.tensor(syllable_ix, dtype=torch.int64)
        seq_lengths_tensor = torch.tensor([len(new_tokens)], dtype=torch.int64)

        # Stack tensors along a new dimension to create features tensor
        features = torch.stack((ch_ix_tensor, syllable_ix_tensor), dim=0)

        # Reshape features tensor
        features = features.unsqueeze(0)

        return list(txt), (features, seq_lengths_tensor)
