# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from attacut import preprocessing, utils

class SequenceDataset(Dataset):
    def __init__(self, path: str = None):
        if path:
            self.load_preprocessed_data(path)

    @staticmethod
    def prepare_model_inputs(inputs, device="cpu"):

        x, seq_lengths = inputs[0]
        x = x.to(device)
        y = inputs[1].float().to(device).reshape(-1)

        return (x, seq_lengths), y, y.shape[0]
    



class SyllableCharacterSeqDataset(SequenceDataset):
    def setup_featurizer(self, path: str):

        with open(f"{path}/characters.json", "r", encoding="utf-8") as f:
            self.ch_dict = json.load(f)

        with open(f"{path}/syllables.json", "r", encoding="utf-8") as f:
            self.sy_dict = json.load(f)

        return dict(
            num_char_tokens=len(self.ch_dict),
            num_tokens=len(self.sy_dict)
        )

    def make_feature(self, txt):
        syllables = preprocessing.syllable_tokenize(txt)

        sy2ix, ch2ix = self.sy_dict, self.ch_dict

        ch_ix, syllable_ix = [], []

        for syllable in syllables:
            six = preprocessing.syllable2ix(sy2ix, syllable)

            chs = list(
                map(
                    lambda ch: preprocessing.character2ix(ch2ix, ch),
                    list(syllable)
                )
            )

            ch_ix.extend(chs)
            syllable_ix.extend([six]*len(chs))

        features = np.stack((ch_ix, syllable_ix), axis=0) \
            .reshape((1, 2, -1)) \
            .astype(np.int64)

        seq_lengths = np.array([features.shape[-1]], dtype=np.int64)

        return list(txt), (torch.from_numpy(features), torch.from_numpy(seq_lengths))

    
   