import torch
from torch.utils.data import Dataset
import json
from attacut import preprocessing



class SyllableCharacterSeqDataset():

    def make_feature(self, txt, ch_dict, sy_dict):
        print(f"txt: {txt}")
        syllables = preprocessing.syllable_tokenize(txt)
        print(f"syllable : {syllables}")
        sy2ix, ch2ix = sy_dict, ch_dict

        ch_ix, syllable_ix = [], []

        for syllable in syllables:
            six = preprocessing.syllable2ix(sy2ix, syllable)
            print(f"six : {six}")
            chs = list(
                map(
                    lambda ch: preprocessing.character2ix(ch2ix, ch),
                    list(syllable)
                )
            )
            ch_ix.extend(chs)
            syllable_ix.extend([six]*len(chs))

        # Convert Python lists to PyTorch tensors
        ch_ix_tensor = torch.tensor(ch_ix, dtype=torch.int64)
        syllable_ix_tensor = torch.tensor(syllable_ix, dtype=torch.int64)
        seq_lengths_tensor = torch.tensor([len(syllables)], dtype=torch.int64)

        # Stack tensors along a new dimension to create features tensor
        features = torch.stack((ch_ix_tensor, syllable_ix_tensor), dim=0)

        # Reshape features tensor
        features = features.unsqueeze(0)

        return list(txt), (features, seq_lengths_tensor)
