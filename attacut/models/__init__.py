import torch
from typing import Dict, List
import torch.nn as nn
import torch.nn.functional as F
from .layer import ConvolutionBatchNorm
from attacut.statics.config import STATIC_MODEL_CONFIG
from attacut.statics.pattern import *
import ssg
import string
import json

class Attacut(nn.Module):
    def __init__(self, model_config: Dict = STATIC_MODEL_CONFIG,model: str = "attacut_sc"):
        super(Attacut, self).__init__()
        
        self.characters_dict = self.load_dict("attacut/statics/characters.json")
        self.syllables_dict = self.load_dict("attacut/statics/syllables.json")
        
        no_chars = len(self.characters_dict)
        no_syllables = len(self.syllables_dict)
        
        conv_filters = model_config['conv']
        self.dropout = torch.nn.Dropout(p=model_config['do']) 
        
        self.ch_embeddings = nn.Embedding(no_chars, model_config['embc'], padding_idx=0)
        self.sy_embeddings = nn.Embedding(no_syllables, model_config['embs'], padding_idx=0)
        
        emb_dim = model_config['embc'] + model_config['embs']

        self.conv1 = ConvolutionBatchNorm(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionBatchNorm(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionBatchNorm(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, model_config['l1'])
        self.linear2 = nn.Linear(model_config['l1'], 1)
        
        
    @classmethod
    def from_checkpoint(cls, model_path, cuda=False, eval=True):
        model = cls()  # Create an instance of the Attacut class
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        device = torch.device("cuda" if cuda else "cpu")
        model.device = device
        model.to(device)
        
        if eval:
            model.eval()
        
        return model

    def forward(self, inputs):
        x, seq_lengths = inputs

        x_char, x_syllable = x[:, 0, :], x[:, 1, :]

        ch_embedding = self.ch_embeddings(x_char)
        sy_embedding = self.sy_embeddings(x_syllable)

        embedding = torch.cat((ch_embedding, sy_embedding), dim=2)

        embedding = embedding.permute(0, 2, 1)

        conv1 = self.dropout(self.conv1(embedding).permute(0, 2, 1))
        conv2 = self.dropout(self.conv2(embedding).permute(0, 2, 1))
        conv3 = self.dropout(self.conv3(embedding).permute(0, 2, 1))

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        out = out.view(-1)

        return out
    
    def load_dict(self, data_path: str) -> Dict:
            with open(data_path, "r", encoding="utf-8") as f:
                dict = json.load(f)
            return dict

    def inferrence(self, txt: str, sep="|", pred_threshold=0.5) -> List[str]:
        if txt == "":  # handle empty input string
            return [""]
        if not txt or not isinstance(txt, str):  # handle None
            return []

        # Convert input text to feature tensors
        tokens, features = self.prepare_tensor(txt, self.characters_dict, self.syllables_dict)
        inputs = (
            features,
            torch.Tensor(0) 
        )
        # Prepare model inputs
        x, seq_lengths = inputs[0]
        x = x.to(self.device)

        # Pass inputs through the model
        probs = torch.sigmoid(self((x, seq_lengths)))  # Call the model directly to instantiate an object
        # Convert probabilities to predictions
        preds = probs > pred_threshold
        

        # Convert predictions to CPU tensor
        preds_cpu = preds.cpu()
        # Construct words from prediction labels {0, 1}
        curr_word = tokens[0]
        words = []
        for s, p in zip(tokens[1:], preds_cpu[1:]):
            if p == 0:
                curr_word = curr_word + s
            else:
                words.append(curr_word)
                curr_word = s

        words.append(curr_word)

        return words
    
    def syllable2ix(self, sy2ix: Dict[str, int], syllable: str) -> int:
        if ARABIC_RX.match(syllable):
            token = "<ENGLISH>"
        elif NUMBER_RX.match(syllable):
            token = "<NUMBER>"
        else:
            token = syllable

        return sy2ix.get(token, sy2ix.get("<UNK>"))
    
    def character2ix(self, ch2ix: Dict[str, int], character: str) -> int:
        if character == "":
            return ch2ix.get("<PAD>")
        elif character in string.punctuation:
            return ch2ix.get("<PUNC>")

        return ch2ix.get(character, ch2ix.get("<UNK>"))



    def prepare_tensor(self, txt, ch_dict, sy_dict):
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
            six = self.syllable2ix(sy2ix, syllable)
            chs = list(
                map(
                    lambda ch: self.character2ix(ch2ix, ch),
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

