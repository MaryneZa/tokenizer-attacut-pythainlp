import torch
from typing import Dict, List
from attacut import dataloaders, preprocessing
import torch.nn as nn
from attacut.models import ConvolutionBatchNorm
import torch.nn.functional as F




class Attacut(nn.Module):
    dataset: dataloaders.SequenceDataset = dataloaders.SyllableCharacterSeqDataset()
    def __init__(self, model: str = "attacut-sc"):
        super(Attacut, self).__init__()
        self.dataset = Attacut.dataset
        data_config: Dict = self.dataset.setup_featurizer("attacut/artifacts/attacut-sc")
        print(f"data_config : {data_config}")
        no_chars = data_config['num_char_tokens']
        no_syllables = data_config['num_tokens']
        model_config = "embc:16|embs:8|conv:16|l1:16|do:0.0"
        conv_filters = 64

        self.ch_embeddings = nn.Embedding(no_chars, 32, padding_idx=0)
        self.sy_embeddings = nn.Embedding(no_syllables, 16, padding_idx=0)

        emb_dim = 48  # Sum of embedding dimensions (32 + 16)

        self.dropout = torch.nn.Dropout(p=0.0)  # No dropout in this configuration

        self.conv1 = ConvolutionBatchNorm(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionBatchNorm(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionBatchNorm(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, 32)
        self.linear2 = nn.Linear(32, 1)
        
    @classmethod
    def from_checkpoint(cls, model_path):
        model = cls()  # Create an instance of the Attacut class
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        cls.model = model
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

    def tokenizer(self, txt: str, sep="|", device="cpu", pred_threshold=0.5) -> List[str]:
        if txt == "":  # handle empty input string
            return [""]
        if not txt or not isinstance(txt, str):  # handle None
            return []

        tokens, features = Attacut.dataset.make_feature(txt)

        inputs = (
            features,
            torch.Tensor(0)  # dummy label when won't need it here
        )

        x, _, _ = Attacut.dataset.prepare_model_inputs(inputs, device=device)
        probs = torch.sigmoid(Attacut.model(x))  # Call the class directly to instantiate an object

        preds = probs.cpu().detach().numpy() > pred_threshold

        words = preprocessing.find_words_from_preds(tokens, preds)
        return words
