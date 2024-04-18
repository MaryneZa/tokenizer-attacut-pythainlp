import torch
from typing import Dict, List
from attacut import dataloaders, preprocessing
import torch.nn as nn
import torch.nn.functional as F
from attacut.models import ConvolutionBatchNorm, STATIC_MODEL_CONFIG




class Attacut(nn.Module):
    def __init__(self, model_config: Dict = STATIC_MODEL_CONFIG,model: str = "attacut_sc"):
        super(Attacut, self).__init__()
        self.dataset = dataloaders.SyllableCharacterSeqDataset()
        # data_config: Dict = self.dataset.setup_featurizer("attacut/models/attacut_sc")
        # no_chars = data_config['num_char_tokens']
        # no_syllables = data_config['num_tokens']
        self.characters_dict = model_config["characters"]
        self.syllables_dict = model_config["syllables"]
        no_chars = len(self.characters_dict)
        no_syllables = len(self.syllables_dict)
        # print(f"data_config : {data_config}")
        
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
        self.model_params = model_config
        
        
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

    # def tokenizer(self, txt: str, sep="|", device="cpu", pred_threshold=0.5) -> List[str]:
    #     if txt == "":  # handle empty input string
    #         return [""]
    #     if not txt or not isinstance(txt, str):  # handle None
    #         return []

    #     tokens, features = self.dataset.make_feature(txt)

    #     inputs = (
    #         features,
    #         torch.Tensor(0)  # dummy label when won't need it here
    #     )

    #     x, _, _ = self.dataset.prepare_model_inputs(inputs, device=device)
    #     probs = torch.sigmoid(self.model(x))  # Call the class directly to instantiate an object
    #     preds = probs > pred_threshold

    #     # Convert predictions to CPU tensor
    #     preds_cpu = preds.cpu()

    #     # Convert boolean tensor to list of words
    #     words = preprocessing.find_words_from_preds(tokens, preds_cpu)
    #     return words
    def tokenizer(self, txt: str, sep="|", device="cpu", pred_threshold=0.5) -> List[str]:
        if txt == "":  # handle empty input string
            return [""]
        if not txt or not isinstance(txt, str):  # handle None
            return []

        # Convert input text to feature tensors
        tokens, features = self.dataset.make_feature(txt, self.characters_dict, self.syllables_dict)
        print(f"tokens : {tokens}, features : {features}")
        inputs = (
            features,
            torch.Tensor(0)  # dummy label when won't need it here
        )
        # Prepare model inputs
        x, seq_lengths = inputs[0]
        x = x.to(device)
        print(f"x : {x}")

        # Pass inputs through the model
        probs = torch.sigmoid(self.model((x, seq_lengths)))  # Call the model directly to instantiate an object

        # Convert probabilities to predictions
        print(f"probs : {probs}")
        preds = probs > pred_threshold
        
        print(f"pred : {preds}")

        # Convert predictions to CPU tensor
        preds_cpu = preds.cpu()

        # Convert boolean tensor to list of words
        words = preprocessing.find_words_from_preds(tokens, preds_cpu)
        return words
