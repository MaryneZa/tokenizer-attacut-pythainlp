import torch
from .version import __version__
from typing import Dict, List
from attacut import artifacts, dataloaders, logger, models, preprocessing, utils

log = logger.get_logger(__name__)

class Attacut:
    def __init__(self, model: str = "attacut-sc"):
        
        # resolve model's path
        model_path = artifacts.get_path(model)
        params = utils.load_training_params(model_path)
        model_name = params.name
        log.info("loading model %s" % model_name)
        model_cls: models.BaseModel = models.get_model(model_name)

        # instantiate dataset
        dataset: dataloaders.SequenceDataset = model_cls.dataset()

        # load necessary dicts into memory
        data_config: Dict = dataset.setup_featurizer(model_path)

        # instantiate model
        self.model = model_cls.load(
            model_path,
            data_config,
            params.params
        )
        

        self.dataset = dataset

    def from_checkpoint(self, txt: str, sep="|", device="cpu", pred_threshold=0.5) -> List[str]:
        if txt == "":  # handle empty input string
            return [""]
        if not txt or not isinstance(txt, str):  # handle None
            return []

        tokens, features = self.dataset.make_feature(txt)

        inputs = (
            features,
            torch.Tensor(0)  # dummy label when won't need it here
        )

        x, _, _ = self.dataset.prepare_model_inputs(inputs, device=device)
        probs = torch.sigmoid(self.model(x))

        preds = probs.cpu().detach().numpy() > pred_threshold

        words = preprocessing.find_words_from_preds(tokens, preds)

        return words
