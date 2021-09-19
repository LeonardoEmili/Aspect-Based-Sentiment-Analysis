import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

from pytorch_lightning import Trainer
from stud.dataset import ABSADataset
from stud.utils import HParams, download_nltk_resources, load_hparams, load_hparams_dict, load_tokenizer, Vectors
from stud.torch_utils import gpus
from stud.models.polarity_classifier import PolarityClassifier
from stud.models.absa_classifier import ABSAClassifier
from stud.models.category_classifier import CategoryClassifier
from stud.models.multistep_classifier import MultistepClassifier

# Download required resources, from cache if available
download_nltk_resources()
cached_vectors: Vectors = Vectors.from_cached()

def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    hparams = load_hparams('model/polarity_hparams.pth', device=device, test_only=True)
    model = PolarityClassifier.load_from_checkpoint('model/polarity_classifier.ckpt', hparams=hparams)
    return StudentModel(model)

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.
    """
    hparams = load_hparams_dict('model/multistep_hparams.pth', device=device, test_only=True)
    model = MultistepClassifier.load_from_checkpoint('model/multistep_classifier_ab.ckpt', mode='ab', **hparams)
    return StudentModel(model)

def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    hparams = load_hparams_dict('model/multistep_hparams.pth', device=device, test_only=True)
    model = MultistepClassifier.load_from_checkpoint('model/multistep_classifier_cd.ckpt', mode='cd', **hparams)
    return StudentModel(model)

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    ''' Implementation of the Model class to make predictions over input sentences. '''

    def __init__(self, model):
        self.model = model
        self.extended_bio = model.hparams.extended_bio
        self.device = model.hparams.device
        self.tokenizer = load_tokenizer(model.hparams)

    def predict(self, samples: List[Dict]) -> List[Dict]:
        ds = ABSADataset(
            samples,
            tokenizer=self.tokenizer,
            extended_bio=self.extended_bio,
            cached_vectors=cached_vectors,
            batch_size=32
        )
        trainer = Trainer(gpus=gpus(device=self.device))
        trainer.test(self.model, ds)
        return self.model.predictions
