from typing import *
from dataclasses import asdict
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from stud.models.ner_classifier import NERClassifier
from stud.models.polarity_classifier import PolarityClassifier
from stud.constants import LOGGER_TRAIN_LOSS, LOGGER_VALID_LOSS, LOGGER_TEST_LOSS
from stud.torch_utils import batch_scatter_mean
from torchtext.vocab import Vocab
from stud import utils
import pytorch_lightning as pl
import torch
import torch.nn as nn

def model_from(hparams: utils.HParams, polarity_hparams: Optional[utils.HParams] = None) -> nn.Module:
    ''' Returns the correct model from the input hparams. '''
    if polarity_hparams:
        ner_model = NERClassifier.load_from_checkpoint(hparams.ner_model_path, hparams=hparams)
        polarity_model = PolarityClassifier.load_from_checkpoint(polarity_hparams.polarity_model_path, hparams=polarity_hparams)
        return AspectMultistepClassifier(ner_model, polarity_model)
    if hparams.model_name == 'bert_lstm':
        bert = BertModel.from_pretrained(utils.get_bert_path(hparams))
        # Using BERT as a frozen encoder
        bert.eval()
        return BERTLSTMClassification(hparams, bert)
    if hparams.model_name == 'multistep_classifier':
        raise Exception(f'Missing implementation of AspectMultistepClassifier!')    
    raise Exception(f'{hparams.model_name} not supported!')

class BERTLSTMClassification(nn.Module):
    def __init__(
        self,
        hparams: utils.HParams,
        bert: BertModel
    ):
        super().__init__()
        self.hparams = hparams
        self.bert = bert
        self.lstm = nn.LSTM(self.bert_output_dim, hparams.hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hparams.hidden_dim, hparams.hidden_dim)
        self.fc2 = nn.Linear(hparams.hidden_dim, len(hparams.output_vocab))
        self.dropout = nn.Dropout(hparams.dropout)

    @property
    def bert_output_dim(self) -> int:
        ''' Returns BERT output dimension from the chosen pooling strategy. '''
        return (
            self.hparams.input_dim * len(self.hparams.layers_to_merge)
            if self.hparams.strategy == 'cat'
            else self.hparams.input_dim
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            batch_out = self.bert(**x['batch'], output_hidden_states=True)
        batch_out = utils.merge_layers(
            batch_out.hidden_states, strategy=self.hparams.strategy,
            layers_to_merge=self.hparams.layers_to_merge
        )
        # Merge WordPiece embeddings into token embeddings
        batch_out = batch_scatter_mean(batch_out, x['mask'])

        batch_out, _ = self.lstm(self.dropout(batch_out))
        batch_out = torch.relu(self.dropout(self.fc1(batch_out)))
        batch_out = self.fc2(batch_out)
        return batch_out

class ABSAClassifier(pl.LightningModule):
    '''
    NER classifier identifies aspect terms and polarities for task A+B.

    :param hparams: hyperparameters and target vocab to set up the model
    '''
    def __init__(self, hparams: utils.HParams):
        super().__init__()
        self.save_hyperparameters(asdict(hparams))
        self.output_dim = len(hparams.output_vocab)
        self.model = model_from(hparams)
        self.evaluate_callback = None
        self.ignore_index = hparams.output_vocab.get_default_index()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.aspect_predictions = []
        self.aspect_indexes = []
        self.gold = []

    @property
    def bio_itos(self) -> Tuple[List[int], List[int], int]:
        ''' Returns the extended BIOs indexes. '''
        return (
            utils.vocab_tokens_startswith(self.hparams.output_vocab, 'B'),
            utils.vocab_tokens_startswith(self.hparams.output_vocab, 'I'),
            self.hparams.output_vocab['O'])

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['ner'].view(-1).long())
        metrics = {LOGGER_TRAIN_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def evaluation(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self.model(x)
        loss = 0
        if not self.hparams.test_only:
            loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['ner'].view(-1).long())
        return loss, y_hat

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ):
        loss, logits = self.evaluation(batch)
        metrics = {LOGGER_VALID_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
 
    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int = 0
    ) -> Tuple[List[List[str]], List[List[str]]]:
        x, y = batch
        loss, logits = self.evaluation(batch)
        y_hat = logits.argmax(-1)

        aspect_indexes, polarity_terms = zip(*[utils.extract_aspect_indices(
            prediction_idxs, length, *self.bio_itos, return_tensors=True
            ) for prediction_idxs, length in zip(y_hat.detach().cpu(), x['lengths'])])

        aspect_terms = self.batch_aspects_lookup(x['tokens'], aspect_indexes)
        polarity_terms = self.batch_aggregate_polarities(polarity_terms)

        self.aspect_indexes += aspect_indexes
        self.aspect_predictions += polarity_terms

        if not self.hparams.test_only:
            self.gold += [list(zip(*labels)) for labels in zip(y['aspect'], y['polarity_labels'])]
            assert len(self.aspect_indexes) == len(self.gold)

            metrics = {LOGGER_TEST_LOSS: loss}
            self.log_dict(metrics)
        
        return aspect_terms, polarity_terms

    def batch_aggregate_polarities(
        self,
        batch_idxs: List[int]
    ) -> List[List[str]]:
        return [[utils.aggregate_polarities(idxs, self.hparams.output_vocab) for idxs in sent_idxs] for sent_idxs in batch_idxs]

    def aspects_lookup(
        self,
        tokens: List[str],
        idxs: List[List[int]],
        sep: str = ' '
    ) -> List[List[str]]:
        ''' Returns the collection of tokens indexed by idxs. '''
        return [sep.join([tokens[k].text for k in idx]) for idx in idxs]

    def batch_aspects_lookup(
        self,
        batch_tokens: List[List[str]],
        indexes: List[List[List[int]]]
    ) -> List[List[List[str]]]:
        ''' Batch-version of aspects_lookup. '''
        return [self.aspects_lookup(tokens, idxs) for idxs, tokens in zip(indexes, batch_tokens)]

    def test_epoch_end(self, outputs: Tuple[List[List[str]], List[List[str]]]) -> None:
        aspect_terms, polarity_terms = zip(*[sample for batch in outputs for sample in zip(*batch)])
        self.predictions = [{'targets': list(zip(*pred))} for pred in zip(aspect_terms, polarity_terms)]

        if not self.hparams.test_only and self.evaluate_callback:
            self.gold = [{'targets': [(1, *t) for t in terms]} for terms in self.gold]
            scores, precision, recall, f1 = self.evaluate_callback(self.gold, self.predictions)
            self.log_dict({'precision': precision, 'recall': recall, 'f1': f1})
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


class AspectMultistepClassifier(pl.LightningModule):
    '''
    Experiment with a multistep classifier that predicts labels for A+B using
    individually trained models for task A and task B.
    '''
    def __init__(
        self,
        ner_model: pl.LightningModule,
        polarity_model: pl.LightningModule
    ):
        super().__init__()
        self.ner_model = ner_model
        self.polarity_model = polarity_model
        self.predictions = []
        self.gold = []

        self.aspect_pred = []
        self.polarity_pred = []


    @torch.no_grad()
    def evaluation(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ):
        x, y = batch
        y_hat = self.ner_model.model(x).argmax(-1)

        # Extract aspect indexes from the first model
        x['aspect_indexes'], _ = zip(*[utils.extract_aspect_indices(
            prediction_idxs, length, *self.ner_model.bio_idxs, return_tensors=True
            ) for prediction_idxs, length in zip(y_hat, x['lengths'])])

        self.aspect_pred += [[' '.join([sent_tokens[idx].text for idx in idxs])
            for idxs in sent_idx]
            for sent_idx, sent_tokens in zip(x['aspect_indexes'], x['tokens'])]
        
        y_hat = self.polarity_model.model(x).argmax(-1)

        aspects_indexes = pad_sequence(
            [torch.ones(len(idxs)) for idxs in x['aspect_indexes']],
            batch_first=True, padding_value=self.polarity_model.ignore_index)

        # Apply masking to aspect_indexes, ignoring padded elements
        aspects_mask = aspects_indexes != self.polarity_model.ignore_index
        # Extract predictions from the second model
        self.polarity_pred += [pred[mask].tolist() for pred, mask in zip(y_hat, aspects_mask)]

        # Pair predictions of model A with predictions of model B (i.e. list of (term_i, polarity_i))
        self.predictions += [list(zip(aspects, predictions))
            for aspects, predictions in zip(self.aspect_pred, self.polarity_pred)]

        # Store gold labels to perform evaluation
        gold_mask = y['polarity'] != self.polarity_model.ignore_index
        self.gold += [list(zip(aspects, polarities[mask].int().tolist()))
            for aspects, polarities, mask in zip(y['aspect'], y['polarity'], gold_mask)]

    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ):
        x, y = batch
        logits = self.evaluation(batch)
