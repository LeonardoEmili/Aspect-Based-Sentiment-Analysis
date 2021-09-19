from typing import *
from dataclasses import asdict
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from stud import utils
from stud.torch_utils import batch_scatter_mean
from stud.constants import *
import torch
import torch.nn as nn

def model_from(hparams: utils.HParams) -> nn.Module:
    ''' Returns the correct model from the input hparams. '''
    if hparams.model_name == 'lstm':
        # Baseline model
        return LSTMClassification(hparams)
    if hparams.model_name == 'lstm_pos':
        # Using word embeddings + POS embeddings
        return LSTMPOSClassification(hparams)
    if hparams.model_name == 'bert_lstm':
        bert = BertModel.from_pretrained(utils.get_bert_path(hparams))
        # Using BERT as a frozen encoder
        bert.eval()
        return BERTLSTMClassification(hparams, bert)
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
        self.lstm = nn.LSTM(self.bert_output_dim, hparams.hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hparams.hidden_dim, hparams.hidden_dim)
        self.fc2 = nn.Linear(hparams.hidden_dim, len(hparams.output_vocab))
        self.dropout = nn.Dropout(hparams.dropout)

    @property
    def bert_output_dim(self):
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

class LSTMClassification(nn.Module):
    def __init__(
        self,
        hparams: utils.HParams
    ):
        super().__init__()
        self.hparams = hparams
        self.lstm = nn.LSTM(hparams.input_dim, hparams.hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hparams.hidden_dim, hparams.hidden_dim)
        self.fc2 = nn.Linear(hparams.hidden_dim, len(hparams.output_vocab))
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, _ = self.lstm(self.dropout(x['indices']))
        x = torch.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

class LSTMPOSClassification(nn.Module):
    def __init__(
        self,
        hparams: utils.HParams
    ):
        super().__init__()
        self.hparams = hparams
        self.pos_embedding = nn.Embedding(len(BIO_TAGS)+1, hparams.pos_embedding_dim)
        self.lstm = nn.LSTM(hparams.input_dim, hparams.hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_pos = nn.LSTM(hparams.pos_embedding_dim, hparams.hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * 2 * hparams.hidden_dim, hparams.hidden_dim)
        self.fc2 = nn.Linear(hparams.hidden_dim, len(hparams.output_vocab))
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, pos = x['indices'], x['pos_indices']
        x, _ = self.lstm(self.dropout(x))
        pos = self.pos_embedding(pos)
        pos, _ = self.lstm_pos(self.dropout(pos))
        x = torch.cat((x, pos), dim=-1)
        x = torch.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x


class NERClassifier(pl.LightningModule):
    '''
    NER classifier identifies aspect terms for task A.

    :param hparams: hyperparameters and target vocab to set up the model
    '''
    def __init__(self, hparams: utils.HParams):
        super().__init__()
        self.save_hyperparameters(asdict(hparams))
        self.output_dim = len(hparams.output_vocab)
        self.model = model_from(hparams)
        self.ignore_index = hparams.output_vocab.get_default_index()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.bio_idxs = hparams.output_vocab.lookup_indices(BIO_TAGS)
        self.aspect_indexes = []
        self.aspect_predictions = []
        self.batch_tokens = []
        self.gold = []

    @property
    def predictions(self):
        return [[' '.join([sent_tokens[idx].text for idx in idxs])
            for idxs in sent_idx]
            for sent_idx, sent_tokens in zip(self.aspect_indexes, self.batch_tokens)]

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['ner'].view(-1).long())
        metrics = {LOGGER_TRAIN_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def evaluation(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['ner'].view(-1).long())
        return loss, y_hat

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, logits = self.evaluation(batch)
        metrics = {LOGGER_VALID_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        x, y = batch
        loss, logits = self.evaluation(batch)
        y_hat = logits.argmax(-1)

        aspect_indexes, aspect_predictions = zip(*[utils.extract_aspect_indices(
            prediction_idxs, *self.bio_idxs, return_tensors=True
            ) for prediction_idxs in y_hat.detach().cpu()])

        self.aspect_indexes += aspect_indexes
        self.aspect_predictions += aspect_predictions

        self.batch_tokens += x['tokens']
        self.gold += y['aspect']
        assert len(self.aspect_indexes) == len(self.gold)

        metrics = {LOGGER_TEST_LOSS: loss}
        self.log_dict(metrics)

    def test_epoch_end(self, outputs: List):
        metrics = utils.evaluate_extraction(self.gold, self.predictions, debug=True)
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())