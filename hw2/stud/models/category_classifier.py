from typing import *
from dataclasses import asdict
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from stud.models.ner_classifier import NERClassifier
from stud.models.polarity_classifier import PolarityClassifier
from stud.constants import LOGGER_TRAIN_LOSS, LOGGER_VALID_LOSS, LOGGER_TEST_LOSS
from stud.torch_utils import arg_where_equals, batch_scatter_mean
from torchtext.vocab import Vocab
from stud import utils
import pytorch_lightning as pl
import torch
import torch.nn as nn

def model_from(hparams: utils.HParams, max_length: int) -> nn.Module:
    ''' Returns the correct model from the input hparams. '''
    if hparams.model_name == 'bert_lstm':
        bert = BertModel.from_pretrained(utils.get_bert_path(hparams))
        # Using BERT as a frozen encoder
        bert.eval()
        return BERTLSTMClassification(hparams, bert, max_length=max_length)
    raise Exception(f'{hparams.model_name} not supported!')

class BERTLSTMClassification(nn.Module):
    def __init__(
        self,
        hparams: utils.HParams,
        bert: BertModel,
        max_length: Optional[int] = None
    ):
        super().__init__()
        self.hparams = hparams
        self.bert = bert
        self.max_length = max_length
        self.lstm = nn.LSTM(self.bert_output_dim, hparams.hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(max_length * 2 * hparams.hidden_dim
                                if max_length else 2 * hparams.hidden_dim, hparams.hidden_dim)
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

        batch_out = batch_out.reshape(batch_out.shape[0], -1) if self.max_length else batch_out.mean(dim=1)
        
        batch_out = torch.relu(self.dropout(self.fc1(batch_out)))
        batch_out = self.fc2(batch_out)
        return batch_out

class CategoryClassifier(pl.LightningModule):
    '''
    Category classifier identifies category terms for task C+D.

    :param hparams: hyperparameters and target vocab to set up the model
    :param evaluate_callback: callback function used to evaluate the model
    :param max_length: optional, assumes all sequences to be truncated at max_length
    '''
    def __init__(
        self,
        hparams: utils.HParams,
        evaluate_callback: Optional = None,
        max_length: Optional[int] = None
    ):
        super().__init__()
        self.save_hyperparameters(asdict(hparams))
        self.output_dim = len(hparams.output_vocab)
        self.model = model_from(hparams, max_length)
        self.evaluate_callback = evaluate_callback
        self.ignore_index = hparams.output_vocab.get_default_index()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.gold = []

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y['category_indices'])
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
            loss = self.loss_fn(y_hat, y['category_indices'])
        return loss, y_hat

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ):
        loss, logits = self.evaluation(batch)
        metrics = {LOGGER_VALID_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)

    @property
    def output_vocab(self):
        return self.hparams.output_vocab
 
    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int = 0
    ) -> None:
        x, y = batch
        loss, logits = self.evaluation(batch)
        # Extract predictions
        y_hat = torch.round(torch.sigmoid(logits))
        if not self.hparams.test_only:
            self.gold += [list(zip(*sample)) for sample in zip(y['category_labels'], y['category_polarities_labels'])]
        
        # Predicted categories are 1s, while 0s at position i means that category i was not predicted
        categories_polarities = [self.output_vocab.lookup_tokens(arg_where_equals(pred, 1)) for pred in y_hat]
        predictions = [[tuple(p.split('_')) for p in preds] for preds in categories_polarities]
        return predictions

    def test_epoch_end(self, outputs: List[List[Tuple[str, str]]]) -> None:
        self.predictions = [{'categories': pred} for batch in outputs for pred in batch]
        if not self.hparams.test_only and self.evaluate_callback:
            self.gold = [{'categories': batch} for batch in self.gold]
            *_, ext_precision, ext_recall, ext_f1 = self.evaluate_callback(self.gold, self.predictions, 'Category Extraction')
            *_, snt_precision, snt_recall, snt_f1 = self.evaluate_callback(self.gold, self.predictions, 'Category Sentiment')
            metrics = {
                'cat_extract_precision': ext_precision, 'cat_extract_recall': ext_recall, 'cat_extract_f1': ext_f1,
                'cat_sentiment_precision': snt_precision, 'cat_sentiment_recall': snt_recall, 'cat_sentiment_f1': snt_f1
                }
            self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())