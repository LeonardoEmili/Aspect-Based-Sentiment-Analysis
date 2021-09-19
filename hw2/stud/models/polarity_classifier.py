from typing import *
from dataclasses import asdict
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from stud.constants import LOGGER_TRAIN_LOSS, LOGGER_VALID_LOSS, LOGGER_TEST_LOSS
from stud.torch_utils import get_device, batch_scatter_mean
import pytorch_lightning as pl
from stud import utils
import torch
import torch.nn as nn

class BERTLSTMClassification(nn.Module):
    def __init__(
        self,
        hparams: utils.HParams
    ):
        super().__init__()
        self.hparams = hparams
        self.bert = BertModel.from_pretrained(utils.get_bert_path(hparams))
        # Using BERT as a frozen encoder
        self.bert.eval()
        self.lstm = nn.LSTM(self.bert_output_dim, hparams.hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * 2 * hparams.hidden_dim, hparams.hidden_dim)
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

    def forward(
        self,
        x: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_out = self.bert(**x['batch'], output_hidden_states=True)

        batch_out = utils.merge_layers(
            batch_out.hidden_states, strategy=self.hparams.strategy,
            layers_to_merge=self.hparams.layers_to_merge
        )

        # Merge WordPiece embeddings into token embeddings
        batch_out = batch_scatter_mean(batch_out, x['mask'])
        batch_out, _ = self.lstm(self.dropout(batch_out))

        aspect_embeddings = [[torch.cat((sent_out.mean(0), sent_out[idxs+1].mean(dim=0)))
                                for idxs in aspect_idxs]
                                for aspect_idxs, sent_out in zip(x['aspect_indexes'], batch_out)]
        
        batch_size, _, hidden_dim = batch_out.shape
        aspects_num = len(max(x['aspect_indexes'], key=len))

        # Create a matrix of size (k,n), where k is the number of aspect terms and n the sequence length
        aspects_matrix = torch.zeros((batch_size, aspects_num * 2 * hidden_dim))
        for j, embeddings in enumerate(aspect_embeddings):
            # At train time, skip polarity prediction for unavailable aspect terms
            if len(embeddings) == 0: continue
            embeddings = torch.cat(embeddings, dim=-1)
            aspects_matrix[j, :embeddings.shape[-1]] = embeddings

        # Move the matrix back to device
        aspects_matrix = aspects_matrix.view(-1, 2 * hidden_dim)
        aspects_matrix = aspects_matrix.to(device=get_device(self))

        # Apply transformations and get predictions for the k aspect terms
        aspects_matrix = torch.relu(self.dropout(self.fc1(aspects_matrix)))
        aspects_matrix = self.fc2(aspects_matrix)

        aspects_matrix = aspects_matrix.view(batch_size, aspects_num, -1)
        return aspects_matrix


class PolarityClassifier(pl.LightningModule):
    '''
    Polarity classifier predicts polarities from pre-identified aspect terms.

    :param hparams: hyperparameters and target vocab to set up the model
    '''
    def __init__(self, hparams: utils.HParams):
        super().__init__()
        self.save_hyperparameters(asdict(hparams))
        self.output_dim = len(hparams.output_vocab)
        self.model = BERTLSTMClassification(hparams)
        self.ignore_index = hparams.output_vocab.get_default_index()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.gold = []

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['polarity'].view(-1).long())
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
            loss = self.loss_fn(y_hat.view(-1, self.output_dim), y['polarity'].view(-1).long())
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
        batch_idx: int = 0,
        allow_empty_samples: bool = True
    ) -> List[List[Tuple[str, str]]]:
        x, y = batch
        loss, logits = self.evaluation(batch)
        predictions = logits.argmax(-1)

        aspects_indexes = pad_sequence(
            [torch.ones(len(idxs)) for idxs in x['aspect_indexes']],
            batch_first=True, padding_value=self.ignore_index)

        # Mask out padding values and get model predictions
        aspects_mask = aspects_indexes != self.ignore_index
        polarity_terms = [y_hat[mask].tolist() for y_hat, mask in zip(predictions, aspects_mask)]

        # Polarity vocabulary is used to get from indexes back to labels
        polarity_terms = [self.hparams.output_vocab.lookup_tokens(pred) for pred in polarity_terms]
        predictions = [list(zip(aspects, polarities)) for aspects, polarities in zip(y['aspect'], polarity_terms)]

        if not self.hparams.test_only:
            self.gold += [y[mask].tolist()
                for y, mask in zip(y['polarity'].long(), aspects_mask) if allow_empty_samples or len(y[mask])]

            metrics = {LOGGER_TEST_LOSS: loss}
            self.log_dict(metrics)
        
        return predictions

    def test_epoch_end(self, outputs: List[List[Tuple[str, str]]]):
        self.predictions = [{'targets': pred} for batch in outputs for pred in batch]

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
