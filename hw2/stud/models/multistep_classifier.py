from typing import *
from dataclasses import asdict
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from stud.models.ner_classifier import NERClassifier
from stud.models.polarity_classifier import PolarityClassifier
from stud.layers.positional_encoder import PositionalEncoding
from axial_positional_embedding import AxialPositionalEmbedding
from stud.layers.multihead_attention import MultiheadSelfAttention
from stud.layers.transformer_encoder import TransformerEncoder
from stud.layers.attention_lstm import AttentionLSTM
from stud.constants import *
from stud.torch_utils import batch_scatter_mean
from torchtext.vocab import Vocab
from stud import utils
from torchcrf import CRF
from contextlib import redirect_stdout
import pytorch_lightning as pl
import torch
import torch.nn as nn
import io

def model_from(
    ab_hparams: utils.HParams,
    cd_hparams: utils.HParams,
    category_vocab: Vocab,
    pos_vocab:Vocab
) -> nn.Module:
    ''' Returns the correct model from the input hparams. '''
    if ab_hparams.model_name == 'bert_lstm' and pos_vocab:
        bert = BertModel.from_pretrained(utils.get_bert_path(ab_hparams))
        # Using BERT as a frozen encoder
        bert.eval()
        return BERTLSTMPOSClassification(ab_hparams, cd_hparams, category_vocab, bert, len(pos_vocab))
    raise Exception(f'{ab_hparams.model_name} not supported!')

class BERTLSTMPOSClassification(nn.Module):
    def __init__(
        self,
        ab_hparams: utils.HParams,
        cd_hparams: utils.HParams,
        category_vocab: Vocab,
        bert: BertModel,
        pos_vocab_size: int
    ):
        super().__init__()
        self.ab_hparams = ab_hparams
        self.cd_hparams = cd_hparams
        self.category_vocab = category_vocab
        self.bert = bert

        self.pos_embedding = nn.Embedding(pos_vocab_size, ab_hparams.pos_embedding_dim)
        self.dropout = nn.Dropout(ab_hparams.dropout)
        self.input_dim = self.bert_output_dim + ab_hparams.w2v_embedding_dim + ab_hparams.pos_embedding_dim

        if ab_hparams.sentence_encoder == 'lstm':
            self.sent_dim = 2 * ab_hparams.hidden_dim
            self.lstm = nn.LSTM(self.input_dim, ab_hparams.hidden_dim, batch_first=True, bidirectional=True)
            self.sent_encoder = lambda x: self.lstm(x)[0]
        
        elif ab_hparams.sentence_encoder == 'mlp':
            self.sent_dim = ab_hparams.hidden_dim
            self.sent_encoder = nn.Linear(self.input_dim, self.sent_dim)
        
        elif ab_hparams.sentence_encoder == 'attention_lstm':
            self.sent_dim = self.input_dim + 2 * ab_hparams.hidden_dim
            self.sent_encoder = AttentionLSTM(self.input_dim, ab_hparams)

        elif ab_hparams.sentence_encoder == 'transformer':
            self.sent_dim = self.input_dim
            self.sent_encoder = TransformerEncoder(self.input_dim, ab_hparams)
            
        self.ab_decoder = nn.Linear(self.sent_dim, len(ab_hparams.output_vocab))
        self.cd_decoder = nn.Linear(self.sent_dim, len(category_vocab) * len(cd_hparams.output_vocab))

    @property
    def bert_output_dim(self) -> int:
        ''' Returns BERT output dimension from the chosen pooling strategy. '''
        return (
            self.ab_hparams.input_dim * len(self.ab_hparams.layers_to_merge)
            if self.ab_hparams.strategy == 'cat'
            else self.ab_hparams.input_dim
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            batch_out = self.bert(**x['batch'], output_hidden_states=True)
        batch_out = utils.merge_layers(
            batch_out.hidden_states, strategy=self.ab_hparams.strategy,
            layers_to_merge=self.ab_hparams.layers_to_merge
        )

        # Merge WordPiece embeddings into token embeddings
        batch_out = batch_scatter_mean(batch_out, x['mask'])

        batch_pos = self.pos_embedding(x['pos_indices'])
        batch_out = torch.cat([batch_out, x['indices'], batch_pos], dim=-1)

        batch_out = self.sent_encoder(self.dropout(batch_out))

        ab_out = self.ab_decoder(torch.relu(batch_out))

        batch_out = batch_out.mean(dim=1)

        cd_out = self.cd_decoder(torch.relu(batch_out))
        
        return ab_out, cd_out


class MultistepClassifier(pl.LightningModule):
    '''
    Multistep classifier uses multitask learning to solve both tasks A+B and C+D.

    :param ab_hparams: hyperparameters and target vocab for task A+B
    :param cd_hparams: hyperparameters and target vocab for task C+D
    :param category_vocab: output vocabulary for categories
    :param pos_vocab: output vocabulary for pos tags
    :param evaluate_callback: callback function used to evaluate the model
    :param ab_class_weights: weights tensor used when optimizing the loss for task A+B
    :param cd_class_weights: weights tensor used when optimizing the loss for task C+D
    :param mode: only used during inference, whether to return predictions for A+B or C+D
    '''
    def __init__(
        self,
        ab_hparams: utils.HParams,
        cd_hparams: utils.HParams,
        category_vocab: Vocab,
        pos_vocab: Optional[Vocab] = None,
        evaluate_callback: Optional = None,
        ab_class_weights: Optional = None,
        cd_class_weights: Optional = None,
        mode: Optional[str] = 'cd'
    ):
        super().__init__()
        self.save_hyperparameters(asdict(ab_hparams))
        self.ab_hparams = ab_hparams
        self.cd_hparams = cd_hparams
        self.category_vocab = category_vocab

        self.ab_output_dim = len(ab_hparams.output_vocab)
        self.cd_output_dim = len(cd_hparams.output_vocab)

        self.model = model_from(ab_hparams, cd_hparams, category_vocab, pos_vocab)
        self.evaluate_callback = evaluate_callback
        self.mode = mode
        self.use_crf = ab_hparams.use_crf

        self.ab_ignore_index = ab_hparams.output_vocab.get_default_index()
        self.cd_ignore_index = cd_hparams.output_vocab.get_default_index()
        
        # Define a custom loss function for each subtask (i.e. A+B and C+D)
        if self.use_crf:
            self.crf = CRF(len(ab_hparams.output_vocab), batch_first=True)
            self.crf_reduction = 'token_mean'
            # Manually set to zero some transitions
            # https://github.com/kmkurn/pytorch-crf/issues/63#issuecomment-639927565
            ner_vocab = ab_hparams.output_vocab
            pad_idx, outside_idx = ner_vocab.lookup_indices([PAD_TOKEN, 'O'])
            inside_idxs = ner_vocab.lookup_indices([k for k in ner_vocab.get_itos() if k[0] == 'I'])
            torch.nn.init.constant_(self.crf.start_transitions[pad_idx], CRF_SET_TO_ZERO)
            torch.nn.init.constant_(self.crf.transitions[outside_idx, inside_idxs], CRF_SET_TO_ZERO)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ab_ignore_index, weight=ab_class_weights)
        self.loss_fn2 = nn.CrossEntropyLoss(ignore_index=self.cd_ignore_index, weight=cd_class_weights)

        self.ab_predictions = []
        self.cd_predictions = []

        self.ab_gold = []
        self.cd_gold = []

    @property
    def predictions(self) -> List[Dict]:
        ''' Returns predictions in a format compatible with Docker tester. '''
        if self.mode == 'cd':
            return self.cd_predictions
        return self.ab_predictions

    @property
    def bio_itos(self) -> Tuple[List[int], List[int], int]:
        ''' Returns the extended BIOs indexes. '''
        return (
            utils.vocab_tokens_startswith(self.ab_hparams.output_vocab, 'B'),
            utils.vocab_tokens_startswith(self.ab_hparams.output_vocab, 'I'),
            self.ab_hparams.output_vocab['O'])
        
    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        ab_hat, cd_hat = self.model(x)

        if self.use_crf:
            loss = -self.crf(ab_hat, y['ner'].long(), mask=x['padding_mask'], reduction=self.crf_reduction)
        else:
            loss = self.loss_fn(ab_hat.view(-1, self.ab_output_dim), y['ner'].view(-1).long())
        loss += self.loss_fn2(cd_hat.view(-1, self.cd_output_dim), y['category_indices'].view(-1).long())
        
        metrics = {LOGGER_TRAIN_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def evaluation(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        ab_hat, cd_hat = self.model(x)
        loss = 0

        if not self.hparams.test_only:
            if self.use_crf:
                loss = -self.crf(ab_hat, y['ner'].long(), mask=x['padding_mask'], reduction=self.crf_reduction)
            else:
                loss = self.loss_fn(ab_hat.view(-1, self.ab_output_dim), y['ner'].view(-1).long())
            loss += self.loss_fn2(cd_hat.view(-1, self.cd_output_dim), y['category_indices'].view(-1).long())
        return loss, ab_hat, cd_hat

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int
    ):
        loss, ab_logits, cd_logits = self.evaluation(batch)
        x, y = batch

        ab_preds = self.ab_predictions_from_logits(ab_logits, x)
        cd_preds = self.cd_predictions_from_logits(cd_logits)

        self.ab_gold += [list(zip(*labels)) for labels in zip(y['aspect'], y['polarity_labels'])]
        self.cd_gold += [list(zip(*sample)) for sample in zip(y['category_labels'], y['category_polarities_labels'])]

        metrics = {LOGGER_VALID_LOSS: loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return ab_preds, cd_preds

    def batch_aggregate_polarities(
        self,
        batch_idxs: List[int]
    ) -> List[List[str]]:
        return [[utils.aggregate_polarities(idxs, self.ab_hparams.output_vocab) for idxs in sent_idxs] for sent_idxs in batch_idxs]

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

    def ab_predictions_from_logits(
        self,
        logits: torch.Tensor,
        x: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        ''' Returns the predictions for task A+B. '''
        if self.use_crf:
            ab_preds = self.crf.decode(logits, mask=x['padding_mask'])
        else:
            ab_preds = logits.argmax(-1).detach().cpu()

        aspect_indexes, polarity_terms = zip(*[utils.extract_aspect_indices(
            prediction_idxs, length, *self.bio_itos, return_tensors=True
            ) for prediction_idxs, length in zip(ab_preds, x['lengths'])])

        aspect_terms = self.batch_aspects_lookup(x['tokens'], aspect_indexes)
        polarity_terms = self.batch_aggregate_polarities(polarity_terms)

        ab_preds = [{'targets': list(zip(*pred))} for pred in zip(aspect_terms, polarity_terms)]
        return ab_preds

    def cd_predictions_from_logits(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        ''' Returns the predictions for task C+B. '''
        cd_preds = logits.view(-1, len(CATEGORY_TAGS), len(self.cd_hparams.output_vocab)).argmax(-1)
        cd_preds = [[(self.category_vocab.lookup_token(k), self.cd_hparams.output_vocab.lookup_token(p))
                    for k, p in enumerate(pred) if p > 1] for pred in cd_preds]

        cd_preds = [{'categories': pred} for pred in cd_preds]
        return cd_preds

    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int = 0
    ) -> None:
        x, y = batch
        loss, ab_logits, cd_logits = self.evaluation(batch)

        ab_preds = self.ab_predictions_from_logits(ab_logits, x)
        cd_preds = self.cd_predictions_from_logits(cd_logits)

        if not self.ab_hparams.test_only:
            self.ab_gold += [list(zip(*labels)) for labels in zip(y['aspect'], y['polarity_labels'])]
            self.cd_gold += [list(zip(*sample)) for sample in zip(y['category_labels'], y['category_polarities_labels'])]

            metrics = {LOGGER_TEST_LOSS: loss}
            self.log_dict(metrics)

        return ab_preds, cd_preds

    def validation_epoch_end(self, outputs: List[List[Tuple[str, str]]]) -> None:
        self.ab_predictions, self.cd_predictions = zip(*[sample for batch in outputs for sample in zip(*batch)])

        if self.evaluate_callback:
            # Disable verbose logging, using wandb for this purpose
            with redirect_stdout(io.StringIO()):
                self.ab_gold = [{'targets': [(1, *t) for t in terms]} for terms in self.ab_gold]
                self.cd_gold = [{'categories': batch} for batch in self.cd_gold]
                ab_scores, ab_precision, ab_recall, ab_f1 = self.evaluate_callback(self.ab_gold, self.ab_predictions, 'Aspect Sentiment')
                c_scores, ext_precision, ext_recall, ext_f1 = self.evaluate_callback(self.cd_gold, self.cd_predictions, 'Category Extraction')
                cd_scores, snt_precision, snt_recall, snt_f1 = self.evaluate_callback(self.cd_gold, self.cd_predictions, 'Category Sentiment')
                metrics = {
                    'trainer/val_aspect_sentiment_f1': ab_f1, 'trainer/val_cat_extract_f1': ext_f1, 'trainer/val_cat_sentiment_f1': snt_f1,
                    'trainer/val_aspect_sentiment_f1_macro': ab_scores['ALL']['Macro_f1'],
                    'trainer/val_cat_extract_f1_macro': c_scores['ALL']['Macro_f1'],
                    'trainer/val_cat_sentiment_f1_macro': cd_scores['ALL']['Macro_f1']
                    }
                self.log_dict(metrics)

        self.ab_gold = []
        self.cd_gold = []

    def test_epoch_end(self, outputs: List[List[Tuple[str, str]]]) -> None:
        self.ab_predictions, self.cd_predictions = zip(*[sample for batch in outputs for sample in zip(*batch)])

        if not self.ab_hparams.test_only and self.evaluate_callback:
            self.ab_gold = [{'targets': [(1, *t) for t in terms]} for terms in self.ab_gold]
            self.cd_gold = [{'categories': batch} for batch in self.cd_gold]
            ab_scores, ab_precision, ab_recall, ab_f1 = self.evaluate_callback(self.ab_gold, self.ab_predictions, 'Aspect Sentiment')
            c_scores, ext_precision, ext_recall, ext_f1 = self.evaluate_callback(self.cd_gold, self.cd_predictions, 'Category Extraction')
            cd_scores, snt_precision, snt_recall, snt_f1 = self.evaluate_callback(self.cd_gold, self.cd_predictions, 'Category Sentiment')
            metrics = {
                'aspect_sentiment_precision': ab_precision, 'aspect_sentiment_recall': ab_recall, 'aspect_sentiment_f1': ab_f1, 'aspect_sentiment_f1_macro': ab_scores['ALL']['Macro_f1'],
                'cat_extract_precision': ext_precision, 'cat_extract_recall': ext_recall, 'cat_extract_f1': ext_f1, 'cat_extract_f1_macro': c_scores['ALL']['Macro_f1'],
                'cat_sentiment_precision': snt_precision, 'cat_sentiment_recall': snt_recall, 'cat_sentiment_f1': snt_f1, 'cat_sentiment_f1_macro': cd_scores['ALL']['Macro_f1']
                }
            self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())