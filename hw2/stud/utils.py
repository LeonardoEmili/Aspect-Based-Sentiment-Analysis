from typing import *
from stud.constants import *
from stud.torch_utils import gpus, compute_scatter_mask
from tqdm.notebook import tqdm
from torchtext.vocab import build_vocab_from_iterator, Vocab, vocab
from collections import OrderedDict, namedtuple, Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from dataclasses import dataclass, asdict, field
from transformers import PreTrainedTokenizerFast, logging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer
from operator import itemgetter
from random import sample
import math
import torchtext.vocab
import torch
import torch.nn as nn
import os
import nltk

# Set verbosity level to ERROR, ignoring warnings
logging.set_verbosity_error()

@dataclass
class HParams:
    output_vocab: Vocab
    input_dim: int = 768
    pos_embedding_dim: int = 50
    batch_size: int = 256
    hidden_dim: int = 256
    dropout: float = 0.6
    lr: int = 0.1
    model_name: str = 'bert_lstm'
    bert_name: str = 'bert-base-uncased'
    cached_bert_path: str = 'model/bert-base-uncased'
    layers_to_merge: List[int] = field(default_factory=lambda: [-1, -2, -3, -4])
    strategy: str = 'cat'
    epochs: int = 10
    ner_model_path: Optional[str] = None
    polarity_model_path: Optional[str] = None
    device: str = 'cuda'
    test_only: bool = False
    extended_bio: bool = False
    pos_embedding_dim: int = 30
    w2v_embedding_dim: int = 300
    sentence_encoder: str = 'lstm'
    num_heads: int = 2
    use_crf: bool = True

Sentence = namedtuple('Sentence', 'id spans categories polarities')
Span = namedtuple('Span', 'text tokens aspect polarity')
Token = namedtuple('Token', 'text lemma pos')

class Vectors(object):
    '''
    Extend torchtext.Vectors to support Word2Vec pretrained word embeddings:
    https://pytorch.org/text/stable/_modules/torchtext/vocab.html#Vectors
    '''
    def __init__(
        self,
        name: str,
        cache: Optional[str] = None,
        url: Optional[str] = None,
        unk_init: Optional[Callable[..., torch.Tensor]] = torch.Tensor.zero_,
        max_vectors: Optional[int] = None,
        train_vocab: Optional[Vocab] = None,
    ):
        self.name = name
        if name == 'Word2Vec':
            url = WORD2VEC_BIN_PATH if url is None else url
            self.vocab, self.vectors = load_pretrained_embeddings(
                train_vocab=train_vocab, path=url,
                words_limit=max_vectors, tag=name,
            )
        else:
            vocab.Vectors(name, cache, url, unk_init, max_vectors)

    def __len__(self):
        return len(self.vectors)

    @staticmethod
    def from_cached(path: str = WORD2VEC_CACHE_PATH):
        return torch.load(path)

    @property
    def dim(self):
        return self.vectors.shape[1]

    @property
    def itos(self):
        return self.vocab.get_itos()

    @property
    def stoi(self):
        return self.vocab.get_stoi()

    def __getitem__(self, token: str):
        return self.vectors[self.vocab[token]]    

    def get_vecs_by_tokens(
        self,
        tokens: List[str],
        lower_case_backup: bool = False
    ):
        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True
        if not lower_case_backup:
            indices = [self[token] for token in tokens]
        else:
            indices = [self[token] if token in self.stoi
                       else self[token.lower()]
                       for token in tokens]
        vecs = torch.stack(indices)
        return vecs[0] if to_reduce else vecs

def build_vocab(
    symbols: List[str],
    specials: List[str] = [PAD_TOKEN],
    min_freq: int = 1
) -> Vocab:
    ''' Returns a torchtext.Vocab object from input symbols. '''
    vocab = build_vocab_from_iterator(
        [symbols], specials=specials,
        special_first=True, min_freq=min_freq
        )
    vocab.set_default_index(PADDING_INDEX)
    return vocab

def load_pretrained_embeddings(
    train_vocab: Optional[Vocab] = None,
    path: str = WORD2VEC_BIN_PATH,
    words_limit: int = 1_000_000,
    tag: str = 'Word2Vec',
    delim: str = ' ',
) -> Tuple[Vocab, torch.Tensor]:
    ''' Loads pretrained embeddings from file and maps vocabulary words to vectors. '''
    if tag == 'Word2Vec':
        # Word2Vec are originally stored as binary file, parse it into plain file to be used later
        decode_word2vec_binaries(path)
    
    # Define the mapping to vectorize sentences and the embedding tensor
    vocab_words = [PAD_TOKEN, UNK_TOKEN]
    vectors_store = []

    with open(f'{path}.txt', 'r') as f:
        if tag == 'Word2Vec':
            n, embedding_size = map(int, next(f).split())
        elif tag == 'GloVe':
            n, embedding_size = (None, 300)
        else:
            raise Exception('Supported embeddings are Word2Vec and GloVe.')

        # Initialize three vectors: respectively for <PAD> and <UNK> tokens
        vectors_store.append(torch.zeros(embedding_size))
        vectors_store.append(torch.zeros(embedding_size))

        n = min(n, words_limit)
        progress = tqdm(f, total=n, desc=f'Loading pretrained {tag} embeddings')
        for i, line in enumerate(progress):
            # Read up to words_limit elements (special tokens excluded)
            if words_limit is not None and len(vocab_words) >= words_limit + 2: break
            word, *embedding = line.split(delim)
            # It is important to only use words that are present in the training set
            if train_vocab is not None and word not in train_vocab: continue
            vocab_words.append(word)
            embedding = torch.tensor([float(c) for c in embedding])
            vectors_store.append(embedding)

    out_vocab = vocab(OrderedDict([(w,1) for w in vocab_words]))
    out_vocab.set_default_index(PADDING_INDEX)
    vectors_store = torch.stack(vectors_store)
    return out_vocab, vectors_store

# TODO: remove this function
def decode_word2vec_binaries(path: str) -> None:
    ''' Utility function used to decode Word2Vec embeddings from .bin file. '''
    if not os.path.exists(f'{path}.txt'):
        from gensim.models.keyedvectors import KeyedVectors
        # Import KeyedVectors to extract the word2vec data structure and save it into .txt file
        word2vec = KeyedVectors.load_word2vec_format(f'{path}.bin', binary=True)
        word2vec.save_word2vec_format(f'{path}.txt', binary=False)
        del word2vec    # Free it from memory and build the vocab by ourselves for the sake of the hw

def simple_collate_fn(
    batch: List[Tuple[torch.Tensor,Dict[str, Union[torch.Tensor, List[int]]]]],
    padding_value: int = PADDING_INDEX
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    ''' A simple collate function that expects pre-tokenized input sentences.'''
    X, Y = zip(*batch)
    X, X_pos_tags, tokens, aspect_indexes = zip(*[(
        x['indices'],x['pos_indices'], x['tokens'],x['aspect_indexes']) for x in X])
    lengths = torch.tensor([len(x) for x in X])
    X = pad_sequence(X, batch_first=True, padding_value=padding_value)
    X_pos_tags = pad_sequence(X_pos_tags, batch_first=True, padding_value=padding_value)
    ner_labels, polarity_indexes, aspect_labels, polarity_labels = zip(*[
        (y['ner'],y['polarity'],y['aspect'], y['polarity_labels']) for y in Y])
    ner_labels = pad_sequence(ner_labels, batch_first=True, padding_value=padding_value)
    polarity_indexes = pad_sequence(polarity_indexes, batch_first=True, padding_value=padding_value)
    return {
        'indices': X,
        'pos_indices': X_pos_tags,
        'lengths': lengths,
        'tokens': tokens,
        'aspect_indexes': aspect_indexes
    }, {
        'ner': ner_labels,
        'polarity': polarity_indexes,
        'aspect': aspect_labels,
        'polarity_labels': polarity_labels
        }

def collate_fn(
    tokenizer: PreTrainedTokenizerFast,
    batch: List[Tuple[torch.Tensor,Dict[str, Union[torch.Tensor, List[int]]]]],
    padding_value: int = PADDING_INDEX
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    '''
    Efficient collate function to pad pre-tokenized input sentences in a batch,
    it's default when using BERT-like tokenizers.
    '''
    X, Y = zip(*batch)
    X, X_pos_tags, tokens, mask, aspect_indexes = zip(*[(
        x['indices'], x['pos_indices'], x['tokens'], x['mask'], x['aspect_indexes']) for x in X])
    lengths = torch.tensor([len(t) for t in tokens])
    X = pad_sequence(X, batch_first=True, padding_value=padding_value)
    X_pos_tags = pad_sequence(X_pos_tags, batch_first=True, padding_value=padding_value)
    padding_mask = torch.arange(max(lengths))[None, :] < lengths[:, None]

    batch = tokenizer(
        [[x.text for x in t] for t in tokens], is_split_into_words = True,
        padding = True, truncation = True, return_tensors = 'pt'
    )
    
    ner_labels, polarity_indexes, aspect_labels, polarity_labels = zip(*[
        (y['ner'], y['polarity'], y['aspect'], y['polarity_labels']) for y in Y])
    
    category_labels, category_polarities_labels, category_indices, category_idxs, category_polarity_idxs = zip(*[
        (y['category_labels'], y['category_polarities_labels'], y['category_indices'],
        y['category_idxs'], y['category_polarity_idxs']) for y in Y])

    category_indices = torch.stack(category_indices)
    category_idxs = torch.stack(category_idxs)
    category_polarity_idxs = torch.stack(category_polarity_idxs)

    ner_labels = pad_sequence(ner_labels, batch_first=True, padding_value=padding_value)
    polarity_indexes = pad_sequence(polarity_indexes, batch_first=True, padding_value=padding_value)
    return {
        'indices': X,
        'batch': batch,
        'mask': mask,
        'pos_indices': X_pos_tags,
        'lengths': lengths,
        'tokens': tokens,
        'aspect_indexes': aspect_indexes,
        'padding_mask': padding_mask
    }, {
        'ner': ner_labels,
        'polarity': polarity_indexes,
        'aspect': aspect_labels,
        'polarity_labels': polarity_labels,
        'category_labels': category_labels,
        'category_polarities_labels': category_polarities_labels,
        'category_indices': category_indices,
        'category_idxs': category_idxs,
        'category_polarity_idxs': category_polarity_idxs
        }

def extract_ner_labels(
    X: Dict[str, torch.Tensor],
    Y: torch.Tensor,
    ner_vocab: Vocab
) -> List[List[str]]:
    outputs = []
    for k, (y, l, tokens) in enumerate(zip(Y, X['lengths'], X['tokens'])):
        span_tokens, tmp_tokens = [], []
        for j, label in enumerate(y[:l]):
            if ner_vocab['B'] == label or ner_vocab['I'] == label:
                tmp_tokens.append(tokens[j].text)
            elif len(tmp_tokens) > 0:
                span_tokens.append(' '.join(tmp_tokens))
                tmp_tokens = []
        outputs.append(span_tokens)
    return outputs

def log_n_samples(
    predicted_labels: List[List[str]],
    gold_labels: List[List[str]],
    n: int = 10
) -> None:
    ''' Debugging function to log model predictions. '''
    assert len(predicted_labels) == len(gold_labels) and len(predicted_labels) >= n
    for gold, pred in sample(list(zip(gold_labels, predicted_labels)), n):
        print(f'Gold: {gold}, predicted {pred}')
        print('\n======================\n')

def download_nltk_resources() -> None:
    ''' Downloader function for NLTK resources. '''
    success = True
    success &= nltk.download('wordnet')
    success &= nltk.download('punkt')
    success &= nltk.download('averaged_perceptron_tagger')
    success &= nltk.download('universal_tagset')
    if not success:
        raise Exception('Some of the needed resources cannot be downloaded, please try again.')

def wordpiece_mask(
    sent: List[List[str]],
    tokenizer: PreTrainedTokenizerFast,
    add_special_tokens: bool = True,
    pad_max_length: Optional[int] = None
) -> torch.Tensor:
    ''' Utility function used to apply wordpiece-level tokenization to a list of tokens. '''
    # Get wordpiece embeddings for each token in the input sentence
    encoded_span = tokenizer(
        sent, add_special_tokens=False, return_tensors='pt',
        padding=True, truncation=True
    )
    # Compute the mask to identify tokens that are from the same input token
    mask = compute_scatter_mask(encoded_span.input_ids, add_special_tokens)
    if pad_max_length and len(mask) > 1:
        mask = mask[:-1]
        n = pad_max_length + 2 if add_special_tokens else pad_max_length
        if n - len(mask) > 0:
            padding_mask = torch.arange(n - len(mask)) + mask[-1] + 1
            mask = torch.cat([mask, padding_mask])
    return mask

def merge_layers(
    x: Tuple[torch.Tensor], # tuple of n layers, usually n=12 with BERT
    layers_to_merge: List[int] = [-1, -2, -3, -4],
    strategy: str = 'cat'
    ) -> torch.Tensor:
    ''' Applies a pooling strategy to the input layers. '''
    if strategy == 'cat':
        return torch.cat([x[l] for l in layers_to_merge], dim=-1)
    elif strategy == 'sum':
        return sum([x[l] for l in layers_to_merge])
    elif strategy == 'mean':
        raise NotImplementedError('TODO: missing implementation of the mean strategy.')
    else:
        raise NotImplementedError('Use `cat` or `sum` as strategy.')

def pl_trainer(
    monitor: str = LOGGER_VALID_LOSS,
    mode: str = 'min',
    dirpath: str = TRAINER_DIRPATH,
    max_epochs: int = 50,
    log_every_n_steps: int = 5,
    deterministic: bool = True,
    use_cuda: bool = True,
    precision: int = 32
) -> Trainer:
    ''' Returns a pytorch_lightning trainer object according to the specified params. '''
    return Trainer(
        callbacks=[ModelCheckpoint(monitor=monitor, mode=mode, dirpath=dirpath)],
        logger=WandbLogger(),
        gpus=gpus(use_cuda=use_cuda),
        max_epochs=max_epochs,
        deterministic=deterministic,
        log_every_n_steps=log_every_n_steps,
        precision=precision
    )

def safe_index(
    arr: List[Any],
    obj: Any,
    k: int = 0,
    fallback_fn: Callable[..., int] = len
) -> int:
    '''
    Safer implementation of the List `index()` function that allows a fallback function
    to be called in case the object is not in the array. Its safety now relies on
    the safety of the fallback_fn function, when provided.

    :param arr: the input array
    :param obj: the object to search for
    :param k: the offset from which starting to search for
    :param fallback_fn: the fallback function used in case of fails
    :return int
    '''
    return k + arr[k:].index(obj) if obj in arr[k:] else fallback_fn(arr)

def safe_indices(
    arr: List[Any],
    objs: List[Any],
    k: int = 0,
    fallback_fn: Callable[..., int] = len
) -> int:
    '''
    Allows the function `safe_index` to be called on every objs, returning the first
    index found by the function.

    :param arr: the input array
    :param objs: the objects to search for
    :param k: the offset from which starting to search for
    :param fallback_fn: the fallback function used in case of fails
    :return int
    '''
    return min([safe_index(arr, obj, k, fallback_fn) for obj in objs])

def extract_aspect_indices(
    indices: Union[torch.Tensor, List[str], List[int]],
    length: List[int],
    b_tokens: List[Union[str, int]],
    i_tokens: List[Union[str, int]],
    o_token: Union[str, int],
    enforce_bio_schema: bool = True,
    return_tensors: bool = False
) -> Union[List[List[int]], List[torch.Tensor]]:
    '''
    Extracts indexes and predicted labels for aspect terms in the input sentence.
    
    :param indices: the list of BIO tags that denote the presence of NER entities
    :param b_tokens: the begin tokens (either the token itself or its identifier)
    :param i_tokens: the inside tokens (either the token itself or its identifier)
    :param o_token: the outside token (either the token itself or its identifier)
    :param enforce_bio_schema: whether the BIO schema should apply to the input indices (e.g. predictions)
    :param return_tensors: whether to return the output as pt tensor or list
    :return a list of lists (or tensors) specifying the position of NER entities
    '''
    assert len(indices) >= 1
    # Match function signature
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    if isinstance(b_tokens, int):
        b_tokens = [b_tokens]
    if isinstance(i_tokens, int):
        i_tokens = [i_tokens]
    if isinstance(length, torch.Tensor):
        length = length.item()

    indices = indices[:length]
    
    if enforce_bio_schema:
        new_indices = [indices[0] if indices[0] in b_tokens or indices[0] == o_token else o_token]
        last_idx = new_indices[0]
        for idx in indices[1:]:
            if idx in i_tokens and not (last_idx in b_tokens or last_idx in i_tokens):
                last_idx = o_token
            else:
                last_idx = idx
            new_indices.append(last_idx)

        indices = new_indices

    aspect_indexes = [[k] + [u for u in range(k+1, safe_indices(indices, b_tokens, k+1))
                        if indices[u] in i_tokens]
                        for k,idx in enumerate(indices)
                        if idx in b_tokens]

    aspect_labels = [[indices[idx] for idx in idxs] for idxs in aspect_indexes]
    if return_tensors:
        aspect_indexes = [torch.tensor(aspect_idx) for aspect_idx in aspect_indexes]
    return aspect_indexes, aspect_labels

def vocab_tokens_startswith(vocab: Vocab, pattern: str):
    ''' Utility function used to lookup for indices starting with `pattern`. '''
    return vocab.lookup_indices([t for t in vocab.get_itos() if t.startswith(pattern)])

def build_train_vocab(
    train_ds: Dataset,
    min_freq: int = 1,
    pad_token: str = PAD_TOKEN
) -> Vocab:
    ''' Returns the vocabulary computed on the train dataset. '''
    return build_vocab(
        (getattr(t, train_ds.token_field) for sent in train_ds for s in sent.spans for t in s.tokens),
        specials=[pad_token],
        min_freq=min_freq
    )

def build_target_vocabs(
    specials: List[str] = [NONE_TOKEN],
    pad_token: str = PAD_TOKEN
) -> Dict[str, Vocab]:
    ''' Builds output vocabularies for each subtask. '''
    polarity_vocab = build_vocab(POLARITY_TAGS, specials=[pad_token]+specials)
    ner_vocab = build_vocab(BIO_TAGS, specials=[pad_token])
    pos_vocab = build_vocab(POS_TAGS, specials=[pad_token])
    category_vocab = build_vocab(CATEGORY_TAGS, specials=[])

    ner_vocab_ext = build_vocab(
        [f'{b}_{p}' for b in ['B', 'I'] for p in POLARITY_TAGS if str(p) != NONE_TOKEN] + ['O'],
        specials=[pad_token])

    category_vocab_ext = build_vocab(
        [f'{c}_{p}' for c in CATEGORY_TAGS for p in POLARITY_TAGS if str(p) != NONE_TOKEN],
        specials=[])
        
    return {
        'ner': ner_vocab,
        'ner_ext': ner_vocab_ext,
        'polarity': polarity_vocab,
        'pos': pos_vocab,
        'category': category_vocab,
        'category_ext': category_vocab_ext
    }

def get_bert_path(hparams: HParams) -> str:
    ''' Prevents downloading BERT weights if already available. '''
    return (hparams.cached_bert_path
            if os.path.exists(hparams.cached_bert_path)
            else hparams.bert_name)
        
def load_hparams_dict(path: str, strict: bool = False, **kwargs) -> HParams:
    ''' Retrieves multiple hyperparams from file, conveniently packed into an utility function. '''
    hparams_dict = {k: HParams(**v, **kwargs) if isinstance(v, dict) else v for k,v in torch.load(path).items()}
    hparams_dict['strict'] = strict
    return hparams_dict

def load_hparams(path: str, **kwargs) -> HParams:
    ''' Retrieves the hyperparams dict from file. '''
    return HParams(**torch.load(path), **kwargs)

def load_tokenizer(hparams: HParams) -> BertTokenizer:
    return BertTokenizer.from_pretrained(get_bert_path(hparams))

def aggregate_polarities(
    polarity_indexes: List[int],
    polarity_vocab: Vocab,
    strategy: str = 'first'
) -> str:
    ''' Aggregates polarity predictions for MWE. '''
    polarities = polarity_vocab.lookup_tokens(polarity_indexes)
    if len(polarities) == 0: return polarities
    if strategy == 'first':
        polarity = polarities[0]
    elif strategy == 'frequent':
        polarity, _ = Counter(polarities).most_common(1)[0]
    return polarity if polarity == 'O' else polarity[2:]

def safe_itemgetter(*items, fallback_value: Optional = None):
    '''
    Implementation of safe itemgetter from operator that returns elements from
    a collection of *valid* keys. Whenever a given key is not present in obj, it
    is simply ignored instead of throwing KeyError.
    '''
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return obj[item]
    else:
        def g(obj):
            return tuple(obj[item] if item in obj else fallback_value for item in items)
    return g

def pad_sentence(
    spans: List[Span],
    pad_max_length: Optional[int],
    pad_token: str,
    trailing: bool = True,
    truncation: bool = True
    ) -> List[Span]:
    ''' Pad the input sentence adding `None` tokens to reach pad_max_length tokens. '''
    if pad_max_length:
        n_tokens = len([t for s in spans for t in s.tokens])
        pad_tokens = [Token(pad_token, pad_token, 'X')] * (pad_max_length - n_tokens)
        pad_text = ' '.join([t.text for t in pad_tokens])
        if len(spans) and spans[-1].polarity == NONE_TOKEN:
            span_text = spans[-1].text + ' ' + pad_text if trailing else pad_text + ' ' + spans[-1].text
            span_tokens = spans[-1].tokens + pad_tokens if trailing else pad_tokens + spans[-1].tokens
            spans[-1] = Span(span_text, span_tokens, spans[-1].aspect, spans[-1].polarity)
        else:
            spans.append(Span(pad_text, pad_tokens, NONE_TOKEN, NONE_TOKEN))

        if truncation:
            new_spans = []
            cnt = 0
            for span in spans:
                new_tokens = []
                for token in span.tokens:
                    if cnt >= pad_max_length:
                        continue
                    cnt += 1
                    new_tokens.append(token)
                new_text = ' '.join([t.text for t in new_tokens])
                if len(new_tokens) > 0:
                    new_spans.append(Span(new_text, new_tokens, span.aspect, span.polarity))
            spans = new_spans
            if cnt > pad_max_length:
                    print(f'Sequence of length {cnt} was truncated to maximum length {pad_max_length}.')
    return spans

def compute_class_weight(
    train_ds: Dataset,
    aspects_vocab: Vocab,
    categories_vocab: Vocab
) -> Tuple[torch.Tensor, torch.Tensor]:
    aspect_polarities = dict(Counter([p for _, s in train_ds for p in s['polarity_labels']]))
    category_polarities = dict(Counter([p for _, s in train_ds for p in s['category_polarities_labels']]))

    total_aspects = sum(aspect_polarities.values())
    total_categories = sum(category_polarities.values())

    ner_symbols = aspects_vocab.get_itos()
    aspect_weights = {f'{t}_{k}':total_aspects/v for k,v in aspect_polarities.items() for t in ['B', 'I']}
    aspect_weights['O'] = min(aspect_weights.values())
    aspect_weights[PAD_TOKEN] = PADDING_INDEX

    polarity_simbols = categories_vocab.get_itos()
    category_weights = {k:total_categories/v for k,v in category_polarities.items()}
    category_weights[NONE_TOKEN] = min(category_weights.values())
    category_weights[PAD_TOKEN] = PADDING_INDEX

    aspect_weights = torch.tensor(itemgetter(*ner_symbols)(aspect_weights), dtype=torch.float)
    category_weights = torch.tensor(itemgetter(*polarity_simbols)(category_weights), dtype=torch.float)

    return aspect_weights, category_weights

def _mask_duplicated(
    grouped: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    pad_value: int = PADDING_INDEX,
    ignore_value: int = 9
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    ''' Applies masking to duplicated sample to avoid inconsistent samples. '''
    first = grouped[0][1]['ner']
    # Stack grouped samples and check which elements differ using an arithmetic trick
    stacked = torch.stack([y['ner'] for _, y in grouped])
    mask = torch.tensor([sum(stacked[:, k]) != t*stacked.shape[0] for k,t in enumerate(first)])

    # Store valid values **before** masking
    prev_mask = stacked != ignore_value
    prev_values = stacked[prev_mask]

    # Apply padding and restore previous valid values
    stacked[:, mask] = pad_value
    stacked[prev_mask] = prev_values

    # Save masked ner indices as ground truth
    for k in range(len(grouped)): grouped[k][1]['ner'] = stacked[k]
    return grouped

def mask_duplicated(
    samples: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    # Group duplicated to see the differences among them
    grouped_by_id = []
    for x, y in samples:
        if x['id'] > len(grouped_by_id) - 1:
            grouped_by_id.append([])
        grouped_by_id[-1].append((x,y))

    masked_samples = [masked_sample for grouped in grouped_by_id for masked_sample in _mask_duplicated(grouped)]
    return masked_samples