from typing import *
from stud.constants import *
from torchtext.vocab import Vocab
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial
from operator import itemgetter
from stud.utils import safe_itemgetter
from tqdm.notebook import tqdm
from transformers import PreTrainedTokenizerFast

import pytorch_lightning as pl
import torch
import json
from stud import utils
import nltk
import sys

class DatasetSplit(Dataset):
    def __init__(
        self,
        args: List[Dict],
        token_field: str = 'lemma',
        pad_max_length: Optional[int] = None,
        truncation: bool = True,
        pad_token: str = PAD_TOKEN
    ):            
        self.token_field = token_field if token_field is not None else 'lemma'
        self.sents, self.targets, self.categories = self.read_samples(args)
        self.samples = list(zip(self.sents, self.targets, self.categories))
        self.pad_max_length = pad_max_length
        self.truncation = truncation
        self.pad_token = pad_token

    @property
    def target_default_dict(self) -> Dict[str, Any]:
        return {
            'indices': [(-1, -1)],
            'aspects': [NONE_TOKEN],
            'polarities': [NONE_TOKEN]
        }
    
    @property
    def target_columns(self) -> List[str]:
        return ['indices', 'aspects', 'polarities']

    @property
    def categories_default_dict(self) -> Dict[str, Any]:
        return {
            'categories': [NONE_TOKEN],
            'polarities': [NONE_TOKEN]
        }

    @property
    def categories_columns(self) -> List[str]:
        return ['categories', 'polarities']

    def read_samples(
        self,
        samples: List[Dict],
        file_header: Tuple[str] = ('categories', 'targets', 'text'),
        
    ) -> Tuple[List[str], Dict[str, Union[Tuple[int], str]]]:       
        # Initialize sentences (i.e. text field) as well as the other fields 
        *other, sents = zip(*[safe_itemgetter(*file_header, fallback_value=[])(s) for s in samples])
        targets = [()] * len(sents)
        categories = [()] * len(sents)

        # Unpack available data from different fields
        if len(other) >= 1: *other, targets = other
        if len(other) >= 1: *_, categories = other

        targets = [dict(zip(self.target_columns, zip(*t))) if len(t) else self.target_default_dict for t in targets]
        categories = [dict(zip(self.categories_columns, zip(*t))) if len(t) else self.categories_default_dict for t in categories]
        return list(sents), targets, categories

    def process_span(
        self,
        span: str,
        tokenizer: Callable[[str], str],
        pos_tagger: Callable[[str], str],
        lemmatizer: Callable[[str], str],
        aspect: Optional[str] = None,
        polarity: Optional[str] = None
    ) -> utils.Span:
        tokens = [utils.Token(t,lemmatizer(t),p) for t,p in pos_tagger(tokenizer(span))]
        return utils.Span(span, tokens, aspect, str(polarity))

    def process_sentence(
        self,
        id: int,
        sentence: str,
        targets: Dict[str, Union[str, Tuple[int]]],
        categories: Dict[str, Union[str, Tuple[int]]],
        tokenizer: Callable[[str], str],
        pos_tagger: Callable[[str], str],
        lemmatizer: Callable[[str], str],
    ) -> utils.Sentence:
        # Process target
        if 'polarities' in targets:
            indices, aspects, aspect_polarities = zip(*sorted(
                zip(targets['indices'], targets['aspects'], targets['polarities'])))
        else:
            indices, aspects = zip(*sorted(zip(targets['indices'], targets['aspects'])))
            aspect_polarities = [None] * len(indices)

        categories, category_polarities = zip(*sorted(zip(categories['categories'], categories['polarities'])))

        assert len(indices) > 0
        spans = []
        last_idx = 0
        for (start, end), aspect, aspect_polarity in zip(indices, aspects, aspect_polarities):
            start, end = (start, end) if start > -1 else (0, len(sentence))
            prefix_span = self.process_span(sentence[last_idx:start], tokenizer, pos_tagger, lemmatizer)
            aspect_span = self.process_span(sentence[start: end], tokenizer, pos_tagger,
                                            lemmatizer, aspect=aspect, polarity=aspect_polarity)
            if len(prefix_span.tokens) > 0:
                spans.append(prefix_span)
            if len(aspect_span.tokens) > 0:
                spans.append(aspect_span)
            last_idx = end
        if last_idx < len(sentence):
            spans += [self.process_span(sentence[last_idx:], tokenizer, pos_tagger, lemmatizer)]

        if self.pad_max_length:
            spans = utils.pad_sentence(spans, self.pad_max_length, self.pad_token, truncation=self.truncation)

        return utils.Sentence(id, spans, categories, category_polarities)

    def process_sentences(
        self,
        tokenizer: Callable[[str], str] = nltk.word_tokenize,
        pos_tagger: Callable[[str], str] = lambda t: nltk.pos_tag(t, tagset=NLTK_POS_TAGSET),
        lemmatizer: Callable[[str], str] = nltk.stem.WordNetLemmatizer().lemmatize,
    ) -> None:
        self.samples = [self.process_sentence(idx, *sample, tokenizer, pos_tagger, lemmatizer) for idx, sample in enumerate(self)]

    def vectorize_span(
        self,
        span: utils.Span,
        vectors: utils.Vectors
    ) -> torch.Tensor:
        return vectors.get_vecs_by_tokens(
            [getattr(t, self.token_field) for t in span.tokens]
        )

    def vectorize_sentence(
        self,
        sentence: utils.Sentence,
        target_vocabs: Dict[str, Vocab],
        vectors: Optional[utils.Vectors] = None,
        tokenizer: Optional = None,
        return_tensors: bool = False,
        extended_bio: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        aspect_labels = [span.aspect for span in sentence.spans if str(span.aspect) != NONE_TOKEN]
        token_polarity_labels = [span.polarity for span in sentence.spans for _ in span.tokens]
        token_polarity_indices = target_vocabs['polarity'].lookup_indices(token_polarity_labels)
        if return_tensors:
            token_polarity_indices = torch.tensor(token_polarity_indices)

        polarity_labels = [span.polarity for span in sentence.spans if str(span.polarity) != NONE_TOKEN]
        polarity_indexes = target_vocabs['polarity'].lookup_indices(polarity_labels)

        if return_tensors:
            polarity_indexes = torch.tensor(polarity_indexes)

        category_labels = list(sentence.categories)
        category_polarities_labels = list(sentence.polarities)

        if NONE_TOKEN in category_labels:
            # Equivalently we can check if category_indices has only the pad index (the default one). It means that
            # we haven't any categories for this sample, hence replace the placeholder with an empty list.
            category_labels = []
            category_polarities_labels = []

        category_x_polarities = [f'{c}_{p}' for c,p in zip(category_labels, category_polarities_labels)]

        # Create category labels for multinomial classification
        category_indices = torch.zeros(len(target_vocabs['category']), dtype=torch.long)
        available_idxs = torch.tensor(target_vocabs['category'].lookup_indices(category_labels), dtype=torch.long)
        category_indices[available_idxs] = torch.tensor(target_vocabs['polarity'].lookup_indices(category_polarities_labels), dtype=torch.long)
        if len(category_labels):
            unavailable_idxs, *_ = (torch.arange(len(category_indices)) != available_idxs.unsqueeze(1)).all(0).nonzero(as_tuple=True)
            category_indices[unavailable_idxs] = target_vocabs['polarity'][NONE_TOKEN]
        
        category_idxs = torch.zeros(len(target_vocabs['category']))
        _idxs = torch.tensor(target_vocabs['category'].lookup_indices(category_labels), dtype=torch.long)
        category_idxs[_idxs] = 1

        category_polarity_idxs = torch.zeros(len(target_vocabs['category']))
        for c, p in zip(category_labels, category_polarities_labels):
            category_polarity_idxs[target_vocabs['category'][c]] = target_vocabs['polarity'][p]
        
        if not extended_bio:
            ner_labels = [['O' if str(span.aspect) == NONE_TOKEN else 'I' for _ in span.tokens]
                            for span in sentence.spans]
            ner_labels = ['B' if l == 'I' and k == 0 else l for span in ner_labels
                            for k,l in enumerate(span)]
            current_ner_vocab = target_vocabs['ner']
        else:
            ner_labels = [['O' if str(span.aspect) == NONE_TOKEN else f'I_{span.polarity}'
                            for _ in span.tokens] for span in sentence.spans]
            ner_labels = [f'B{l[1:]}' if l[0] == 'I' and k == 0 else l
                            for span in ner_labels for k,l in enumerate(span)]
            current_ner_vocab = target_vocabs['ner_ext']

        ner_indices = current_ner_vocab.lookup_indices(ner_labels)
        b_tokens = [l for l in current_ner_vocab.get_itos() if l.startswith('B')]
        i_tokens = [l for l in current_ner_vocab.get_itos() if l.startswith('I')]

        if return_tensors:
            ner_indices = torch.tensor(ner_indices)

        aspect_indexes = [[k] + [u for u in range(k+1, utils.safe_indices(ner_labels, b_tokens, k+1))
                            if ner_labels[u] in i_tokens]
                            for k,label in enumerate(ner_labels) if label in b_tokens]
        if return_tensors:
            aspect_indexes = [torch.tensor(aspect_idx) for aspect_idx in aspect_indexes]

        tokens = [t.text for span in sentence.spans for t in span.tokens]
        mask = utils.wordpiece_mask(tokens, tokenizer, True, self.pad_max_length) if tokenizer else None

        # Encode input sentence (not needed if using pretrained tokenizer)
        x = None
        if vectors:
            x = torch.cat([self.vectorize_span(span, vectors) for span in sentence.spans], dim=0)

        tokens = [t for span in sentence.spans for t in span.tokens]
        pos_indices = [target_vocabs['pos'][t.pos] for span in sentence.spans for t in span.tokens]
        if return_tensors:
            pos_indices = torch.tensor(pos_indices)
        
        return {
            'id': sentence.id,
            'tokens': tokens,
            'indices': x,
            'mask': mask,
            'pos_indices': pos_indices,
            'aspect_indexes': aspect_indexes
        }, {
            'ner': ner_indices,
            'token_polarity': token_polarity_indices,
            'polarity': polarity_indexes,
            'aspect': aspect_labels,
            'polarity_labels': polarity_labels,
            'category_labels': category_labels,
            'category_polarities_labels': category_polarities_labels,
            'category_indices': category_indices,
            'category_idxs': category_idxs,
            'category_polarity_idxs': category_polarity_idxs
        }
    def _duplicate_aspects(self, sample):
        aspect_terms = [(s.aspect, s.polarity) for s in sample.spans if str(s.aspect) != NONE_TOKEN]
        if len(aspect_terms) == 0:
            return [sample]
        new_samples = []
        for aspect, polarity in aspect_terms:
            new_spans = []
            visited_index = -1
            for k, span in enumerate(sample.spans):
                if str(span.aspect) == NONE_TOKEN:
                    new_span = span
                else:
                    if visited_index >= 0:   # already visited
                        new_span = utils.Span(span.text, span.tokens, NONE_TOKEN, NONE_TOKEN)
                    else:
                        new_span = span
                        visited_index = k

                new_spans.append(new_span)
            new_samples.append(utils.Sentence(sample.id, new_spans, sample.categories, sample.polarities))
            spans = [utils.Span(s.text, s.tokens, NONE_TOKEN, NONE_TOKEN) if l == visited_index else s for l, s in enumerate(sample.spans)]
            sample = utils.Sentence(sample.id, spans, sample.categories, sample.polarities)
        return new_samples

    def duplicate_aspects(self):
        self.samples = [duplicated for sample in self for duplicated in self._duplicate_aspects(sample)]

    def mask_duplicated(self):
        self.samples = utils.mask_duplicated(self)

    def vectorize_sentences(
        self,
        target_vocabs: Dict[str, Vocab],
        vectors: Optional[utils.Vectors] = None,
        tokenizer: Optional = None,
        return_tensors: bool = False,
        extended_bio: bool = False
    ) -> None:
        self.samples = [self.vectorize_sentence(
            sample, target_vocabs, vectors, tokenizer, return_tensors, extended_bio=extended_bio)
            for sample in self]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.samples[idx]

class ABSADataset(pl.LightningDataModule):
    vectors = None

    def __init__(
        self,
        dev_arg: List[Dict],
        train_arg: Optional[List[Dict]] = None,
        batch_size: int = 256,
        collate_fn: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
        force_vectors_init: bool = False,
        cached_vectors: Optional[utils.Vectors] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        augment_train_data: bool = False,
        extended_bio: bool = False,
        use_class_weights: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = partial(utils.collate_fn, tokenizer) if tokenizer else collate_fn
        self.vectors = cached_vectors or ABSADataset.vectors
        self.prepare_data(train_arg, dev_arg, force_vectors_init, tokenizer, augment_train_data, extended_bio, **kwargs)
        self.ab_class_weights, self.cd_class_weights = None, None
        if use_class_weights:
            self.ab_class_weights, self.cd_class_weights = utils.compute_class_weight(self.train_ds, self.ner_ext_vocab, self.polarity_vocab)

    @classmethod
    def restaurants_from_path(cls, **kwargs):
        return cls.from_path(laptops_dev_path=None, merge_dev_sets=False, **kwargs)

    @classmethod
    def laptops_from_path(cls, **kwargs):
        return cls.from_path(restaurants_dev_path=None, merge_dev_sets=False, **kwargs)

    @classmethod
    def from_path(
        cls,
        laptops_train_path: str = LAPTOPS_TRAIN_PATH,
        laptops_dev_path: str = LAPTOPS_DEV_PATH,
        restaurants_train_path: str = RESTAURANTS_TRAIN_PATH,
        restaurants_dev_path: str = RESTAURANTS_DEV_PATH,
        merge_dev_sets: bool = False,
        **kwargs
    ):
        laptops_train, laptops_dev = [], []
        if laptops_dev_path:
            with open(laptops_train_path, 'r') as train_f, open(laptops_dev_path, 'r') as dev_f:
                laptops_train = json.load(train_f)
                laptops_dev = json.load(dev_f)

        restaurants_train, restaurants_dev = [], []
        if restaurants_dev_path:
            with open(restaurants_train_path, 'r') as train_f, open(restaurants_dev_path, 'r') as dev_f:
                restaurants_train = json.load(train_f)
                restaurants_dev = json.load(dev_f)
        
        train_set = laptops_train+restaurants_train
        dev_set = laptops_dev+restaurants_dev if merge_dev_sets else restaurants_dev
        if restaurants_dev_path is None:
            dev_set = laptops_dev
        return cls(dev_set, train_arg=train_set, **kwargs)

    def prepare_data(
        self,
        train_arg: List[Dict],
        dev_arg: List[Dict],
        force_vectors_init: bool,
        tokenizer: PreTrainedTokenizerFast,
        augment_train_data: bool,
        extended_bio: bool,
        duplicate_samples: bool = True,
        **kwargs
    ):
        if train_arg:
            self.train_ds = DatasetSplit(train_arg, **kwargs)
            self.train_ds.process_sentences()
            self.train_vocab = utils.build_train_vocab(self.train_ds)
        elif tokenizer is None:
            raise NotImplementedError('Load the train vocab from file to perform inference.')

        self.dev_ds = DatasetSplit(dev_arg, **kwargs)
        self.dev_ds.process_sentences()

        if train_arg:
            self.update_vectors(force_vectors_init=force_vectors_init)

        self.target_vocabs = utils.build_target_vocabs()
        vectorizer_kwargs = {
            'return_tensors': True, 'tokenizer': tokenizer, 'extended_bio': extended_bio
        }
        if train_arg and augment_train_data:
            self.train_ds.duplicate_aspects()
        
        if train_arg:
            self.train_ds.vectorize_sentences(self.target_vocabs, self.vectors, **vectorizer_kwargs)
            if augment_train_data:
                self.train_ds.mask_duplicated()
        self.dev_ds.vectorize_sentences(self.target_vocabs, self.vectors, **vectorizer_kwargs)

    def update_vectors(
        self,
        max_vectors: int = 1_000_000,
        force_vectors_init: bool = False,
        use_pretrained_vocab: bool = False
    ) -> None:
        if self.vectors is None or force_vectors_init:
            new_vectors = utils.Vectors(
                'Word2Vec', max_vectors=max_vectors,
                train_vocab=self.train_vocab if use_pretrained_vocab else None
            )
            self.vectors = new_vectors
            ABSADataset.vectors = new_vectors


    def _dataloader_kwargs(self, mode: str = 'train') -> Dict[str, Any]:
        return {
            'batch_size': self.batch_size,
            'drop_last': False,
            'shuffle': True if mode == 'train' else False,
            'collate_fn': self.collate_fn
        }

    @property
    def configs(self) -> Dict[str, any]:
        return {
            'ab_class_weights': self.ab_class_weights,
            'cd_class_weights': self.cd_class_weights
            }

    @property
    def feature_size(self) -> int:
        return self.vectors.dim

    @property
    def pos_vocab(self) -> Vocab:
        return self.target_vocabs['pos']

    @property
    def category_vocab(self) -> Vocab:
        return self.target_vocabs['category']

    @property
    def category_ext_vocab(self) -> Vocab:
        return self.target_vocabs['category_ext']
    
    @property
    def ner_ext_vocab(self) -> Vocab:
        return self.target_vocabs['ner_ext']

    @property
    def ner_vocab(self) -> Vocab:
        return self.target_vocabs['ner']

    @property
    def polarity_vocab(self) -> Vocab:
        return self.target_vocabs['polarity']
    
    @property
    def train_dataloader_kwargs(self):
        return self._dataloader_kwargs('train')

    @property
    def val_dataloader_kwargs(self):
        return self._dataloader_kwargs('val')

    @property
    def test_dataloader_kwargs(self):
        return self._dataloader_kwargs('test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_ds, **self.val_dataloader_kwargs)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_ds, **self.test_dataloader_kwargs)
