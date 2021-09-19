from typing import List

# Output tags
POS_TAGS: List[str] = ['NOUN', 'VERB', 'ADP', '.', 'DET', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PRT', 'NUM', 'X']
CATEGORY_TAGS: List[str] = ["anecdotes/miscellaneous", "price", "food", "ambience", "service"]
POLARITY_TAGS: List[str] = ['positive', 'conflict', 'negative', 'neutral']
BIO_TAGS: List[str] = ['B', 'I', 'O']

# Sentence-level symbols
PAD_TOKEN: str = '[PAD]'
UNK_TOKEN: str = '[UNK]'
NONE_TOKEN: str = 'None'

# Logger constants
LOGGER_TRAIN_LOSS: str = 'trainer/train_loss'
LOGGER_VALID_LOSS: str = 'trainer/val_loss'
LOGGER_TEST_LOSS: str = 'trainer/test_loss'

# Trainer constants
TRAINER_DIRPATH: str = 'checkpoints/'

# NLTK constants
NLTK_POS_TAGSET = 'universal'

# Dataset path constants
LAPTOPS_TRAIN_PATH: str = '../../data/laptops_train.json'
LAPTOPS_DEV_PATH: str = '../../data/laptops_dev.json'
RESTAURANTS_TRAIN_PATH: str = '../../data/restaurants_train.json'
RESTAURANTS_DEV_PATH: str = '../../data/restaurants_dev.json'

# Word2Vec data constants
WORD2VEC_BIN_PATH: str = '../../data/GoogleNews-vectors-negative300'
WORD2VEC_CACHE_PATH: str = 'model/w2v_weights.pth'

# Miscellaneous constants
PADDING_INDEX = 0
CRF_SET_TO_ZERO = -1e9
AXIAL_SHAPE_DEFAULT = (16, 16)  # max length 256