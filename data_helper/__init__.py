PADDING_TOKEN = "<pad>"
START_OF_SENTENCE_TOKEN = "<sos>"
END_OF_SENTENCE_TOKEN = "</eos>"
UNKNOWN_TOKEN = "<unk>"

PADDING_ID = 0
START_OF_SENTENCE_ID = 1
END_OF_SENTENCE_ID = 2

from data import get_training_dataset
from vocab import Vocab
from vocab import build_vocab