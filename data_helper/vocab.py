import tensorflow as tf
import tensorflow.contrib.lookup as lookup
import codecs
import collections
from operator import itemgetter

from data_helper import UNKNOWN_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN

class Vocab(object):
  def __init__(self,
               vocabulary_file=None,
               vocab_size=None,
               num_oov_buckets=1):
    self.vocabulary_file = vocabulary_file
    self.vocab_size = vocab_size
    self.num_oov_buckets = num_oov_buckets

  def vocabulary_lookup(self):
    # build a lookup table for word2ids, <unk> token is the last one in the table
    return lookup.index_table_from_file(
      vocabulary_file=self.vocabulary_file,
      vocab_size=self.size - self.num_oov_buckets,
      num_oov_buckets=self.num_oov_buckets)

  def vocabulary_lookup_reverse(self):
    # build a table for ids2word
    return lookup.index_to_string_table_from_file(
      vocabulary_file=self.vocabulary_file,
      vocab_size=self.size - self.num_oov_buckets,
      default_value=UNKNOWN_TOKEN)
  
  @property
  def size(self):
    if self.vocab_size is not None:
      return self.vocab_size
    else:
      with codecs.open(self.vocabulary_file, 'r', 'utf-8') as f:
        i = 0
        for i, _ in enumerate(f):
          pass
      return i + 1 + self.num_oov_buckets


def build_vocab(file_path_list, output_path, vocab_size=None):
  
  # create a counter with (word, count) pairs
  counter = collections.Counter()
  for file_path in file_path_list:
    with codecs.open(file_path, 'r', 'utf-8') as f:
      for line in f:
        for word in line.strip().split():
          if word != '':
            counter[word] += 1

  # sort the count with the count
  sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
  sorted_words = [x[0] for x in sorted_word_to_cnt]
  sorted_words = [PADDING_TOKEN, START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN] + sorted_words
  if vocab_size is not None:
    if len(sorted_words) > vocab_size:
      sorted_words = sorted_words[:size]

  with codecs.open(output_path, 'w', 'utf-8') as f:
    for word in sorted_words:
      f.write(word + '\n')
