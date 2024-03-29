import tensorflow as tf
from vocab import Vocab
from tensorflow.contrib.framework import nest
from data_helper import START_OF_SENTENCE_ID, END_OF_SENTENCE_ID

def make_features(element=None, features=None, vocabulary=None):
  if vocabulary is None:
    raise ValueError('vocabulary can not be None')
  if features is None:
    features = {}
  tokens = tf.string_split([element]).values
  ids = tf.cast(vocabulary.lookup(tokens), tf.int32)
  features["length"] = tf.size(tokens)
  features["tokens"] = tokens
  features["ids"] = ids
  return features

def make_labels(element=None, labels=None, vocabulary=None):
  if vocabulary is None:
    raise ValueError('vocabulary can not be None')
  if labels is None:
    labels = {}
  tokens = tf.string_split([element]).values
  labels["tokens"] = tokens
  ids = tf.cast(vocabulary.lookup(tokens), tf.int32)
  bos = tf.constant([START_OF_SENTENCE_ID], dtype=ids.dtype)
  eos = tf.constant([END_OF_SENTENCE_ID], dtype=ids.dtype)
  labels["ids"] = tf.concat([bos, ids], axis=0)
  labels["ids_out"] = tf.concat([ids, eos], axis=0)
  labels["length"] = tf.shape(labels["ids"])[0]
  return labels
  

def filter_length(maximum_features_length, maximum_labels_length, intercept=False):
  def _predicate(features, labels):
    features_length = features["length"]
    labels_length = labels["length"]
    features_len_ok = tf.logical_and(
        tf.greater(features_length, 1), tf.less_equal(features_length, maximum_features_length))
    labels_len_ok = tf.logical_and(
        tf.greater(labels_length, 1), tf.less_equal(labels_length, maximum_labels_length))
    return tf.logical_and(features_len_ok, labels_len_ok)
  def _intercept(features, labels):
    features_length = features["length"]
    labels_length = labels["length"]
    def _features_intercept():
      features["ids"] = features["ids"][:maximum_features_length]
      features["length"] = maximum_features_length
      features["tokens"] = features["tokens"][:maximum_features_length]
      return features
    def _labels_intercept():
      labels["ids"] = labels["ids"][:maximum_labels_length]
      labels["length"] = maximum_labels_length
      labels["tokens"] = labels["tokens"][:maximum_labels_length]
      labels["ids_out"] = tf.concat([labels["ids_out"][:maximum_labels_length-1], 
                                    [labels["ids_out"][-1]]], axis=0)
      return labels
    
    features = tf.cond(tf.less(features_length, maximum_features_length), 
                       lambda: features, _features_intercept)
    labels = tf.cond(tf.less(labels_length, maximum_labels_length),
                       lambda : labels, _labels_intercept)
    
    return (features, labels)
  if not intercept:
    return lambda dataset : dataset.filter(_predicate)
  else:
    return lambda dataset : dataset.map(_intercept)

def batch_pad_dataset(batch_size, padded_shapes, batch_type="examples", bucket_width=None):
  def _key_func(features, labels):
    features_length = features["length"]
    labels_length = labels["length"]
    bucket_id = tf.constant(0, dtype=tf.int32)
    if features_length is not None:
      bucket_id = tf.maximum(bucket_id, features_length // bucket_width)
    if labels_length is not None:
      bucket_id = tf.maximum(bucket_id, labels_length // bucket_width)
    return tf.cast(bucket_id, tf.int64)

  def _reduce_func(unused_key, dataset):
    return dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

  def _window_size_func(key):
    # if bucket_width > 1:
    #   key += 1
    size = batch_size // (key * bucket_width)
    return tf.cast(size, tf.int64)

  if bucket_width is None:
    return lambda dataset : dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
  if batch_type == "examples":
    return tf.contrib.data.group_by_window(_key_func, _reduce_func, window_size=batch_size)
  if batch_type == "tokens":
    return tf.contrib.data.group_by_window(_key_func, _reduce_func, window_size_func=_window_size_func)
  else:
    raise ValueError(
        "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))

def get_training_dataset(features_file,
                         labels_file,
                         features_vocab_file,
                         labels_vocab_file,
                         batch_size,
                         batch_type="examples",
                         share_vocab=False,
                         intercept=False,
                         shuffle_buffer_size=None,
                         bucket_width=None,
                         maximum_features_length=None,
                         maximum_labels_length=None,
                         single_pass=False):

                      
  features_dataset = tf.data.TextLineDataset(features_file)
  features_vocab = Vocab(vocabulary_file=features_vocab_file)
  features_vocab = features_vocab.vocabulary_lookup()
  features_dataset = features_dataset.map(
    lambda args: make_features(args, vocabulary=features_vocab))

  labels_dataset = tf.data.TextLineDataset(labels_file)
  if share_vocab is not None:
    labels_vocab = features_vocab
  else:
    labels_vocab = Vocab(vocabulary_file=labels_vocab_file)
    labels_vocab = labels_vocab.vocabulary_lookup()
  labels_dataset = labels_dataset.map(
    lambda args: make_labels(args, vocabulary=labels_vocab))


  dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

  if shuffle_buffer_size is not None:
    dataset = dataset.shuffle(shuffle_buffer_size)

  dataset = dataset.apply(filter_length(maximum_features_length=maximum_features_length,
                                        maximum_labels_length=maximum_labels_length,
                                        intercept=intercept))
  padded_shapes = nest.map_structure(lambda shape: shape.as_list(), dataset.output_shapes)

  dataset = dataset.apply(
    batch_pad_dataset(batch_size=batch_size, 
                      padded_shapes=padded_shapes,
                      batch_type=batch_type,
                      bucket_width=bucket_width))
  if not single_pass:
    dataset = dataset.repeat()

  return dataset