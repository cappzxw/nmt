import tensorflow as tf
from data_helper import build_vocab
from data_helper import get_training_dataset

def test_build_vocab():
  file_path_1 = "./data/source.txt"
  file_path_2 = "./data/target.txt"
  output_path = "./data/vocab.txt"
  build_vocab(file_path_list=[file_path_1, file_path_2],
              output_path=output_path)

def test_get_training_dataset():
  dataset = get_training_dataset('./data/source.txt',
                                 './data/target.txt',
                                 './data/vocab.txt',
                                 './data/vocab.txt',
                                 batch_size=20,
                                 batch_type="tokens",
                                 bucket_width=1,
                                 intercept=True,
                                 share_vocab=True,
                                 maximum_features_length=5,
                                 maximum_labels_length=2)

  iterator = dataset.make_initializable_iterator()
  source, target = iterator.get_next()


  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    try:
      s, t = sess.run([source, target])
      print s["length"]
      print t["length"]
    except tf.errors.OutOfRangeError:
      print "end of the dataset"

if __name__ == "__main__":
  # test_build_vocab()
  test_get_training_dataset()