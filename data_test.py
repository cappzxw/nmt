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
                                 2,
                                 bucket_width=5,
                                 share_vocab=False,
                                 maximum_features_length=50,
                                 maximum_labels_length=20)

  iterator = dataset.make_initializable_iterator()
  source, target = iterator.get_next()


  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    s, t = sess.run([source, target])
    print s["tokens"]
    print t["tokens"]

if __name__ == "__main__":
  # test_build_vocab()
  test_get_training_dataset()