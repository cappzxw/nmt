## NMT 简单结构(学习使用)

## Data Pipeline
使用tensorflow dataset api 构建输入流; lookup api 构建词表映射关系, 构建过程中使用的大多数变量都是tensor值
```shell
data_helper
├── __init__.py
├── data.py
└── vocab.py
```
vocab.py 中主要使用 `tensorflow.contrib.lookup` api 接口, 其中 `index_table_from_file` 返回 `tf.int64` 的table映射, `num_oov_buckets` 设置\<unk>个数，lookup 自动建立\<unk>对应的id, 为table中的最后一位, vocabulary_file 中一般含有pad token, sos token 和 eos token, 不必含有unk token; `index_to_string_table_from_file` 则返回 ids2word 的 table, `default_value` 为 unk token
```python
lookup.index_table_from_file(
  vocabulary_file=self.vocabulary_file,
  vocab_size=self.size - self.num_oov_buckets,
  num_oov_buckets=self.num_oov_buckets)
```
使用 tensorflow api 构建映射table, 方便后续为 dataset 提供服务

data.py 使用 tf.data 的 dataset 相关 api 接口, 

### Ref:

[google nmt](https://github.com/tensorflow/nmt/tree/0be864257a76c151eef20ea689755f08bc1faf4e)

[opennmt-tf](https://github.com/OpenNMT/OpenNMT-tf)