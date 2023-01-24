# DataMining-Twitter-sentiment-analysis
运行环境：
```
> TensorFlow v2.7.0
> Python 3.8
> Cuda 11.2
> GPU：RTX 3080（10.5GB) *1
> CPU：Xeon Gold 6142 CPU *6
> 内存：45GB
```
## Part 1 数据处理

相应代码在 `Data process.ipynb` 中。

其中已有对于每个部分的code cell的markdown嵌入式说明功能，并且配有部分代码注释。

根据本研究数据预处理Case study的需要（相应模块已标注），运行相应功能的cell block（请严格按照指示顺序执行，否则会报错或者处理出错误的数据格式）。

* 功能说明：
  本ipynb中实现了：

1. 数据文档处理（删除不需要信息，进行正负标签划分）

2. 颜表情处理 (*在去除颜表情之前，我们需要先去除网址与数字，为了避免把网址中"http:/", "https:/"中的":/"误判为我们的表情)

3. 去除粘连词（*在研究粘连词分割时，调用了  `wordninja.split`  后，在下文就不用再调用原先的NLTK分词模型 `word_tokenize` 了 ）

4. 正则化去除标点

5. 分词

6. 去停用词（*此处后期进行了词库调整，代码呈现最终版）

7. 按照比例划分train:valid数据集（9：1）并统计数据特征（最长句，数据量，数据分布情况）

8. 创建迷你beta数据集（10K，用于初期模型调试与调参与快速实验）

   

## Part 2 词向量表示

相应代码在 `LSTM_3baseline.ipynb` 中。 

其中已有对于每个部分的code cell的markdown嵌入式说明功能，并且配有部分代码注释。

根据本研究Case study需要（POS），运行相应功能的cell block（请严格按照指示顺序执行，并更改模型输入：无POS时，模型输入embedding变量为 `train_embed_weights` ，有POS时，模型输入embedding变量为 `train_imp_embed_weights` ）。

* 流程说明： 

1.  载入 `gensim` 的word2vec模型   `word2vec_model` 
2.  对照数据词表，对于在模型词典中的，我们直接引入预训练完的embedding，未搜索到的我们提供两种方式：1.赋0； 2. 随机赋值。（由函数 `get_word2vec_embeddings()` 实现）
3.  词性标注优化词向量（可尝试两种库（SPACY OR NLTK)， Spacy可能会在某些平台出现版本不适配的问题，POS维度：17维；NLTK法若运行错误，请回到本文件第一个代码单元执行可解决错误，POS维度：20维）；我们给每一个词进行词性标注，并生成对应的ONE-HOT vector, 连接在原有embedding之后。

## Part 3 模型实现

相应代码在 `LSTM_3baseline.ipynb` 中。 

其中已有对于每个部分的code cell的markdown嵌入式说明功能，并且配有部分代码注释。

* 包含部分：

1. 模型包含LSTM, BiLSTM，BiLSTM-Attention三个模型的实现，以 `tensorflow` 搭建；
2. 每个模型模块包含：模型搭建、模型训练以及模型验证/测试部分。

