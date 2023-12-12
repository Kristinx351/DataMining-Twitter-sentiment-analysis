# DataMining-Twitter-sentiment-analysis

Operating environment:
```
> TensorFlow v2.7.0
> Python 3.8
> Cuda 11.2
> GPU: RTX 3080 (10.5GB) *1
> CPU: Xeon Gold 6142 CPU *6
> Memory: 45GB
```

## Part 1 Data Processing

The corresponding code is in `Data process.ipynb`.

The notebook includes code cells with embedded Markdown explanations for each section, along with some code comments.

To meet the requirements of the data preprocessing case study for this research (indicated in the respective modules), you need to run the corresponding cell blocks in the specified order. Failure to do so may result in errors or incorrect data formatting.

* Functionality:
  This `ipynb` notebook implements the following:

1. Data document processing (removing unnecessary information and dividing into positive and negative labels).

2. Emoticon processing (*Before removing emoticons, we need to remove URLs and numbers to avoid mistakenly identifying ":/" in "http:/" and "https:/" as emoticons).

3. Handling word contractions (*When studying word contractions, we use `wordninja.split`. After that, we don't need to use the original NLTK tokenizer `word_tokenize` anymore).

4. Normalizing and removing punctuation.

5. Tokenization.

6. Removing stop words (*The stop word corpus has been adjusted in this step. The presented code represents the final version).

7. Splitting the dataset into training and validation sets in a 9:1 ratio and analyzing data features (longest sentence, data size, data distribution).

8. Creating a mini beta dataset (10K) for initial model debugging, parameter tuning, and quick experiments.

## Part 2 Word Embedding

The corresponding code is in `LSTM_3baseline.ipynb`.

The notebook includes code cells with embedded Markdown explanations for each section, along with some code comments.

To meet the requirements of the POS (part-of-speech) case study for this research, you need to run the corresponding cell blocks in the specified order. Also, make sure to change the model input accordingly: without POS, the model input embedding variable should be `train_embed_weights`, and with POS, the model input embedding variable should be `train_imp_embed_weights`.

* Procedure:

1. Load the `gensim` word2vec model `word2vec_model`.
2. Compare the data vocabulary with the model's dictionary. For words found in the model's dictionary, we directly import the pre-trained embeddings. For words not found, we provide two options: 1) assign 0; 2) assign a random value (implemented by the function `get_word2vec_embeddings()`).
3. Optimize word embeddings with POS tags (you can try two libraries: SPACY or NLTK). If there are any issues related to version compatibility with Spacy on certain platforms, you can resolve them by executing the first code cell in this file. The POS dimensions are: 17 dimensions for Spacy and 20 dimensions for NLTK. We assign a POS tag to each word and generate the corresponding ONE-HOT vector, which is concatenated with the original embedding.

## Part 3 Model Implementation

The corresponding code is in `LSTM_3baseline.ipynb`.

The notebook includes code cells with embedded Markdown explanations for each section, along with some code comments.

* It contains the following sections:

1. Implementation of LSTM, BiLSTM, and BiLSTM-Attention models using `tensorflow`.
2. Each model module includes the following parts: model construction, model training, and model validation/testing.
