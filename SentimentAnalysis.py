import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

import tensorflow as tf
import bert
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk.corpus
from keras.layers import Input,Dropout,Dense,Reshape,Conv1D,Bidirectional,LSTM,MaxPooling1D,Flatten,Concatenate
from keras import Model
from keras.models import Sequential
from keras.regularizers import L2
from sklearn.preprocessing import LabelBinarizer
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from keras.utils import plot_model
# Use stopwords list from nltk
lst_stopwords = nltk.corpus.stopwords.words("english")
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    # Clean (convert to lowercase and remove punctuations and characters and then strip)
    # The function is not optimized for speed but split into various steps for pedagogical purpose
    
    text = str(text).lower()
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize (convert from string to list)
    lst_text = text.split()
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)
    return text
def bert_encoding(texts, tokenizer, max_len=512):
    '''
    Function to encode text into tokens, masks, and segment_ids for BERT embedding layer input.
    
    :param: texts - the texts to tokenize
    :param: tokenizer - the BERT tokenizer that will be used to tokenize the texts
    :param: max_len - the maximum length of an input sequence (the sequence of tokens to be embedded)
    
    :output: all_tokens - the texts turned into tokens and padded for match length, returned as np.array
    :output: all_masks - masks for each text denoted sequence length and pad length, returned as np.array
    :output: all_segments - segment_ids for each text, all blank, returned as np.array
    '''
    all_tokens = [] # initiated list for tokens
    all_masks = [] # initiated list for masks
    all_segments = [] # initiated list for segment_ids
    
    # Iterate through all texts
    for text in texts:
        
        # Tokenize text
        text = tokenizer.tokenize(text)
        
        # Make room for the CLS and SEP tokens
        text = text[:max_len-2]
        
        # Create the input sequence beginning with [CLS] and ending with [SEP]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        
        # Determine how much padding is required (max_length - length of the input sequence)
        pad_len = max_len - len(input_sequence)
        
        # Create token ids, used by BERT
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        
        # Pad the tokens by 0's for the pad length determined above
        tokens += [0] * pad_len
        
        # Create the masks for the sequence, with the 1 for each token id and 0 for all padding
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        
        # All empty segment_ids for the max length
        segment_ids = [0] * max_len
        
        # Append all tokens, masks, and segment_ids to the initialized lists
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):
    # INPUTS
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_type_ids')
    
    # BERT EMBEDDING
    _, sequence_output = bert_layer([input_word_ids,input_mask,input_type_ids])
    clf_output = tf.keras.layers.Reshape((32,24))(sequence_output[:,0,:])
    
    
    # CHANNEL 2 - LSTM
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200,
                                                             return_sequences=True))(clf_output)
    flat = tf.keras.layers.Flatten()(lstm)

    drop = tf.keras.layers.Dropout(0.2)(flat)
    
    dense_1 = tf.keras.layers.Dense(100,activation='relu')(drop)
    output = tf.keras.layers.Dense(5,activation='softmax')(dense_1)

    
    return tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],outputs=[output])




ratings = pd.read_csv("normalizedsentimentratings.csv",delimiter=";")   
ratings = ratings[ratings["review_score"]!=int(4000)]
ratings_1 = ratings[ratings["review_score"]==int(1)]
ratings_1 = ratings_1.sample(2000)
ratings_2 = ratings[ratings["review_score"]==int(2)]
ratings_2 = ratings_2.sample(2000)
ratings_3 = ratings[ratings["review_score"]==int(3)]
ratings_3 = ratings_3.sample(2000)
ratings_4 = ratings[ratings["review_score"]==int(4)]
ratings_4 = ratings_4.sample(2000)
ratings_5 = ratings[ratings["review_score"]==int(5)]
ratings_5 = ratings_5.sample(2000)
balanced_ratings = pd.concat([ratings_1,ratings_2,ratings_3,ratings_4,ratings_5])
balanced_ratings = balanced_ratings.sample(frac=1)
balanced_ratings = balanced_ratings.reset_index()
balanced_ratings['review_content'] = balanced_ratings["review_content"].apply(lambda x:
utils_preprocess_text(x,flg_stemm=False,flg_lemm=True,lst_stopwords=lst_stopwords))
balanced_ratings['review_content']= balanced_ratings["review_content"].apply(
lambda x: x.lower())
balanced_ratings['review_score']=balanced_ratings["review_score"].apply(lambda x: x-1)
targets = balanced_ratings["review_score"]
BertTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

# Load the tokenizer with the preloaded vocab file and lower case function
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
X_train,X_test,Y_train,Y_test = train_test_split(list(balanced_ratings['review_content']),targets,test_size=0.2,random_state=28)

train_input = bert_encoding(X_train, tokenizer, max_len=100)
test_input = bert_encoding(X_test, tokenizer, max_len=100)

model = build_model(bert_layer,max_len=100)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
train_history = model.fit(train_input,Y_train,epochs=10,batch_size=128,validation_split=0.2)
