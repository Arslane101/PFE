import numpy as np
import pandas as pd 
import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_hub as hub
import bert
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import re
from sklearn.model_selection import train_test_split
import math

def clean_text(text):
    '''
    Function to clean text and remove unnecessary components and ease tokenization.
    :param: text: the string to be cleaned
    :output: text: the cleaned string
    '''
    text = re.sub(r'https?://\S+', '', text) # remove link
    text = re.sub(r'#\w+', '', text) # remove hashtags
    text = re.sub(r'@\w+', '', text) # remove mentions
    text = re.sub(r'\n', ' ', text) # remove linebreaks
    text = re.sub(r'\s+', ' ', text) # remove leading and trailing spaces
    return text
def bert_encoding(texts, tokenizer, max_len=512):
    all_tokens = [] # initiated list for tokens
    all_masks = [] # initiated list for masks
    all_segments = [] # initiated list for segment_ids
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
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
    _,sequence_output = bert_layer([input_word_ids,input_mask,input_type_ids])
    clf_output = sequence_output
    
    # CHANNEL 1 - LSTM-RNN
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,activation='sigmoid',return_sequences=True))(clf_output)
    drop = tf.keras.layers.Dropout(0.3)(lstm)
    rnn = tf.keras.layers.SimpleRNN(128)(drop)
    flat = tf.keras.layers.Flatten()(rnn)
    dense = tf.keras.layers.Dense(128,activation='relu')(flat)
    
    """ CHANNEL 2 - RNN-LSTM
    rnn = tf.keras.layers.SimpleRNN(128,return_sequences=True)(clf_output)
    drop = tf.keras.layers.Dropout(0.3)(rnn)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,activation='sigmoid'))(drop)
    flat = tf.keras.layers.Flatten()(lstm)
    dense = tf.keras.layers.Dense(128,activation='relu')(flat)"""
    
    
    #OUTPUT
    output_target = tf.keras.layers.Dense(5,activation='softmax')(dense)
    
    return tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],outputs=output_target)
def PredictSentiment(rating,comment,alpha):
    texts = list()   
    texts.append(comment)
    text = bert_encoding(texts, tokenizer, max_len=100)
    results = model.predict(text)
    results = np.argsort(results.reshape(5))[::-1] 
    if(math.isnan(rating)):
        return results[0]
    else: rating = int(alpha*float(int(results[0])+1) + float(100)*(1-alpha)*float(rating))
    return rating 
df_amazon = pd.read_json('reviews_Video_Games_5.json', lines=True)    
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

df_amazon['reviewText'] = df_amazon['reviewText'].apply(lambda x: clean_text(x))
df_amazon['reviewText'] = df_amazon['reviewText'].apply(lambda x: x.lower())
df_amazon['overall']=df_amazon['overall'].apply(lambda x: x-1)

X_train,X_test,Y_train,Y_test = train_test_split(df_amazon['reviewText'],targets,test_size=0.3,random_state=28)
train_input = bert_encoding(X_train, tokenizer, max_len=100)
test_input = bert_encoding(X_test, tokenizer, max_len=100)

model = build_model(bert_layer,max_len=100)
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
train_history = model.fit(train_input,Y_train,epochs=10,batch_size=250,verbose=1,validation_split=0.1)
model.save("SentimentModel")


#Exploitation du modèle
ratings = pd.read_csv("normalizedreviews.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
ratings['review_content'] = ratings['review_content'].apply(lambda x: utils_preprocess_text(x))

ratings['review_content'] =ratings['review_content'].apply(lambda x: x.lower())

for i in range(ratings.shape[0]):
    if(pd.isnull(ratings['review_content'][i])):
        ratings.loc[i,'rating'] = ratings.loc[i,'rating']
    else:
        ratings.loc[i,'rating'] = PredictSentiment(ratings['rating'][i],ratings['review_content'][i],0.2)