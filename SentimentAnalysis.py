from matplotlib import pyplot
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim.downloader as api
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, Bidirectional, CuDNNLSTM, GRU, BatchNormalization, SimpleRNN
from keras import backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def rmse(y_real, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_real), axis=-1))
ratings = pd.read_csv("copy.csv",delimiter=";")   
ratings = ratings[ratings["review_score"]==float(5000)]
#Lowercase and Punctuation Recmoval
ratings['review_content'] = ratings["review_content"].str.lower()
ratings['review_content'] = ratings['review_content'].str.replace('[^\w\s]','')
#Remove Stopwords
stops = set(stopwords.words('english'))
ratings['review_content'] = ratings['review_content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
#Stemming
stemmer = nltk.stem.SnowballStemmer("english")
ratings['review_content'] = ratings['review_content'].apply(lambda x: [stemmer.stem(y) for y in x.split()])
ratings['review_content'] = ratings['review_content'].str.join(" ")
#Word2Vec
vocabulary_size = 20000
### Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(ratings['review_content'])
sequences = tokenizer.texts_to_sequences(ratings['review_content'])
data = pad_sequences(sequences, maxlen=50)
encoder = LabelEncoder()
encoder.fit(ratings['review_score'])
encoded_Y = encoder.transform(ratings['review_score'])
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X_train_embedding, X_test_embedding, y_train_embedding, y_test_embedding = train_test_split(data, dummy_y, test_size=0.2, stratify=dummy_y, random_state=42)
w2vec_gensim  = api.load('glove-wiki-gigaword-100')
vector_size = 100
embedding_matrix_w2v = np.zeros((vocabulary_size ,vector_size))
for word, index in tokenizer.word_index.items():
    if index < vocabulary_size: # since index starts with zero 
        if word in w2vec_gensim.wv.vocab:
            embedding_matrix_w2v[index] = w2vec_gensim[word]
        else:
            embedding_matrix_w2v[index] = np.zeros(100)
## create model
epoch = 20
batch = 128
predicted_rating = 0.75
real_rating = 1 - predicted_rating

model_w2v_rnn = Sequential()
model_w2v_rnn.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix_w2v], trainable=False))
model_w2v_rnn.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_w2v_rnn.add(Dense(5, activation='softmax'))
model_w2v_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[rmse,'mae'])
history_model_w2v_rnn = model_w2v_rnn.fit(X_train_embedding, y_train_embedding, validation_data=(X_test_embedding, y_test_embedding), epochs = epoch + 1, batch_size=batch)

pyplot.plot(history_model_w2v_rnn.history['rmse'], label='training rmse')
pyplot.plot(history_model_w2v_rnn.history['val_rmse'], label='validation rmse')
pyplot.plot(history_model_w2v_rnn.history['mae'], label='training mae')
pyplot.plot(history_model_w2v_rnn.history['val_mae'], label='validation mae')
pyplot.legend()
pyplot.xlabel('EPOCH')
pyplot.show()