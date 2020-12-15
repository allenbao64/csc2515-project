import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model

np.random.seed(0)

train = pd.read_pickle('flattened_270K_train.pkl')
train = train[['book_title', 'sent_spoil', 'sentence']]

cv = pd.read_pickle('flattened_15K_cv.pkl')
cv = cv[['book_title', 'sent_spoil', 'sentence']]

test = pd.read_pickle('flattened_15K_test.pkl')
test = test[['book_title', 'sent_spoil', 'sentence']]

train.to_csv('train.csv')
cv.to_csv('cv.csv')
test.to_csv('test.csv')

train = pd.read_csv('train.csv')
cv = pd.read_csv('cv.csv')
train = pd.concat([train, cv], ignore_index=True)
test = pd.read_csv('test.csv')

train_text = (train['book_title'].map(str) + ' ~~~ ' + train['sentence'].map(str)).to_numpy()
train_labels = train['sent_spoil'].to_numpy().astype(np.int32)

test_text = (test['book_title'].map(str) + ' ~~~ ' + test['sentence'].map(str)).to_numpy()
test_labels = test['sent_spoil'].to_numpy().astype(np.int32)

'''
import json

with open('goodreads_reviews_spoiler.json') as f:
    lines = f.read().splitlines()
df = pd.DataFrame(lines)
df.columns = ['json_element']
df = df.sample(frac=0.2, random_state=1)
df['json_element'].apply(json.loads) #9.70
df = pd.json_normalize(df['json_element'].apply(json.loads)) #42.36
df.to_pickle('goodreads.pkl')
'''

df = pd.read_pickle('goodreads.pkl')
df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d")

'''
users = df.groupby('user_id').count()
users_mean = users['timestamp'].mean()
users_median = users['timestamp'].median()

books = df.groupby('book_id').count()
books_mean = books['timestamp'].mean()
books_median = books['timestamp'].median()

spoilers = df.spoiler.value_counts()

reviews = df.loc[df['spoiler'] == '0', ['text']]
sentences = reviews.text.str.len()
sentences_mean = sentences.mean()
sentences_median = sentences.median()
'''
df = df[['timestamp', 'review_sentences']]
reviews = df.explode('review_sentences')
sentences = reviews['review_sentences'].apply(lambda x: ','.join(map(str, x)))
reviews['review_sentences'] = sentences

reviews = reviews.sample(n=600000, random_state=1)
reviews['review_sentences'] = reviews['review_sentences'].str.split(',', n=1)

reviews.to_pickle('reviews.pkl')
reviews = pd.read_pickle('reviews.pkl')

'''
train = reviews[reviews['timestamp'] < '2015-01-01']
train = train['review_sentences']
test = reviews[reviews['timestamp'] >= '2015-01-01']
test = test['review_sentences']
test = test.sample(n=10000, random_state=1)
'''

'''
reviews = reviews['review_sentences']
train, test = train_test_split(reviews, test_size=0.02)

train_text = train.map(lambda x: x[1]).to_numpy()
train_labels = train.map(lambda x: x[0]).to_numpy().astype(np.int32)
test_text = test.map(lambda x: x[1]).to_numpy()
test_labels = test.map(lambda x: x[0]).to_numpy().astype(np.int32)
'''

reviewMaxLen = 600

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(train_text)
sequences = tokenizer.texts_to_sequences(train_text)
padded = pad_sequences(sequences, maxlen=reviewMaxLen)

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 32, input_length=reviewMaxLen) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(32, dropout=0.1, return_sequences=True))
model.add(LSTM(32, dropout=0.2))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.003)
model.compile(loss='binary_crossentropy',optimizer=opt, metrics=[tf.keras.metrics.AUC()])
model.summary()

model.fit(padded, train_labels, validation_split=0.0527, epochs=8, batch_size=64, verbose=1)

model.save('model')

model = load_model('model')

# Run model on test set to predict new reviews
predictions = []
i = 0
for text in test_text:
    if i % 100 == 0:
        print(i)
    i += 1
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=reviewMaxLen)
    prediction = model.predict(pad).item()
    predictions.append(prediction)
predictions = np.array(predictions)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_text, predictions)
auc_keras = auc(fpr_keras, tpr_keras)
