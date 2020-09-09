
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, GlobalMaxPool1D, Dropout, Flatten, Bidirectional, LSTM
from keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS


def strip_stop_words(input_str):
    tokens = input_str.split()
    result = [i for i in tokens if not i in STOP_WORDS]

    return ' '.join(result)


def process_string(input_string):
    input_str = input_string

    input_str = input_str.lower()

    input_str = re.sub(r'\d+', '', input_str)

    input_str = input_str.replace('\'', ' \'')

    # first stip to get stopwords starting with '
    input_str = strip_stop_words(input_str)

    input_str = input_str.translate(str.maketrans('', '', string.punctuation))

    input_str = strip_stop_words(input_str)

    return input_str


train = pd.read_csv('train.csv', index_col=False)
test = pd.read_csv('test.csv', index_col=False)

train_10 = train[:10]
train_1k = train[:1000]

# mock
# train = train_1k
# test = test[:100]

print(train.sample(5))

print(test.sample(5))

train.toxic.value_counts(normalize=True).plot.bar(title='toxic')
# plt.show()
train.severe_toxic.value_counts(normalize=True).plot.bar(title='severe_toxic')
# plt.show()
train.obscene.value_counts(normalize=True).plot.bar(title='obscene')
# plt.show()
train.threat.value_counts(normalize=True).plot.bar(title='threat')
# plt.show()
train.insult.value_counts(normalize=True).plot.bar(title='inslut')
# plt.show()
train.identity_hate.value_counts(normalize=True).plot.bar(title='identity_hate')
# plt.show()

print(train.sample(5))

num_words = 2000
max_len = 200
tokenizer = Tokenizer(num_words)

lst = [process_string(comment) for comment in train.comment_text]
processed_comments = np.asarray(lst)

tokenizer.fit_on_texts(processed_comments)

train_sequences = tokenizer.texts_to_sequences(processed_comments)
padded_train = pad_sequences(train_sequences, maxlen=max_len)

test_sequences = tokenizer.texts_to_sequences(test.comment_text)
padded_test = pad_sequences(test_sequences, maxlen=max_len)


y = train.iloc[:,2:].values

train_sequences[:1]

# copy paste; Had no time to play with different networks
model = Sequential([Embedding(num_words, 32, input_length=max_len),
                   Bidirectional(LSTM(32, return_sequences=True)),
                   GlobalMaxPool1D(),
                   Dense(32,activation='relu'),
                   Dense(6,activation='sigmoid')
                   ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 20
epoch = 1 # put 2 later when you have time
history = model.fit(padded_train, y, batch_size, epochs=epoch, validation_split=.25,
                    callbacks=[early_stopping_callback])


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])


test_ids = test.id
predicted = model.predict(padded_test)

print(predicted)

# cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
test['toxic'] = predicted[:, :1]
test['severe_toxic'] = predicted[:, 1:2]
test['obscene'] = predicted[:, 2:3]
test['threat'] = predicted[:, 3:4]
test['insult'] = predicted[:, 4:5]
test['identity_hate'] = predicted[:, 5:6]

print(test)

# sample_sub=pd.read_csv('sample_submission.csv', index_col=False)
# print("samples")
# print(sample_sub.sample(5))

# test.drop(['comment_text'], axis=1, inplace=True)

# print(test)


# test.to_csv('toxic_comments_classification.csv', index=False)


num_dota_chats = 100_000

dota_chats = pd.read_csv("dota2_chat_messages.csv", nrows=num_dota_chats)

dota_chats.drop(['match', 'time', 'slot'], axis=1, inplace=True)

dota_chats['text'] = dota_chats['text'].astype(str)

lst = [process_string(comment) for comment in dota_chats.text]
processed_dota_comments = np.asarray(lst)
dota_sequences = tokenizer.texts_to_sequences(processed_dota_comments)
padded_dota_comments = pad_sequences(dota_sequences, maxlen=max_len)
predicted_dota = model.predict(padded_dota_comments)

dota_chats['toxic'] = predicted_dota[:, :1]
dota_chats['severe_toxic'] = predicted_dota[:, 1:2]
dota_chats['obscene'] = predicted_dota[:, 2:3]
dota_chats['threat'] = predicted_dota[:, 3:4]
dota_chats['insult'] = predicted_dota[:, 4:5]
dota_chats['identity_hate'] = predicted_dota[:, 5:6]

dota_chats.to_csv('dota_classification.csv', index=False)