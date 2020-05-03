import re
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from classification_outputs import output_derivations
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Input, Flatten, SpatialDropout1D, TimeDistributed, GRU, LSTM

classes = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]

df = pd.read_csv('augmented_riskfactor_sentences.csv', encoding='latin_1')
df = df[pd.notnull(df['Sentence'])]
df.Risk_Factor.value_counts()

df = df.reset_index(drop=True)
df = df.sample(frac=1)

def clean_text(text):
    text = text.lower()
    text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text)
    text = re.compile('[^0-9a-z #+_]').sub('', text)
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('english')))
    return text

df['Sentence'] = df['Sentence'].apply(clean_text)
df['Sentence'] = df['Sentence'].str.replace('\d+', '')

text = list(df['Sentence'])
labels = pd.get_dummies(df['Risk_Factor']).values

t = Tokenizer(); t.fit_on_texts(text)

max_length = 375

text = pad_sequences(sequences=t.texts_to_sequences(text), maxlen=max_length)

word_index = t.word_index
print('Found %s unique tokens.' % len(word_index))

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.25)

embeddings_index = {}

f = open("glove.6B/glove.6B.100d.txt", encoding="utf8")
for line in f: embeddings_index[line.split()[0]] = np.asarray(line.split()[1:], dtype='float32')
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
       
embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=max_length, trainable=False) # Setting trainable=False to prevent the weights from being updated during training
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(embedded_sequences)
x = LSTM(units=75, dropout=0.2, recurrent_dropout=0.2, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(x)
x = GRU(50, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
x = TimeDistributed(Dense(10))(x)
x = SpatialDropout1D(rate=0.2)(x)
x = Flatten()(x)
x = Dense(50)(x)
x = Dense(50)(x)
x = Dense(50)(x)
preds = Dense(len(classes), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("embeddings_lstm_model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("embeddings_lstm_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=1, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])
  
model = load_model("embeddings_lstm_model.hdf5")

y_pred = model.predict(X_test)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

output_derivations(predictions, actuals, y_pred, y_test, classes, "embeddings_lstm")