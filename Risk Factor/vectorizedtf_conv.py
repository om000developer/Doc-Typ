import re
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from sklearn import random_projection
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

import sklearn.metrics as sm

classes = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]

df = pd.read_csv('augmented_riskfactor_sentences.csv', encoding='latin_1')
df = df[pd.notnull(df['Sentence'])]

df = df.reset_index(drop=True); df = df.sample(frac=1)

def clean_text(text):
    text = text.lower()
    text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text)
    text = re.compile('[^0-9a-z #+_]').sub('', text)
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('english')))
    return text
df['Sentence'] = df['Sentence'].apply(clean_text)

X_core = df['Sentence'].values
midway = int(X_core.shape[0]/2)

######################

from sklearn.feature_extraction.text import HashingVectorizer

hashing_vect = HashingVectorizer(n_features=15000)

fitted_vect = hashing_vect.fit(X_core[:midway])
with open('hashing_fitted_vect.pickle', 'wb') as fin: pickle.dump(fitted_vect, fin)

X_hash = fitted_vect.transform(X_core[:midway]).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_hash, pd.get_dummies(df['Risk_Factor'][:midway]).values, test_size = 0.2, random_state = 42)

RandomProjection = random_projection.GaussianRandomProjection(n_components=4000)
X_train = RandomProjection.fit_transform(X_train)
X_test = RandomProjection.transform(X_test)
pickle.dump(RandomProjection, open("rp_hash.pickle", "wb"))

X_train_hash = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_hash = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_hash.shape[1], 1)))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("hash-vec_model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("hash-vec_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = model.fit(X_train_hash, y_train, validation_data=(X_test_hash, y_test), epochs=25, batch_size=2, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])

model = load_model("hash-vec_model.hdf5")

y_pred = model.predict(X_test_hash)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))

print(acc); print(kappa)

######################

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_features=15000)

fitted_vect = count_vect.fit(X_core[midway:])
with open('count_fitted_vect.pickle', 'wb') as fin: pickle.dump(fitted_vect, fin)

X_count = fitted_vect.transform(X_core[midway:]).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_count, pd.get_dummies(df['Risk_Factor'][midway:]).values, test_size = 0.2, random_state = 42)

RandomProjection = random_projection.GaussianRandomProjection(n_components=4000)
X_train = RandomProjection.fit_transform(X_train)
X_test = RandomProjection.transform(X_test)
pickle.dump(RandomProjection, open("rp_count.pickle", "wb"))

X_train_count = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_count = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

for i in range(3): model.layers[i].trainable = False
for i in range(3, 9): model.layers[i].trainable = True
ll = model.layers[8].output
ll = Dense(64)(ll)
ll = Dense(len(classes), activation="softmax")(ll)

new_model = Model(inputs=model.input, outputs=ll)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("count-vec_model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("count-vec_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = model.fit(X_train_count, y_train, validation_data=(X_test_count, y_test), epochs=25, batch_size=2, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])

model = load_model("count-vec_model.hdf5")

y_pred = model.predict(X_test_count)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))

print(acc); print(kappa)

######################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, max_features=15000)

fitted_vect = tfidf_vect.fit(X_core[midway:])
with open('tfidf_fitted_vect.pickle', 'wb') as fin: pickle.dump(fitted_vect, fin)

X_tfidf = fitted_vect.transform(X_core).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, pd.get_dummies(df['Risk_Factor']).values, test_size = 0.1, random_state = 42)

RandomProjection = random_projection.GaussianRandomProjection(n_components=4000)
X_train = RandomProjection.fit_transform(X_train)
X_test = RandomProjection.transform(X_test)
pickle.dump(RandomProjection, open("rp_tfidf.pickle", "wb"))

X_train_tfidf = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_tfidf = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

for i in range(3): model.layers[i].trainable = False
for i in range(3, 10): model.layers[i].trainable = True
ll = model.layers[9].output
ll = Dense(16)(ll)
ll = Dense(len(classes), activation="softmax")(ll)

new_model = Model(inputs=model.input, outputs=ll)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("tfidf-vec_model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("tfidf-vec_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = model.fit(X_train_tfidf, y_train, validation_data=(X_test_tfidf, y_test), epochs=25, batch_size=2, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])

model = load_model("tfidf-vec_model.hdf5")

y_pred = model.predict(X_test_tfidf)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))

print(acc); print(kappa)