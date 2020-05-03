import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sklearn.metrics as sm
from datetime import datetime
from keras.engine import Layer
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input, Lambda
from sklearn.model_selection import train_test_split
from classification_outputs import output_derivations
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

np.random.seed(7)

text, labels = [], []

classes_dict = {"Major Depressive Disorder": 0, "Attention Deficit Hyperactivity Disorder": 1, "Oppositional Defiant Disorder": 2, "Conduct Disorder": 3, "Pervasive Developmental Disorders": 4, "Intellectual Disability (Mental Retardation)": 5, "Psychotic Disorder": 6, "Adjustment Disorder": 7, "Mood Disorder": 8, "General Anxiety Disorder": 9, "Social Anxiety Disorder": 10, "Seasonal Affective Disorder": 11, "Substance Abuse": 12, "Autism": 13}

classes = list(classes_dict)

"""
lst1, lst2 = [], []
for i in classes_dict:
    lst1.append(i.replace(" ", "_").lower())
    label = [0 for x in range(len(classes))]
    label[classes_dict[i]] = 1
    lst2.append(label)

res = {} 
for key in lst1: 
    for value in lst2: 
        res[key] = value 
        lst2.remove(value) 
        break  
"""

for i in json.loads(open("augmented_diagnosis_cases.json").read()): 
    
    if i['Diagnosis'] != "" and i['Diagnosis'] != "Diagnosis": 
        
        text.append(i['Case']);
        
        label = [0 for x in range(len(classes))]
        for y in i['Diagnosis'].split(", "): label[classes_dict[y]] = 1
        labels.append(label)
    
X_train, X_test, y_train, y_test = train_test_split(np.array(text, dtype=object)[:, np.newaxis], np.array(labels), test_size=0.25)

class ElmoEmbeddingLayer(Layer):
    
    def __init__(self, **kwargs): super(ElmoEmbeddingLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True, name="{}_module".format(self.name))
        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)
    def call(self, x, mask=None): return self.elmo(K.squeeze(K.cast(x, tf.string), axis=1), as_dict=True, signature='default')['default']
    def compute_mask(self, inputs, mask=None): return K.not_equal(inputs, '--PAD--')
    def compute_output_shape(self, input_shape): return (input_shape[0], 1024)

layer_one = Lambda(lambda x: x ** 2)

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    return K.concatenate([K.relu(x), K.relu(-x)], axis=1)
layer_two = Lambda(antirectifier)

def linear_transform(x):
  v1 = tf.Variable(1., name='multiplier')
  v2 = tf.Variable(0., name='bias')
  return x*v1 + v2
layer_three = Lambda(linear_transform)

def noise(x):
    mu, sigma1, sigma2, sigma3 = 0, 0.1, 0.15, 0.2
    noise1 = np.random.normal(mu, sigma1, x.shape) 
    noise2 = np.random.normal(mu, sigma2, x.shape) 
    noise3 = np.random.normal(mu, sigma3, x.shape) 
    return noise1 + noise2 + noise3
layer_four = Lambda(lambda x: x + noise(x))

layer_five = Lambda(lambda x: -x)

layers = [layer_one, layer_two, layer_three, layer_four, layer_five]
layers_paired = []
for i in range(len(layers)):
    for j in range(len(layers)): 
        layers_paired.append([layers[i], layers[j]])

def nn(layers_paired, epochs):
    
    for pair in range(len(layers_paired)):
            
        def build_model(): 
          
          input_text = Input(shape=(1,), dtype="string")
          embedding = ElmoEmbeddingLayer()(input_text)
          
          custom_lambda1 = layers_paired[pair][0](embedding)
          dense = Dense(256, activation='relu')(custom_lambda1)
          custom_lambda2 = layers_paired[pair][1](dense)
          pred = Dense(len(classes), activation='softmax')(custom_lambda2)
        
          model = Model(inputs=[input_text], outputs=pred)
        
          model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
          model.summary()
          
          return model
        
        model = build_model()
        
        if len(layers_paired) == 1: 
            model_name = "elmo_ann.hdf5"
            model_history_name = "elmo_ann_history.csv"
        else:
            model_name = "elmo_ann_models/elmo_ann_" + str(pair) + ".hdf5"
            model_history_name = "elmo_ann_models/elmo_ann_history_" + str(pair) + ".csv"
        
        checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
        csv_logger = CSVLogger(model_history_name, separator=',', append=False)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.0001)
        tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=1, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])
        
        model = load_model(model_name)
        
        y_pred = model.predict(X_test)
        predictions, actuals = [], []
        for i in range(len(y_pred)): 
            predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
            actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])
            
        open("elmo_ann_models/accuracies.txt", "a").write("\nPair " + str(pair) + ": " + str(sm.accuracy_score(predictions, actuals)) + "")
    
"""
nn(layers_paired, 1)

cnts = open("elmo_ann_models/accuracies.txt", "r").read().split(" ")
max_no = 0
for x in cnts:
    try: max_no = int(x) if int(x) > max_no else max_no
    except: pass
optimal_pair = int(cnts[cnts.index(str(max_no)) - 1].replace(":", ""))

with open("lambda_output.txt", "w") as fp: fp.write(str(optimal_pair))

"""

optimal_pair = int(open("lambda_output.txt", "r").read())
nn([layers_paired[optimal_pair]], 5)

model = load_model("elmo_ann.hdf5")

y_pred = model.predict(X_test)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])
            
output_derivations(predictions, actuals, y_pred, y_test, classes, "elmo_ann")