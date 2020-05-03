import numpy as np
import sklearn.metrics as sm
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D

np.random.seed(7)

"""
text, label_text = [], []

for i in json.loads(open("augmented_diagnosis_cases.json").read()): 
    if i['Diagnosis'] != "" and i['Diagnosis'] != "Diagnosis": 
        text.append(i['Case']);
        label_text.append(i['Diagnosis'].split(", ")[0].replace(" ", "_").lower())

import os, json, pytextrank, networkx as nx
from PIL import Image

for t in range(len(text)):
    
    path_stage0 = "o0.json"; path_stage1 = "o1.json"
    
    file_dic = {"id" : 0, "text" : text[t]}
    loaded_file_dic = json.loads(json.dumps(file_dic))
    
    with open(path_stage0, 'w') as outfile: json.dump(loaded_file_dic, outfile)
    
    with open(path_stage1, 'w') as f:
        for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
            f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
            print(pytextrank.pretty_print(graf._asdict()))
    
    graph, ranks = pytextrank.text_rank(path_stage1)
    pytextrank.render_ranks(graph, ranks)
    
    nx.draw(graph, with_labels=True);
    plt.savefig("txtranks/png/" + label_text[t] + '_' + str(t) + ".png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    im = Image.open("txtranks/png/" + label_text[t] + '_' + str(t) + ".png").convert('L').resize((300, 200))
    np.save('txtranks/np/' + label_text[t] + '_' + str(t) + '.npy', np.array(im))
    
classes_dict = {'major_depressive_disorder': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_deficit_hyperactivity_disorder': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'oppositional_defiant_disorder': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'conduct_disorder': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'pervasive_developmental_disorders': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'intellectual_disability_(mental_retardation)': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 'psychotic_disorder': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 'adjustment_disorder': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 'mood_disorder': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 'general_anxiety_disorder': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 'social_anxiety_disorder': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 'seasonal_affective_disorder': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 'substance_abuse': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 'autism': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

labels, numpies = [], []
for np_path in os.listdir("txtranks/np/"):
    try:
        labels.append(classes_dict["_".join(np_path.split("_")[:-1])])
        numpies.append(np.load('txtranks/np/' + np_path))
    except: pass
    
np.save('txtranks/np/X.npy', np.array(numpies).astype(np.float))
np.save('txtranks/np/y.npy', np.array(labels).astype(np.float))
"""

X_train, X_test, y_train, y_test = train_test_split(np.load('txtranks/np/X.npy'), np.load('txtranks/np/y.npy'), test_size = 0.10, random_state = 42)   

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) / np.max(X_train)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1) / np.max(X_test)

train_X, valid_X, train_ground, valid_ground = train_test_split(X_train, X_train, test_size=0.2, random_state=13)

input_img = Input(shape = (200, 300, 1))

def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):   
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer='rmsprop')
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=1, epochs=35, verbose=1, validation_data=(valid_X, valid_ground))

plt.figure()
plt.plot(range(35), autoencoder_train.history['loss'], 'bo', label='Training loss')
plt.plot(range(35), autoencoder_train.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss'); plt.legend(); plt.show()

autoencoder.save_weights('autoencoder.h5')

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(y_train.shape[1], activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img, fc(encode))

for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[0:19]): l1.set_weights(l2.get_weights())

for layer in full_model.layers[0:19]: layer.trainable = False

full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("graph_conv_autoencoder.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("graph_conv_autoencoder_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = full_model.fit(X_train, y_train, batch_size=1, epochs=35, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])

model = keras.models.load_model("graph_conv_autoencoder.hdf5")

y_pred = model.predict(X_test)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))

print(acc); print(kappa)