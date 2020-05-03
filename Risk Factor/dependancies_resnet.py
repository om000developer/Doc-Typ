import numpy as np
import pandas as pd
import sklearn.metrics as sm
from datetime import datetime
from keras import applications
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

classes = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]

df = pd.read_csv('augmented_riskfactor_sentences.csv', encoding='latin_1')
df = df[pd.notnull(df['Sentence'])]
df.Risk_Factor.value_counts()

df = df.reset_index(drop=True)
df = df.sample(frac=1)

"""
import en_core_web_sm 
from PIL import Image
from pathlib import Path
from spacy import displacy
nlp = en_core_web_sm.load()
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

for i in range(len(df['Sentence'])):
    lbl = df['Risk_Factor'][i]
    if str(lbl) != "nan": lbl = df['Risk_Factor'][i].replace("/", "___").replace(" ", "_")
    else: lbl = "nan"
    output_path = Path("dependancies/svg/" + lbl + '_' + str(i) + ".svg")
    svg = displacy.render(nlp(df['Sentence'][i]), style="dep", jupyter=False)
    output_path.open("w", encoding="utf-8").write(svg)
    renderPM.drawToFile(svg2rlg("dependancies/svg/" + lbl + '_' + str(i) + ".svg"), "dependancies/png/" + lbl + '_' + str(i) + ".png", fmt="PNG")
    im = Image.open("dependancies/png/" + lbl + '_' + str(i) + ".png").convert('L').resize((1000, 200))
    np.save('dependancies/np/' + lbl + '_' + str(i) + '.npy', np.array(im))

classes_dict = {"alcohol_and_drugs": [1, 0, 0, 0, 0, 0], "engagement": [0, 1, 0, 0, 0, 0], "neutral": [0, 0, 1, 0, 0, 0], "positive": [0, 0, 0, 1, 0, 0], "self_harm___harm_to_others": [0, 0, 0, 0, 1, 0], "vulnerability": [0, 0, 0, 0, 0, 1]}

labels, numpies = [], []
for np_path in os.listdir("dependancies/np/"):
    try:
        labels.append(classes_dict["_".join(np_path.split("_")[:-1])])
        numpies.append(np.load('dependancies/np/' + np_path))
    except: pass
    
np.save('dependancies/np/X.npy', np.array(numpies).astype(np.float))
np.save('dependancies/np/y.npy', np.array(labels).astype(np.float))
"""

X_train, X_test, y_train, y_test = train_test_split(np.load('dependancies/np/X.npy'), np.load('dependancies/np/y.npy'), test_size = 0.10, random_state = 42)   
 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (200, 1000, 1))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
predictions = Dense(6, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

model.summary(); #plot_model(model, show_shapes=True, to_file='residual_module.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("dep_resnet.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("dep_resnet_history.csv", separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.00001)
tensorboard = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=1, callbacks=[checkpoint, csv_logger, reduce_lr, tensorboard])

model = load_model("dep_resnet.hdf5")
 
y_pred = model.predict(X_test)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])

acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))

print(acc); print(kappa)