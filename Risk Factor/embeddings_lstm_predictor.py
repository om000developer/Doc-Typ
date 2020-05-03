def pred_emb(sample_sentence):

    import numpy as np, keras, pickle
    from keras.preprocessing.sequence import pad_sequences
    
    with open('tokenizer.pickle', 'rb') as handle: tokenizer = pickle.load(handle)
    model = keras.models.load_model("riskfactor_model.hdf5")
    pred = model.predict(pad_sequences(tokenizer.texts_to_sequences([sample_sentence]), maxlen=375))
    labels = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]
    
    return labels[np.argmax(pred)]