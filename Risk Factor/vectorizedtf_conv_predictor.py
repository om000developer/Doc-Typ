def pred_vec(sample_sentence):

    import re, numpy as np, pandas as pd, sklearn, keras, pickle
    from datetime import datetime
    from nltk.corpus import stopwords
    from sklearn.decomposition import PCA
    
    def clean_text(text):
        text = text.lower()
        text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text)
        text = re.compile('[^0-9a-z #+_]').sub('', text)
        text = ' '.join(word for word in text.split() if word not in set(stopwords.words('english')))
        return text
    
    sample_sentence = clean_text(sample_sentence)
    
    vectorizer = pickle.load(open("tfidf_fitted_vect.pickle", "rb"))
    model = keras.models.load_model("tfidf-vec_model.hdf5")
    
    sample_sentence = vectorizer.transform([sample_sentence]).toarray()
    
    rp_reload = pickle.load(open("rp_tfidf.pickle", "rb"))
    sample_sentence = rp_reload.transform(sample_sentence)
    sample_sentence = sample_sentence.reshape(sample_sentence.shape[0], sample_sentence.shape[1], 1)
    
    y_pred = model.predict(sample_sentence)
    labels = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]
    
    return labels[np.argmax(y_pred)]