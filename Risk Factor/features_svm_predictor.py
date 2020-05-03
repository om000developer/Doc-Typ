def pred_fea(sample_sentence):
    
    import numpy as np, pandas as pd, nltk, ast
    import sklearn, en_core_web_sm, string, joblib
    from textblob import TextBlob
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize 
    from sklearn import preprocessing, svm
    import sklearn.metrics as sm
    
    nlp = en_core_web_sm.load()
    
    noun_count = 0; adj_count = 0; verb_count = 0;
    for wrd in nltk.pos_tag(word_tokenize(sample_sentence)): 
        if wrd[1] == "NN" or wrd[1] == "NNS" or wrd[1] == "NNP" or wrd[1] == "NNPS": noun_count = noun_count + 1
        if wrd[1] == "JJ" or wrd[1] == "JJS" or wrd[1] == "JJR" or wrd[1] == "NNPS": adj_count = adj_count + 1
        if wrd[1] == "VB" or wrd[1] == "VBD" or wrd[1] == "VBG" or wrd[1] == "VBN" or wrd[1] == "VBP" or wrd[1] == "VBZ": verb_count = verb_count + 1
    nav = ((noun_count + adj_count + verb_count) / 3) / len(sample_sentence.split(" "))
    
    baseline_sentence = {"this", "is", "a", "neutral", "sentence."}
    l1 = []; l2 = []
    Y_set = {w for w in word_tokenize(sample_sentence)}
    rvector = baseline_sentence.union(Y_set)  
    for w in rvector: 
        if w in baseline_sentence: l1.append(1)
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    for i in range(len(rvector)): c += l1[i] * l2[i] 
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5) 
        
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    feature_vector = [TextBlob(sample_sentence).sentiment.polarity, TextBlob(sample_sentence).sentiment.subjectivity, sum(1 for i in nlp(sample_sentence).ents)/len(sample_sentence.split(" ")), nav, cosine, len(sample_sentence.replace(" ", "")), sum(1 for y in sample_sentence.split(" ") if y in set(stopwords.words('english'))), count(sample_sentence, string.punctuation)/len(sample_sentence.replace(" ", ""))]
    
    with open("ga_output.txt", "r") as fp: best_solution_indices = ast.literal_eval(fp.read().replace(" ", ", "))
    feature_vector_new = []
    for feature in best_solution_indices: feature_vector_new.append(feature_vector[feature])
    
    loaded_model = joblib.load('riskfactor_svc.sav')
    y_pred = loaded_model.predict(np.array([feature_vector_new]))
    
    labels = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]
    
    return labels[y_pred[0]]