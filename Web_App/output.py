def deliverables(basic_info, case, facial_emotion, speech_sentiment):
    
    import os, re, json, time, networkx as nx, RAKE
    import numpy as np, pandas as pd, keras, pickle
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import cosine_similarity
    from static.tr4w import TextRank4Keyword
    
    sentences = case.split(". ")
    
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    def remove_stopwords(sen): return " ".join([i for i in sen if i not in stopwords.words('english')])
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    
    word_embeddings = {}
    f = open('static/glove.6B.100d.txt', encoding='utf-8')
    for line in f: word_embeddings[line.split()[0]] = np.asarray(line.split()[1:], dtype='float32')
    f.close()
    
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0: v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else: v = np.zeros((100,))
      sentence_vectors.append(v)
      
    sim_mat = np.zeros([len(sentences), len(sentences)])
    
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j: sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    ex_summary1 = []
    for i in range(5): ex_summary1.append(ranked_sentences[i][1])
    
    msg = "Extractive Summary 1 Generated!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    tr4w = TextRank4Keyword()
    tr4w.analyze(case, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
    keywords, probabilities = tr4w.get_keywords(5)
        
    ex_summary2 = []
    for keyword in keywords: ex_summary2.append(re.findall(r"([^.]*?"+keyword+"[^.]*\.)", case))
    ex_summary2 = list(set([item for sublist in ex_summary2 for item in sublist]))
    
    msg = "Extractive Summary 2 Generated!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    ex_summary = list(set(ex_summary1 + ex_summary2))
    
    ordered_ex_summary = []
    for ori_stc in sentences:
        for sel_stc in ex_summary:
            if sel_stc in ori_stc: 
                ordered_ex_summary.append(sel_stc)
    ordered_ex_summary = ". ".join(ordered_ex_summary)
    
    with open('outputs/extractive_summary.txt', 'w') as f: f.write(basic_info + ordered_ex_summary)
    
    msg = "Extractive Summary Combined & Exported Locally!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    classes_dict = {0: "Major Depressive Disorder", 1: "Attention Deficit Hyperactivity Disorder", 2: "Oppositional Defiant Disorder", 3: "Conduct Disorder", 4: "Pervasive Developmental Disorder", 5: "Intellectual Disability (Mental Retardation)", 6: "Psychotic Disorder", 7: "Adjustment Disorder", 8: "Mood Disorder", 9: "General Anxiety Disorder", 10: "Social Anxiety Disorder", 11: "Seasonal Affective Disorder", 12: "Substance Abuse", 13: "Autism Spectrum Disorder"}
    
    t = Tokenizer(); t.fit_on_texts([case])
    content = pad_sequences(sequences=t.texts_to_sequences([case]), maxlen=500)
    model = keras.models.load_model("static/diagnostic_model.hdf5")
    diagnosis = list(np.array(model.predict(content)).flatten())
    max1 = classes_dict[diagnosis.index(max(diagnosis))]; diagnosis[diagnosis.index(max(diagnosis))] = 0;
    max2 = classes_dict[diagnosis.index(max(diagnosis))];
    
    with open('static/external_resources.json') as data_file:    
        for v in json.load(data_file): 
            if v['diagnosis'] == max1: about1, treatment1 = v['about'], v['treatment']
            if v['diagnosis'] == max2: about2, treatment2 = v['about'], v['treatment']
        
    with open('outputs/diagnosis_info.json', 'w') as f: f.write(json.dumps([ {"diagnosis": max1, "link": about1}, {"diagnosis": max2, "link": about2} ]))
    
    msg = "Predicted Diagnosis & Exported Locally!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    with open('static/tokenizer.pickle', 'rb') as handle: tokenizer = pickle.load(handle)
    model = keras.models.load_model("static/riskfactor_model.hdf5")
    
    risk_factor_probabilities = []
    for sentence in case.split("."):
        preds = [ model.predict(pad_sequences(tokenizer.texts_to_sequences([sentence]), maxlen=250)) ]
        risk_factor_probabilities.append(np.array(preds))
    risk_factor_probabilities = [ round(p*100, 2) for p in np.mean(np.array(risk_factor_probabilities), axis=0)[0][0] ]
    
    msg = "Predicted Risk-Factors & Exported Locally!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    extracted_segments = [ i[0] for i in RAKE.Rake(os.path.join(os.getcwd(), "static/smartstoplist.txt")).run(case)[:25] ]
    probabilities = [ float(i[1]*10) for i in RAKE.Rake(os.path.join(os.getcwd(), "static/smartstoplist.txt")).run(case)[:25] ]
    
    corresponding_sentences = []
    for segment in extracted_segments: corresponding_sentences.append(re.findall(r"([^.]*?"+segment+"[^.]*\.)", case))
    corresponding_colors = {"alcohol_and_drugs": "#A367-DC", "engagement": "#6771DC", "neutral": "#6794DC", "positive": "#67B7DC", "self_harm___harm_to_others": "#C767DC", "vulnerability": "#8067DC"}
    
    risk_factors_colors = []
    for stcs in corresponding_sentences:
        preds = []
        for stc in stcs: preds.append(model.predict(pad_sequences(tokenizer.texts_to_sequences([stc]), maxlen=250)))
        risk_factors_colors.append(corresponding_colors[list(corresponding_colors)[np.argmax(np.mean(np.array(preds), axis=0))]])
    
    with open('outputs/riskfactor_info.json', 'w') as f: f.write(json.dumps([ dict(zip(list(corresponding_colors), risk_factor_probabilities)) ]))
    
    msg = "Important Segments (+ Probabilities w/ Risk-Factor Predictions) Extracted!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    bubble_chart_params = []
    for x in range(len(extracted_segments)):
        bubble_chart_params.append({"Shortname": extracted_segments[x].title(), "Name": extracted_segments[x].title(), "Count": probabilities[x], "Category": {v: k for k, v in corresponding_colors.items()}[risk_factors_colors[x]], "Color": risk_factors_colors[x]})
    
    with open('outputs/bubble_chart_params.json', 'w') as f: f.write(json.dumps(bubble_chart_params))
    
    msg = "Bubble Chart Data Parameters Formatted!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    facial_emotion_text = "The patient's most prevalent facial emotion, that was detected was '" + max(facial_emotion).lower() + "', with a secondary stress on '" + max(list(filter((max(facial_emotion)).__ne__, facial_emotion))).lower() + "'."
    with open('outputs/facial_summary.txt', 'w') as f: f.write(facial_emotion_text)
    
    speech_sentiment_text = "The patient's common sentiment that was analyzed, as expressed by speech was '" + max(speech_sentiment).lower() + "'."
    with open('outputs/speech_summary.txt', 'w') as f: f.write(speech_sentiment_text)
    
    msg = "Facial Emotion + Speech Sentiment Summarized!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    speech_dict = { "calm_s_count": speech_sentiment.count('calm'), "happy_s_count": speech_sentiment.count('happy'), "fearful_s_count": speech_sentiment.count('fearful'), "disgust_s_count": speech_sentiment.count('disgust') }
    speech_sentiment_uniques = ["calm", "happy", "fearful", "disgust"]
    
    facial_dict = { "angry_f_count": facial_emotion.count('Angry'), "disgust_f_count": facial_emotion.count('Disgust'), "fear_f_count": facial_emotion.count('Fear'), "happy_f_count": facial_emotion.count('Happy'), "neutral_f_count": facial_emotion.count('Neutral'), "sad_f_count": facial_emotion.count('Sad'), "surprise_f_count": facial_emotion.count('Surprise') }
    facial_emotion_uniques = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    connectogram_graph_params = []
    for j in speech_sentiment_uniques:
         for i in facial_emotion_uniques: connectogram_graph_params.append({ "from": j + " (speech)", "to": i + " (facial)", "value": speech_dict[j + "_s_count"] })
    for i in facial_emotion_uniques:
         for j in speech_sentiment_uniques: connectogram_graph_params.append({ "from": i + " (facial)", "to": j + " (speech)", "value": facial_dict[i + "_f_count"] })
        
    with open('outputs/connectogram_graph_params.json', 'w') as f: f.write(json.dumps(connectogram_graph_params))
    
    msg = "Connectogram Graph Data Parameters Formatted!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    time.sleep(2); 
    msg = "Everything Processed..."
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    time.sleep(2);
    
    open("outputs/output_status.txt", "w").write("finished")
    time.sleep(10); os.remove("outputs/output_status.txt")