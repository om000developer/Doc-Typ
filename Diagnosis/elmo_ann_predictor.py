def pred_elm(sample_case):
    
    import numpy as np, keras, json
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    
    classes_dict = {0: "Major Depressive Disorder", 1: "Attention Deficit Hyperactivity Disorder", 2: "Oppositional Defiant Disorder", 3: "Conduct Disorder", 4: "Pervasive Developmental Disorder", 5: "Intellectual Disability (Mental Retardation)", 6: "Psychotic Disorder", 7: "Adjustment Disorder", 8: "Mood Disorder", 9: "General Anxiety Disorder", 10: "Social Anxiety Disorder", 11: "Seasonal Affective Disorder", 12: "Substance Abuse", 13: "Autism Spectrum Disorder"}
    
    t = Tokenizer(); t.fit_on_texts([sample_case])
    content = pad_sequences(sequences=t.texts_to_sequences([sample_case]), maxlen=500)
    model = keras.models.load_model("elmo_ann.hdf5")
    diagnosis = list(np.array(model.predict(content)).flatten())
    max1 = classes_dict[diagnosis.index(max(diagnosis))]
    
    with open('external_resources.json') as data_file:    
        for v in json.load(data_file): 
            if v['diagnosis'] == max1: about1, treatment1 = v['about'], v['treatment']
            
    return (max1, about1, treatment1)