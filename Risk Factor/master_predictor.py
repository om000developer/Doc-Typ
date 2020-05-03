def master():

    from embeddings_lstm_predictor import pred_emb
    from vectorizedtf_conv_predictor import pred_vec
    from features_svm_predictor import pred_fea
    from dependancies_resnet_predictor import pred_dep
    
    emb_pred = pred_emb(sentence)
    vec_pred = pred_vec(sentence)
    fea_pred = pred_fea(sentence)
    dep_pred = pred_dep(sentence)
    
    predictions = [emb_pred, vec_pred, fea_pred, dep_pred]
    
    from statistics import mode
    
    try: final_pred = mode(predictions)
    except: final_pred = emb_pred
    
    return final_pred