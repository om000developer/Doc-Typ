def pred_net(sample_case):
        
    import numpy as np, keras
    from pathlib import Path
    from spacy import displacy
    from PIL import Image
    import json, pytextrank, networkx as nx
    import matplotlib.pyplot as plt
    
    path_stage0 = "o0.json"; path_stage1 = "o1.json"
    
    file_dic = {"id" : 0, "text" : sample_case}
    loaded_file_dic = json.loads(json.dumps(file_dic))
    
    with open(path_stage0, 'w') as outfile: json.dump(loaded_file_dic, outfile)
    
    with open(path_stage1, 'w') as f:
        for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
            f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
            print(pytextrank.pretty_print(graf._asdict()))
    
    graph, ranks = pytextrank.text_rank(path_stage1)
    pytextrank.render_ranks(graph, ranks)
    
    nx.draw(graph, with_labels=True);
    plt.savefig("sample_case.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    im = Image.open("sample_case.png").convert('L').resize((300, 200))
    sample_image = np.array([np.array(im)])
    sample_image = sample_image.reshape(sample_image.shape[0], sample_image.shape[1], sample_image.shape[2], 1)
    
    model = keras.models.load_model("graph_conv_autoencoder.hdf5")
     
    y_pred = model.predict(sample_image)
    labels = ['Major Depressive Disorder', 'Attention Deficit Hyperactivity Disorder', 'Oppositional Defiant Disorder', 'Conduct Disorder', 'Pervasive Developmental Disorder', 'Intellectual Disability (Mental Retardation)', 'Psychotic Disorder', 'Adjustment Disorder', 'Mood Disorder', 'General Anxiety Disorder', 'Social Anxiety Disorder', 'Seasonal Affective Disorder', 'Substance Abuse', 'Autism Spectrum Disorder']
    
    max1 = labels[np.argmax(y_pred)]
    
    with open('external_resources.json') as data_file:    
        for v in json.load(data_file): 
            if v['diagnosis'] == max1: about1, treatment1 = v['about'], v['treatment']
            
    return (max1, about1, treatment1)