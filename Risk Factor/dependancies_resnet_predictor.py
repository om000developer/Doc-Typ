def pred_dep(sample_sentence):
        
    import keras
    import numpy as np
    import en_core_web_sm 
    from PIL import Image
    from pathlib import Path
    from spacy import displacy
    nlp = en_core_web_sm.load()
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    
    svg = displacy.render(nlp(sample_sentence), style="dep", jupyter=False)
    Path("sample_sentence.svg").open("w", encoding="utf-8").write(svg)
    renderPM.drawToFile(svg2rlg("sample_sentence.svg"), "sample_sentence.png", fmt="PNG")
    im = Image.open("sample_sentence.png").convert('L').resize((1000, 200))
    sample_image = np.array([np.array(im)])
    sample_image = sample_image.reshape(sample_image.shape[0], sample_image.shape[1], sample_image.shape[2], 1)
    
    model = keras.models.load_model("dep_resnet.hdf5")
     
    y_pred = model.predict(sample_image)
    labels = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]
    
    return labels[np.argmax(y_pred)]