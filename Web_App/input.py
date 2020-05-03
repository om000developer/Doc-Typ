def data(answers_path, main_path):
    
    import os, shutil, time
    from datetime import datetime
    
    videos = []
    for text_filename in os.listdir(answers_path):
        if "video_" in text_filename and ".webm" in text_filename: videos.append(text_filename)
        
    videos.sort(key = lambda date: datetime.strptime(date, 'video_%Y-%m-%d_%H-%M-%S.webm')) 
    
    for video in videos:
        src = answers_path + video
        dest = main_path + "answers/audio_" + "_".join(video.split("_")[1:]).replace(".webm", "") + ".wav"
        os.popen('ffmpeg -i' + ' ' + '"' + src + '"' + ' ' + '"' + dest + '"')
    [ shutil.copy(answers_path + video, main_path + "answers/") for video in videos ]
    
    time.sleep(4)
    video_filenames = [ "answers/" + video for video in [ answer_video for answer_video in os.listdir("answers/") if "video_" in answer_video and ".webm" in answer_video ] ]
    audio_filenames = [ "answers/" + audio for audio in [ answer_audio for answer_audio in os.listdir("answers/") if "audio_" in answer_audio and ".wav" in answer_audio ] ]
    
    msg = "Copied Videos To Main Folder & Extracted Their Audio!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    import speech_recognition as sr
    
    basic_info = open(answers_path + "basic_info.txt").read().split(";")
    base_sentences = "Name: " + basic_info[0] + ". Age: " + basic_info[1] + ". Gender: " + basic_info[2] + ". Location: " + basic_info[3] + ". "
    
    sentences = []
    
    for file in audio_filenames:
        
        r = sr.Recognizer()
        with sr.AudioFile(file) as source: audio = r.record(source)
        sentences.append(r.recognize_sphinx(audio))
    
    combined_answers = ". ".join(sentences)
    
    msg = "Speech Recognized & Converted to Text!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array
    import cv2, numpy as np
    
    face_classifier = cv2.CascadeClassifier('static/Haar_Cascade.xml')
    classifier = load_model('static/Model_v2.h5')
    
    facial_class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']; predicted_facial_labels = []
    
    for file in video_filenames:
        
        cap = cv2.VideoCapture(file)
        
        ret, frame = cap.read()
        ret = True
        
        while ret:
            
            ret, frame = cap.read()
            if ret == True:
                
                faces = face_classifier.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.3,5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[y:y+h,x:x+w],(48,48),interpolation=cv2.INTER_AREA)
                    
                    if np.sum([roi_gray])!=0: predicted_facial_labels.append(facial_class_labels[classifier.predict(np.expand_dims(img_to_array(roi_gray.astype('float')/255.0),axis=0))[0].argmax()])
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
    
    msg = "Facial Emotions Recognized!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    import librosa, soundfile, pickle, numpy as np
    
    loaded_model = pickle.load(open("static/Model.sav", 'rb'))
    
    predicted_speech_labels = []
    
    for file in audio_filenames:
            
        with soundfile.SoundFile(file) as sound_file:
            
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            mfcc, chroma, mel = True, True, True
            
            if chroma: stft=np.abs(librosa.stft(X))
            result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
                
        predicted_speech_labels.append(str(loaded_model.predict([result])[0]))    
    
    msg = "Speech Sentiment Analyzed!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    def delete_contents(file_path):
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e: print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    f_to_del = videos; f_to_del.append("basic_info.txt"); f_to_del.append("end.txt")
    for filename in [answers_path + f for f in f_to_del]: file_path = os.path.join(answers_path, filename); delete_contents(file_path)
    for file_path1 in video_filenames: delete_contents(file_path1)
    for file_path2 in audio_filenames: delete_contents(file_path2)
    
    msg = "Deleted All Input Records!"
    print(msg); open("outputs/output_status.txt", "w").write(msg)
    
    return (base_sentences, combined_answers, predicted_facial_labels, predicted_speech_labels)