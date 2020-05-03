sentence = "socially maintain relations family friends presentation thought related compliance, drug alcohol misuse psychosocial stressors"

import nltk; nltk.download('stopwords')
    
def corpus_extention(sentence):
    
    import warnings; warnings.filterwarnings("ignore")
    
    #############################################
    
    import requests
    from bs4 import BeautifulSoup as bs4
    
    def synonym_extender(sentence):
        
        words = sentence.split(" ")
        
        def synonymize(word):
            
            try:
            
                html_content = requests.get("https://www.thesaurus.com/browse/"+str(word)+"?s=t").text
                soup = bs4(html_content, "lxml")
                synonym = soup.find("ul", attrs={"class": "css-1lc0dpe et6tpn80"}).find("li").find("a").text
                
                if synonym: return synonym
                else: return word
            
            except: return word
        
        for i in range(len(words)): words[i] = synonymize(words[i]) if words[i].isalpha() else words[i]
            
        new_sentence = " ".join(words)
        
        return new_sentence
    
    sentence1 = synonym_extender(sentence)
    
    #############################################
    
    from PyDictionary import PyDictionary
    
    from tr4w import TextRank4Keyword
    
    from nltk.corpus import stopwords; stop_words = set(stopwords.words('english')) 
    
    def definition_extender_w_keyword_ranking_and_non_stopwords(sentence):
        
        words = sentence.split(" ")
        
        def define(word):
            
            try:
                
                import sys, os; sys.stdout = open(os.devnull, "w")
                definition = list(PyDictionary(word).getMeanings()[word].values())[0][0]
                sys.stdout = sys.__stdout__
                
                if definition: return definition
                else: return word
            
            except: return word
        
        for i in range(len(words)): words[i] = define(words[i]) if words[i].isalpha() else words[i]
        
        new_sentence = " ".join(words)
        
        try:
            
            tr4w = TextRank4Keyword()
            tr4w.analyze(new_sentence, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
            keywords, probabilities = tr4w.get_keywords(len(words))
        
            new_sentence_w_keyword_ranking = " ".join(keywords)
            
        except: new_sentence_w_keyword_ranking = new_sentence
        
        try: new_sentence_w_non_stopwords = " ".join([w for w in new_sentence.split(" ") if not w in stop_words and w.isalpha()])
        except: new_sentence_w_non_stopwords = new_sentence
        
        return new_sentence_w_keyword_ranking, new_sentence_w_non_stopwords
    
    sentence2, sentence3 = definition_extender_w_keyword_ranking_and_non_stopwords(sentence)
    
    #############################################
    
    return sentence1, sentence2, sentence3
    
sentence1, sentence2, sentence3 = corpus_extention(sentence)
sentence4, sentence5, sentence6 = corpus_extention(sentence1)
sentence7, sentence8, sentence9 = corpus_extention(sentence4)

print(sentence1); print("\n\n"); print(sentence2); print("\n\n"); print(sentence3); print("\n\n");
print(sentence4); print("\n\n"); print(sentence5); print("\n\n"); print(sentence6); print("\n\n");
print(sentence7); print("\n\n"); print(sentence8); print("\n\n"); print(sentence9); print("\n\n");