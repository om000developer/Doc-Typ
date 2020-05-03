import os, json, re, csv, random, requests
from bs4 import BeautifulSoup as bs4
from nltk import word_tokenize, pos_tag
import nltk; nltk.download("averaged_perceptron_tagger")

cases_requirement = 20
sentences_requirement = 10

with open("diagnostic_cases_60.json", "r") as fp: input = json.load(fp)
    
def corpus_extender(requirement, sentence):

    idx = []; text = word_tokenize(sentence)
        
    word_types = pos_tag(text)
    
    for word in word_types:
        if word[1] ==  "RB" or word[1] ==  "JJ" or word[1] == "NN": idx.append(text.index(word[0]))
    
    for i in idx:
        globals()["synonyms_"+str(i)] = []
        try:
            soup = bs4(requests.get("https://www.thesaurus.com/browse/"+str(text[i])+"?s=t").text, "lxml")
            for synonym in soup.find("ul", attrs={"class": "css-1lc0dpe et6tpn80"}).find_all("li"):
                try: 
                    if len(synonym.find("a").text.split(" ")) == 1: globals()["synonyms_"+str(i)].append(synonym.find("a").text)
                except: pass
        except: pass
        
    for req in range(requirement):
        globals()["sentence_"+str(req+1)] = list(text)
        for i in idx: 
            if globals()["synonyms_"+str(i)] != []:
                globals()["sentence_"+str(req+1)][i] = globals()["synonyms_"+str(i)][random.randint(0, len(globals()["synonyms_"+str(i)])-1)]
        
    multi_word_changes = list(set([ " ".join(globals()["sentence_"+str(req+1)]) for req in range(requirement) ]))
        
    return multi_word_changes, word_types

with open("augmented_diagnosis_cases.csv", "w+") as op: csv.writer(op).writerow(['Case', 'Diagnosis'])

for record in input: 
    with open("augmented_diagnosis_cases.csv", "a+", newline="") as op: csv.writer(op).writerow([record['content'], record['diagnosis']])
    sentences = [stc + ". " for stc in record['content'].split(".")]
    for aug in range(cases_requirement): globals()['case_' + record['Case'] + '_aug_' + str(aug)] = ""
    unique_rnd_stcs, rnd_stcs = [], []
    sentences_requirement = len(sentences) if len(sentences) < sentences_requirement else sentences_requirement
    for i in range(sentences_requirement): 
        def gen_rnd():
            rnd = random.randint(0, len(sentences) - 1)
            if rnd not in unique_rnd_stcs: unique_rnd_stcs.append(rnd)
            else: gen_rnd()
            return unique_rnd_stcs
        gen_rnd()
    for sentence in range(len(sentences)): 
        if sentence in unique_rnd_stcs: multi_word_changes, word_types = corpus_extender(cases_requirement, sentences[sentence])
        multi_word_changes, word_types = corpus_extender(cases_requirement, sentences[sentence])
        for aug in range(cases_requirement):
            try: globals()['case_' + record['Case'] + '_aug_' + str(aug)] += re.sub(' +', ' ', multi_word_changes[aug])
            except: pass
    for aug in range(cases_requirement): 
        globals()['case_' + record['Case'] + '_aug_' + str(aug)] = globals()['case_' + record['Case'] + '_aug_' + str(aug)].replace(",", ", ")
        globals()['case_' + record['Case'] + '_aug_' + str(aug)] = globals()['case_' + record['Case'] + '_aug_' + str(aug)].replace(".", ". ")
        with open("augmented_diagnosis_cases.csv", "a+", newline="") as op: csv.writer(op).writerow([globals()['case_' + record['Case'] + '_aug_' + str(aug)], record['diagnosis']])
    print("Augmented Case: " + record['Case'])

with open('augmented_diagnosis_cases.csv', 'rU') as f: 
    reader = csv.DictReader(f, fieldnames = ("Case", "Diagnosis"))  
    open('augmented_diagnosis_cases.json', 'w').write(json.dumps([ row for row in reader ]))
os.remove('augmented_diagnosis_cases.csv')