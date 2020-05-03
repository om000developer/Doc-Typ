import ast
import nltk
import joblib
import string
import sklearn
import numpy as np
import pandas as pd
import en_core_web_sm
import sklearn.metrics as sm
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import preprocessing, svm
from nltk.tokenize import word_tokenize 
from sklearn.preprocessing import StandardScaler 

classes = ["alcohol_and_drugs", "engagement", "neutral", "positive", "self_harm___harm_to_others", "vulnerability"]

df = pd.read_csv('augmented_riskfactor_sentences.csv', encoding='latin_1')
df = df[pd.notnull(df['Sentence'])]
df = df[pd.notnull(df['Risk_Factor'])]
df.Risk_Factor.value_counts()

df = df.reset_index(drop=True)
df = df.sample(frac=1)
    
nlp = en_core_web_sm.load()

df['Characters_Total'] = df['Sentence'].apply(lambda x: len(x.replace(" ", "")))

df['Stopwords_No'] = df['Sentence'].apply(lambda x: sum(1 for y in x.split(" ") if y in set(stopwords.words('english'))))

count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
df['Punctuation_%'] = df['Sentence'].apply(lambda x: count(x, string.punctuation)/len(x.replace(" ", "")))

df['Polarity'] = df['Sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)

df['Subjectivity'] = df['Sentence'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

df['Entities_%'] = df['Sentence'].apply(lambda x: sum(1 for i in nlp(x).ents)/len(x.split(" ")))

avg_nav_counts = []
for sent in df['Sentence']:
    noun_count = 0; adj_count = 0; verb_count = 0;
    for wrd in nltk.pos_tag(word_tokenize(sent)): 
        if wrd[1] == "NN" or wrd[1] == "NNS" or wrd[1] == "NNP" or wrd[1] == "NNPS": noun_count = noun_count + 1
        if wrd[1] == "JJ" or wrd[1] == "JJS" or wrd[1] == "JJR" or wrd[1] == "NNPS": adj_count = adj_count + 1
        if wrd[1] == "VB" or wrd[1] == "VBD" or wrd[1] == "VBG" or wrd[1] == "VBN" or wrd[1] == "VBP" or wrd[1] == "VBZ": verb_count = verb_count + 1
    nav = ((noun_count + adj_count + verb_count) / 3) / len(sent.split(" "))
    avg_nav_counts.append(nav)
df['Avg_NN_ADJ_VB_%'] = avg_nav_counts

baseline_sentence = {"this", "is", "a", "neutral", "sentence."}
cosine_similarities = []
for sent in df['Sentence']:
    l1 = []; l2 = []
    Y_set = {w for w in word_tokenize(sent)}
    rvector = baseline_sentence.union(Y_set)  
    for w in rvector: 
        if w in baseline_sentence: l1.append(1)
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    for i in range(len(rvector)): c += l1[i] * l2[i] 
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5) 
    cosine_similarities.append(cosine)
df['Cosine_Similarity'] = cosine_similarities

X = df[['Polarity', 'Subjectivity', 'Entities_%', 'Avg_NN_ADJ_VB_%', 'Cosine_Similarity', 'Characters_Total', 'Stopwords_No', 'Punctuation_%']].values

scaler = StandardScaler() 
X = scaler.fit_transform(X) 

encoder = preprocessing.LabelEncoder()
encoder.fit(list(df['Risk_Factor'])); y = encoder.transform(list(df['Risk_Factor']))

"""
import ga

num_samples = X.shape[0]
num_feature_elements = X.shape[1]

train_indices = np.arange(1, num_samples, 4)
test_indices = np.arange(0, num_samples, 4)

sol_per_pop = 8 # Population size.
num_parents_mating = 4 # Number of parents inside the mating pool.
num_mutations = 3 # Number of elements to mutate.

# Defining the population shape.
pop_shape = (sol_per_pop, num_feature_elements)

# Creating the initial population.
new_population = np.random.randint(low=0, high=2, size=pop_shape)
print(new_population.shape)

best_outputs = []
num_generations = 100

for generation in range(num_generations):

    print("Generation : ", generation)

    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(new_population, X, y, train_indices, test_indices, generation)

    best_outputs.append(np.max(fitness))

    # The best result in the current iteration.
    print("Best result : ", best_outputs[-1])

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

    # Adding some variations to the offspring using mutation
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=num_mutations)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(new_population, X, y, train_indices, test_indices)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))[0]
best_match_idx = best_match_idx[0]

best_solution = new_population[best_match_idx, :]
best_solution_indices = np.where(best_solution == 1)[0]
best_solution_num_elements = best_solution_indices.shape[0]
best_solution_fitness = fitness[best_match_idx]

print("\nBest match idx : ", best_match_idx)
print("Best solution : ", best_solution)
print("Selected indices : ", best_solution_indices) # OUTPUT
print("Number of selected elements : ", best_solution_num_elements) 
print("Best solution fitness : ", best_solution_fitness)

with open("ga_output.txt", "w") as fp: fp.write(str(list(best_solution_indices)))

plt.plot(best_outputs)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.show()

"""

# Output features deemed most important by ^^^ were ***[0 1 4 5 6 7]***
with open("ga_output.txt", "r") as fp: best_solution_indices = ast.literal_eval(fp.read().replace(" ", ", "))

X_new = []
for rec in X:
    vals = []
    for feature in best_solution_indices: vals.append(rec[feature])
    X_new.append(vals)
X_new = np.array(X_new)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y, test_size = 0.25)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

joblib.dump(clf, 'riskfactor_svc.sav')

y_pred = clf.predict(X_test)

acc = str(round(sm.accuracy_score(list(y_pred), list(y_test))*100, 3))
kappa = str(round(sm.cohen_kappa_score(list(y_pred), list(y_test)), 3))

print(acc); print(kappa)