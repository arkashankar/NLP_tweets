# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:59:49 2019

@author: Mypc
"""
import pandas as pd
import numpy as np
#import preprocessor as pre
import matplotlib.pyplot as plt
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
import random
import sklearn
from scipy import spatial
import statistics
 

#from sklearn.feature_extraction.text import CountVectorizer
#from gensim import corpora,models,similarities
#nltk.download('punkt')
data = pd.read_excel('Clean_Data_Nepal.xlsx')

X = data.iloc[:,3].values
dfx = pd.DataFrame(X)



y = []
for i in range(len(X)):
    y.append(str(dfx.iloc[i][0]))
dfy = pd.DataFrame(y)

corpus1=[]
corpus = []
for i in range(len(X)):
    text = re.sub(r'http?://\S+',' ', dfy.iloc[i][0], flags = re.MULTILINE)
    #text = dfy.iloc[i][0]
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
    corpus1.append(text)
    
Z=[]
for i in range (80):
    m=random.randint(1,50067)
    Z.append(corpus1[m][:])
dfz= pd.DataFrame(Z)
tok_fact=[nltk.word_tokenize(fact) for fact in Z]

tok_corp= [nltk.word_tokenize(sent) for sent in corpus]


model= Word2Vec(tok_corp,min_count=1, size=32)
vocabulary = model.wv.vocab
vocablist=[v for v in vocabulary.values()] 
mean_vec=[[]]*len(corpus)
for i in range(len(corpus)):
    vector=[]
    for j in range(len(tok_corp[i])):
        vector.append(model.wv[tok_corp[i][j]])
    mean=np.mean(vector,axis=0) 
    mean_vec[i]=(mean)

#vector=model.wv[vocabulary]

#mean_vocab=vector.mean(axis=1)
    
#tf-idf score
DF = {}
for i in range(len(tok_fact)):
    tokens = tok_fact[i]
    for word in tokens:
        try:
            DF[word].add(i)
        except:
            DF[word]={i}

for i in DF:
    DF[i] = len(DF[i])

total_vocab = [x for x in DF]    
    
def doc_freq(token):
    count = 0
    for doc in tok_fact:
        if token in doc:
            count+=1
    return count

def Counter(tokens):
    dict ={}
    for token in tokens:
        dict.update({token:0})
    for token in tokens:
        dict[token] +=1
    return dict

def words_count(tokens):
    return len(tokens)

N = len(tok_corp)
#calculating TF-IDF for each token
tf_idf = {}
for i in range(len(tok_fact)):
    tokens = tok_fact[i]
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count(tokens)
        df = doc_freq(token)
        idf = np.log(N/df+1)
        tf_idf[i,token] = tf*idf
    

vector_embedding=[]
for j in range(len(total_vocab)):
    vector_embedding.append(model.wv[total_vocab[j]]) 
    
Num=0
den=0


new_score = {}
for i in range (len(tok_fact)):
    tokens= tok_fact[i]
    for token in np.unique(tokens):
        score = tf_idf[i,token]
        if token not in new_score:
            new_score.update({token:score})
        else:
            new_score.update({token:max(score,new_score[token])})





for token in new_score:
    Num = Num + new_score[token]*(model.wv[token])
    
for token in new_score:
    den= den+ new_score[token]

V= Num/den

cosine_similar=[]
for i in range(len(mean_vec)):
    #cosine=sklearn.metrics.pairwise.cosine_similarity(mean_vec[i], V)
    cosine=1 - spatial.distance.cosine(mean_vec[i], V)
    cosine_similar.append(cosine)

tweet_no = []
for i in range(len(mean_vec)):
    if cosine_similar[i]<0.8:
        tweet_no.append(tok_corp[i])
    
N_initial={}
for i in range (len(tweet_no)):
    tokens = tweet_no[i]
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count(tokens)
        N_initial[token] = tf
    
   
sorted_N = sorted(N_initial.items(),key=lambda kv: kv[1])
N_words=[]
for  i in range(500):
    N_words.append(sorted_N[i][0])

fact_checkable=[]    
for i in range(len(tok_corp)):
    tok = tok_corp[i]
    for token in tok:
        if token not in N_words and token in total_vocab and tok_corp[i] not in tweet_no :
            fact_checkable.append(tok_corp[i])
            break
        else:
            continue
            







    
    
    




