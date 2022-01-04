# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:49:17 2019
Task 1
@author: Vikram
"""

import pandas as pd
import numpy as np
#import preprocessor as pre
import matplotlib.pyplot as plt
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_excel('Clean_Data_Nepal.xlsx')

X = data.iloc[:,3].values
dfx = pd.DataFrame(X)

y = []
for i in range(len(X)):
    y.append(str(dfx.iloc[i][0]))
dfy = pd.DataFrame(y)

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
     
cv = CountVectorizer(max_features = 1500)
corpus = cv.fit_transform(corpus).toarray()

model = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
z = model.fit_predict(corpus)


c1 = 0
c0 = 0

for i in range(len(z)):
    if z[i]==1:
        c1+=1
    else:
        c0+=1

fctweets = []
for i in range(len(z)):
    if z[i]==1:
        fctweets.append(y[i])
        
        

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 02:17:24 2019
Task 2
@author: Vikram
"""
import os
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
#from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
import operator

#from sklearn.feature_extraction.text import CountVectorizer
#from gensim import corpora,models,similarities
#nltk.download('punkt')
data = pd.read_excel('Clean_Data_Nepal.xlsx')

X = data.iloc[:,3].values
dfx = pd.DataFrame(X)
dffctweets = pd.DataFrame(fctweets)

y = []
for i in range(len(fctweets)):
    y.append(str(dffctweets.iloc[i][0]))
dfy = pd.DataFrame(y)

corpus1=[]
corpus = []
for i in range(len(fctweets)):
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
tok_corp_tweets= [nltk.word_tokenize(sent) for sent in corpus]

news=[]
for f in os.listdir("G:/Dataset/Document from Vikram/nepal-quake-2015-news-articles"):
    with open(f, "r", encoding="utf8") as read_file:
        c = read_file.read(500)
        news.append(c)
        
corpus2=[]
for j in range(len(news)):
    text2=re.sub(r'http?://\S+',' ', news[j], flags = re.MULTILINE)
    text2 = re.sub('[^a-zA-Z]', ' ', text2)
    text2 = re.sub('url',' ',text2)
    text2 = re.sub('date',' ',text2)
    text2.lower()
    #text2.split()
    #ps2= PorterStemmer()
    #text2=[ps2.stem(word) for word in text2 if not word in set(stopwords.words('english'))]
    #text2 = ' '.join(text2)
    corpus2.append(text2)
    

tok_corp= [nltk.word_tokenize(sent) for sent in corpus2]
ps2= PorterStemmer()
news_article=[[]]*len(tok_corp)
stem_sentence=[]
for i in range (len(tok_corp)):
    for word in tok_corp[i]:
        stem_sentence.append(ps2.stem(word))
        news_article[i]= stem_sentence
    stem_sentence=[]




#save news articles in list named articles
# and save processed news articles in list named news_articles

#calculating document frequency
DF = {}
for i in range(len(news_article)):
    tokens = news_article[i]
    for word in tokens:
        try:
            DF[word].add(i)
        except:
            DF[word]={i}

for i in DF:
    DF[i] = len(DF[i])

total_vocab = [x for x in DF]

#funct for counting words in document
#def words_count(tokens):
#    return len(tokens)

def doc_freq(token):
    count = 0
    for doc in news_article:
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

N = len(news_article)
#calculating TF-IDF for each token
tf_idf = {}
for i in range(len(news_article)):
    tokens = news_article[i]
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count(tokens)
        df = doc_freq(token)
        idf = np.log(N/df+1)
        tf_idf[i,token] = tf*idf

def matching_score(query):
    score_dict = {}
    for key in tf_idf:
        if key[1] in query:
            if key[0] in score_dict:
                score_dict[key[0]] += tf_idf[key]
            else:
                score_dict[key[0]] = tf_idf[key]
    return max(score_dict.items(), key=operator.itemgetter(1))[0]

relevant_newsid = []
for i in range(len(tok_corp_tweets)):
    relevant_newsid.append(matching_score(tok_corp_tweets[i]))


