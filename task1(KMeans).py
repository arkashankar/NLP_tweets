# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:49:17 2019

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