import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import models, corpora, similarities

file_encoding = 'cp1252'
ndc = pd.read_csv('C:/Users/mat/Desktop/Kurse/!_Pythonkurs/Python/ndc1.csv', encoding=file_encoding,delimiter=";")
#print(ndc.shape)
ndc1=ndc[['File']]
#print(ndc1['File'])

ndc = ndc[['Text']]#

#print(ndc.head(3))

cleanedndc = []
for paragraph in ndc.Text:
    
    # Tokenize the paragraph
    tokens = nltk.word_tokenize(str(paragraph))
    
    # Make words into lowercase and remove punctuation
    words = [word.lower() for word in tokens if word.isalpha()]

    stop_words = stopwords.words('english')
    words = [word for word in words if not word in stop_words]

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    ndcsma=[]
    ndcsma.append(' '.join(words))
    cleanedndc.append(ndcsma)
model = models.Word2Vec(cleanedndc)
print(model.wv.most_similar('climate'))
#cleanedndc = pd.DataFrame(cleanedndc, columns=['text'])

#cleanedndc.index = range(len(cleanedndc))
#print(cleanedndc.head())

#allWords = []
#for paragraph in cleanedndc.text:
#    allWords = allWords + paragraph.split()
#print(allWords)