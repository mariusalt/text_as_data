import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


countMatrix = pd.read_csv("wordcount.csv")
file_encoding = 'cp1252'
ndc = pd.read_csv('C:/Users/mat/Desktop/Kurse/!_Pythonkurs/Python/ndc1.csv', encoding=file_encoding,delimiter=";")

countMatrix=countMatrix.drop(['e'], axis=1)
countMatrix=countMatrix.drop(['n'], axis=1)
countMatrix=countMatrix.drop(['r'], axis=1)
countMatrix=countMatrix.drop(['c'], axis=1)
countMatrix=countMatrix.drop(['p'], axis=1)
countMatrix=countMatrix.drop(['l'], axis=1)
countMatrix['emission'] = countMatrix['emissions'] + countMatrix['emission'] 
countMatrix=countMatrix.drop(['File'], axis=1)
print(countMatrix.head(5))
print(countMatrix.sum(axis=0).sort_values(ascending=False).head(30))


#tf = countMatrix.copy()
#tf=tf.sum(axis=0).sort_values(ascending=False).head(10)
#tmp = tf > 0
#print(tmp.shape)
#df = tmp.sum(axis=0)
#n = len(tf)
#idf = np.log(n/(df)) 
#tfidf = tf*idf
#tfidf.sum().sort_values(ascending=False).head(10)


#print(cosine_similarity(sentence_m.reshape(1,-1), sentence_m.reshape(1,-1))[0][0])
#print(cosine_similarity(sentence_m.reshape(1,-1), sentence_h.reshape(1,-1))[0][0])
#print(cosine_similarity(sentence_m.reshape(1,-1), sentence_w.reshape(1,-1))[0][0])

count_vect = CountVectorizer()

no_features = 1000
tfidf_vectorizer = TfidfVectorizer(max_features=no_features,stop_words='english')

for v,paragraph in enumerate(ndc.Text):
    ndc.Text[v] = str(paragraph)

tfidf = tfidf_vectorizer.fit_transform(ndc.Text)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print(tfidf.shape)

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print( "Topic %d:" % (topic_idx) )
        print( " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


no_topics = 10

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, learning_method='online').fit(tfidf)

# Get the topic distribution (distribution over words, (n_topics x n_words) )
topic_distribution_lda = lda.components_

# Get the document distribution (distribution over topics)
doc_distribution_lda = lda.transform(tfidf)
display_topics(lda, tfidf_feature_names, 10)