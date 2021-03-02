import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download()

file_encoding = 'cp1252'
ndc = pd.read_csv('C:/Users/mat/Desktop/Kurse/!_Pythonkurs/Python/ndc1.csv', encoding=file_encoding,delimiter=";")
print(ndc.shape)
ndc1=ndc[['File']]
print(ndc1['File'])

ndc = ndc[['Text']]

print(ndc.head(3))

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

    
    cleanedndc.append(' '.join(words))
cleanedndc = pd.DataFrame(cleanedndc, columns=['text'])

cleanedndc.index = range(len(cleanedndc))
print(cleanedndc.head())

allWords = []
for paragraph in cleanedndc.text:
    allWords = allWords + paragraph.split()
print("done1")   
uniqueWords = set(allWords)
print("done2")  
# Define an empty matrix
countMatrix = pd.DataFrame([], columns=list(uniqueWords), index=range(len(cleanedndc)))
print("done3")  

for i in range(len(cleanedndc)):
    for word in countMatrix.columns:
        countMatrix[word][i] = cleanedndc['text'][i].split().count(word)
        print("done4" + str(i))  
# use the document date as the index 
# we use a Pandas datetime parser so Python can understand the dates

dates = ndc1['File']
countMatrix.index = dates
print(countMatrix.head())
print(countMatrix.shape)
countMatrix.to_csv('wordcount.csv', encoding='utf-8')


