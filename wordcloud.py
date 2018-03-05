# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 02:23:15 2018

@author: Faishal
"""
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#input to dataframe
outfile = "D:\\File\\"
dataset = pd.DataFrame.from_csv(outfile + 'ds_asg_data.csv')

#delete row NA
dataset = dataset.dropna()
dataset.isnull().sum()

#List of topic
topic = dataset.groupby('article_topic').size()

#Split text and target
sentence = dataset['article_content'] #text
y = dataset['article_topic'] #target

#preprocessing by regex
sentence = sentence.str.lower()
sentence = sentence.str.replace(r"[^a-zA-Z0-9]+"," ")
sentence = sentence.str.replace(r"([^\w])"," ") 
sentence = sentence.str.replace(r"\b\d+\b", " ")
sentence = sentence.str.replace(r"\s+|\r|\n", " ")
sentence = sentence.str.replace(r"^\s+|\s$", "")

#dataset['article_topic'] = sentence

#stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stopword
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# stemming and stopword process
X = []
index = 1

for item in sentence:
    print('data nomor: {}'.format(index))
    item = stemmer.stem(item)
    item = stopword.remove(item)

    X.append(item)
    index = index + 1
    
X = pd.Series(X)

#if not using stemming and stopword
X = sentence #text

stopwords = set(STOPWORDS)

def show_wordcloud(Stemming, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(Stemming))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(X)
