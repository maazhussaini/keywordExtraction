import spacy
import string
import pandas as pd
import os 
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from langdetect import detect
from translate import Translator
import re
# import networkx as nx

nlp = spacy.load("en_core_web_sm")
stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
punctuation = string.punctuation


def extract_keywords(data):
    try:
        text = data.lower()
        result = []
        try:
            language = detect(text)
            if language != "en":
                translator = Translator(to_lang="en")
                text = translator.translate(text)
            
        except:
            print("Error: Could not detect language or translate text.")
        
        doc = nlp(text)
        for token in doc:
            if not (token.is_stop or token.is_punct):
                result.append(token.text)
        
        
        return result
    except Exception as err:
        print(str(err))

def word_cloud(word_count):
    wordcloud = WordCloud(width=800, height=800, 
                    min_font_size = 10).generate_from_frequencies(word_count)

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

def read_file():
    path = 'Datasets/'
    files = os.listdir(path)
    count = 0
    text = ""
    for file in files:
        if file.endswith(".csv"):
            try:
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                df = df.dropna(subset=['description'])
                descriptions = df['description'].astype(str).tolist()
                for desc in descriptions:
                    desc = desc.lower()
                    desc = "".join(word for word in desc if word not in punctuation)
                    desc = " ".join(word for word in desc.split() if word not in stopwords)
                    text += desc + " "
                
                if count == 2:
                    break
                count += 1

            except Exception as err:
                print("ERROR - "+str(file)+" - "+str(err))
    text = re.sub(r'\\u.+?(?=\s)', '', text)
    return text

text = read_file()

keywords = extract_keywords(text)
word_count = Counter(keywords)
word_cloud(word_count)
print(word_count.most_common(10))

word_count_json = json.dumps(word_count, indent=4)
with open('Export/word_count.json', 'w') as f:
    f.write(word_count_json)
    