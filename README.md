<h2>Introduction</h2>
<pre>
This code is for generating a word cloud from a set of CSV files. The code contains functions for extracting keywords from the data, counting the frequency of each keyword, and generating a word cloud from the frequency data. The code is written in Python and uses the spacy,  pandas, matplotlib, and wordcloud libraries.
<pre>

<h2>Problem Statement</h2>
The goal of this code is to process the text data in a set of CSV files, extract relevant keywords, count the frequency of each keyword, and generate a word cloud to visualize the most frequent keywords.

<h2>Code</h2>
<ul>
Loading the libraries and initializing variables
</ul>
Here, we are loading the spacy library with the `en_core_web_sm` model, which is a small English language model for NLP tasks. We are also creating a set of stop words and punctuation from the spacy library. Stop words are words that are commonly used in a language but do not contribute much to the meaning of a sentence. Punctuation marks are symbols used to separate words, sentences, and clauses.


nlp = spacy.load("en_core_web_sm")
stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
punctuation = string.punctuation



Function to extract keywords
This function takes a single argument data which is a string of `text`. The first step is to convert the text to lowercase. Then, it tries to detect the language of the text using the detect function and translate it to English if it is not already in English. After that, the `nlp` model is used to process the `text` and tokenize it into individual words. The loop then goes through each token and only keeps the tokens that are not stop words or punctuation marks. The list of filtered tokens is returned as the `result`.
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

Function to generate word cloud
This function is used to visualize the frequency of keywords in a word cloud format. It takes a dictionary object `word_count` as input, which contains the keywords and their frequency. The function creates a word cloud object using the WordCloud class from the wordcloud library. The function sets the size of the figure, generates the word cloud using the frequency information, and sets the axis to 'off' to remove the axis labels. Finally, it displays the word cloud using the show method of the Matplotlib library.
def word_cloud(word_count):
    wordcloud = WordCloud(width=800, height=800,
                    min_font_size = 10).generate_from_frequencies(word_count)


    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
   
    plt.show()
Function to read files.
This function reads files from a directory 'Datasets/' and concatenates the contents of the 'description' column of each file. The function first retrieves a list of all files in the directory and loops through each file. If the file ends with '.csv', it reads the file using the pandas library and adds the contents of the 'description' column to the text variable. The function also removes punctuation and stopwords from each description before concatenating them. Finally, the function returns the concatenated text after removing unicode characters.


def read_file():
    path = 'Datasets/'
    files = os.listdir(path)
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


            except Exception as err:
                print("ERROR - "+str(file)+" - "+str(err))


    return text

Future Enhancement:
The keyword extraction feature can be used as a feature in Natural Language Processing (NLP) and Machine Learning (ML) tasks such as sentiment analysis, text classification, named entity recognition, and others.
The code can be enhanced to include NLP and ML techniques to perform tasks such as topic modeling and document clustering on the extracted keywords.
The code can be extended to work with larger datasets and other text-based sources, such as social media platforms and news articles.
The extracted keywords can also be used as input to other NLP and ML algorithms, such as word embeddings and word association rules, to generate more meaningful insights.

