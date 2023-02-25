# Introduction
The code provided is a Python script that extracts keywords from a pandas DataFrame, clusters the keywords based on their semantic similarity using the K-means clustering algorithm, and stores the resulting keyword clusters in a MongoDB database. The purpose of this code is to organize and analyze large amounts of text data efficiently.

# Problem Statement
In today's era, organizations produce and collect a vast amount of textual data such as emails, reports, social media feeds, and customer feedback, among others. Analyzing and organizing such a large amount of data can be a daunting task. Hence, there is a need for automated techniques to extract meaningful insights and themes from text data efficiently. The code provided addresses this challenge by extracting and clustering keywords from text data, making it easier to organize and analyze large amounts of textual data.

# Code
The code consists of three main functions: `extract_keywords()`, `cluster_keywords()`, and `main()`.
* <h3>Extract Keywords</h3>
It takes a Pandas DataFrame as input and returns a list of unique keywords in the DataFrame. It uses the NLTK library to tokenize the text data, convert the words to lowercase, and remove stop words.
```python
def extract_keywords(df: pd.DataFrame) -> List[str]:
    """
    Extracts keywords from a pandas DataFrame.
    Returns a list of unique keywords in the DataFrame.
    """
    keywords = []
    for column in df.columns:
        for cell in df[column]:
            words = word_tokenize(str(cell).lower())
            words = [word for word in words if word.isalpha() and word not in STOPWORDS]
            keywords.extend(words)
    keywords = list(set(keywords))
    return keywords

```

* <h3>Cluster Keywords</h3>
The cluster_keywords function first creates a TF-IDF vectorizer object, which is used to convert the extracted keywords into vectors. TF-IDF stands for Term Frequency-Inverse Document Frequency and is a commonly used technique in NLP to weigh the importance of each word in a document. The vectorizer also removes stop words such as 'the', 'and', 'is' etc., which do not add any useful information.
Next, the vectorized keywords are clustered using the K-Means algorithm, which is a popular unsupervised learning technique used for clustering similar data points. The n_clusters parameter specifies the number of clusters to be formed. This value can be set depending on the dataset and the specific needs of the analysis.
After clustering, the keywords are assigned labels indicating which cluster they belong to. These labels are stored in the cluster_labels variable. Using these labels, a dictionary is created that stores the keywords in each cluster. Finally, the dictionary is converted into a list of lists where each inner list represents a cluster of similar keywords.
The keyword_clusters variable is returned which contains the list of clusters of similar keywords.

This function reads files from a directory 'Datasets/' and concatenates the contents of the 'description' column of each file. The function first retrieves a list of all files in the directory and loops through each file. If the file ends with '.csv', it reads the file using the pandas library and adds the contents of the 'description' column to the text variable. The function also removes punctuation and stopwords from each description before concatenating them. Finally, the function returns the concatenated text.
```Python
def cluster_keywords(keywords: List[str]) -> List[List[str]]:
    """
    Clusters the given keywords into multiple clusters based on their semantic similarity.
    Returns a list of keyword clusters, where each cluster is a list of similar keywords.
    """
    # Create a TF-IDF vectorizer to convert keywords into vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(keywords)

    # Use KMeans algorithm to cluster the keywords
    km = KMeans(n_clusters=NUM_CLUSTERS, n_init='auto')
    km.fit(X)

    # Get the cluster labels for each keyword
    cluster_labels = km.labels_

    # Create a dictionary to store the keywords in each cluster
    clusters = {}
    for i in range(NUM_CLUSTERS):
        cluster_keywords = [keywords[j] for j in range(len(keywords)) if cluster_labels[j] == i]
        clusters[f"Cluster {i+1}"] = cluster_keywords

    # Convert the dictionary to a list of lists
    keyword_clusters = [v for k, v in clusters.items()]

    return keyword_clusters
```

* <h3>Main Function</h3>
The main function reads CSV files from a given directory, extracts and clusters the keywords from each file, and stores the resulting clusters in MongoDB. The function first establishes a connection to MongoDB and initializes a collection to store the keyword clusters. It then iterates over each CSV file in the given directory, reads the file into a pandas DataFrame, extracts and clusters the keywords, and stores the resulting clusters in MongoDB as a document associated with the file. The function handles any errors that occur during the process and logs them using the Python logging module.
```Python
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", help="Path to directory containing CSV files")
    args = parser.parse_args()
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # Iterate over CSV files in the directory
    for file_name in os.listdir(args.dir_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(args.dir_path, file_name)

            # Read CSV file into a pandas DataFrame
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logging.error(f"Error reading CSV file {file_name}: {e}")
                continue

            # Extract keywords from the DataFrame
            try:
                keywords = extract_keywords(df)
            except Exception as e:
                logging.error(f"Error extracting keywords from {file_name}: {e}")
                continue

            # Cluster the keywords
            try:
                keyword_clusters = cluster_keywords(keywords)
            except Exception as e:
                logging.error(f"Error clustering keywords from {file_name}: {e}")
                continue

            # Add the keyword clusters to MongoDB
            try:
                collection.update_one({"file_name": file_name}, {"$set": {"keyword_clusters": keyword_clusters}}, upsert=True)
                logging.info(f"Keyword clusters for {file_name} added to MongoDB")
            except Exception as e:
                logging.error(f"Error adding keyword clusters for {file_name} to MongoDB: {e}")
                continue
```

# Future Enhancement:
The code provided could be further enhanced to include additional features, such as:

1. Visualizing keyword clusters: The keyword clusters could be visualized using tools such as word clouds, network graphs, or dendrograms, making it easier to interpret and communicate the results.

2. Improving clustering accuracy: The clustering accuracy could be improved by experimenting with different vectorization techniques, clustering algorithms, and parameters such as the number of clusters.

3. Scaling the process: The process could be scaled to handle larger amounts of data by parallelizing the keyword extraction and clustering process using distributed computing frameworks such as Apache Spark.

# Conclusion:
The code provided is a valuable tool for extracting and clustering keywords from text data, making it easier to organize and analyze large amounts of textual data. The code is modular, scalable, and extensible, making it suitable for a wide range of text data analysis tasks. The code could be further enhanced by adding additional features and scaling the process to handle larger amounts of data.
