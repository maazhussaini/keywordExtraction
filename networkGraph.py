import os
import logging
import pymongo
import argparse
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# -------

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
STOPWORDS = set(stopwords.words('english'))
MONGO_URI = 'mongodb://localhost:27017/'
MONGO_DB = 'mydatabase'
MONGO_COLLECTION = 'mycollection'
NUM_CLUSTERS = 10

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


def create_network_graph(keyword_clusters: List[List[str]]) -> None:
    """
    Creates a network graph of the given keyword clusters.
    """
    # Create a new graph
    G = nx.Graph()

    # Add nodes to the graph
    for cluster in keyword_clusters:
        for keyword in cluster:
            G.add_node(keyword)

    # Add edges to the graph
    for i in range(len(keyword_clusters)):
        for j in range(i+1, len(keyword_clusters)):
            for keyword1 in keyword_clusters[i]:
                for keyword2 in keyword_clusters[j]:
                    if not G.has_edge(keyword1, keyword2):
                        G.add_edge(keyword1, keyword2)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()

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
            
            # Network graph
            try:
                create_network_graph(keyword_clusters)
            except Exception as e:
                logging.error(f"Error Network Graph from {file_name}: {e}")
                continue

            # Add the keyword clusters to MongoDB
            try:
                collection.update_one({"file_name": file_name}, {"$set": {"keyword_clusters": keyword_clusters}}, upsert=True)
                logging.info(f"Keyword clusters for {file_name} added to MongoDB")
            except Exception as e:
                logging.error(f"Error adding keyword clusters for {file_name} to MongoDB: {e}")
                continue


def delete_all_document():
    # connect to MongoDB
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    

    # delete all documents from the collection
    result = collection.delete_many({})

    # print the number of deleted documents
    print(result.deleted_count, " documents deleted.")

# main()
delete_all_document()