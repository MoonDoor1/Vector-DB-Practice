# Import necessary libraries
import os
import openai
#import vector db
import pickle
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import PointStruct
from qdrant_openapi_client.models.models import Filter, FieldCondition, Range



# Load environment variables
load_dotenv()
data_dir = r"Data"
openai.api_key = os.getenv("OPENAI_API_KEY")
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://ee2934d7-8dc6-4ef0-a5c3-bd7926868fd2.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Set up extractor and parser
entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=True,
    device="cpu",
)

metadata_extractor = MetadataExtractor(extractors=[entity_extractor])
node_parser = SimpleNodeParser.from_defaults(metadata_extractor=metadata_extractor)

# Load the data
def load_data(data_dir):
    if not os.path.exists(data_dir):
        raise Exception(f"Directory {data_dir} does not exist")

    pdf_files = [os.path.join(data_dir, pdf_file) for pdf_file in os.listdir(data_dir) if pdf_file.endswith('.pdf')]
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()


    return documents

# Initialize the database
def initialize_db(client):
    #init db 
    qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing,

    from qdrant_client.models import Distance, VectorParams

    client.recreate_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )


    return client

def get_nodes(documents, filename='nodes.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            nodes_by_document = pickle.load(handle)
    else:
        nodes_by_document = {}
        nodes = node_parser.get_nodes_from_documents(documents)

        for node in nodes:
            file_name = node.metadata['file_name']

            if file_name not in nodes_by_document:
                nodes_by_document[file_name] = []

            nodes_by_document[file_name].append(node)

            # Extract entities
            for entity in node.metadata.get('entities', []):
                # Save the entities as metadata
                node.metadata[entity] = entity

            # Print the available attributes and methods of the node
            print(dir(node))

            # Print entities for each node
            print(f"Entities for node {node.id_}: {node.metadata.get('entities')}")

        with open(filename, 'wb') as handle:
            pickle.dump(nodes_by_document, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return nodes_by_document

# Embed data and upsert it to Pinecone
def embed_data(nodes_by_document, client):
    print("starting embeddings")

    points = []  # List to store PointStruct objects

    for filename, nodes in nodes_by_document.items():
        for i, node in enumerate(nodes):
            doc = node.text
            
            embedding = openai.Embedding.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            vector = embedding["data"][0]["embedding"]

            properties = {
                "text": doc,
                "filename": filename,
            }

            print(f"importing document: {i+1}")

            # Create a PointStruct object and append it to the list
            points.append(PointStruct(
                id=i,
                vector=vector,  # directly use vector
                payload=properties
            ))

    print('finished embeddings starting upsert')

    # Upsert the data to Qdrant
    client.upsert(
        collection_name="my_collection",
        points=points
    )

# Usage:
# semantic_search("some query", qdrant_client)
def semantic_search(query, client, limit=3):
    # Encode the query into a vector using OpenAI
    print('semantic search called')
    embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_vector = embedding["data"][0]["embedding"]

    # Perform the search
    hits = client.search(
        collection_name="my_collection",
        query_vector=query_vector,
        limit=limit
    )

    # Print the results
    for hit in hits:
        print(hit.payload, "score:", hit.score)



    

def main():
    print("initlizing db.. ")
    index = initialize_db(qdrant_client)

    data = load_data(data_dir)
    nodes = get_nodes(data)

    print("Embeding and upseting data...")
    embed_data(nodes, index)
    print("upsert successful!")

    semantic_search("Michael Lewis's Flash Boys", qdrant_client)

if __name__ == "__main__":
    main()