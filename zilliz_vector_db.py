# Import necessary libraries
import os
import openai
import pickle
from pymilvus import MilvusClient
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
data_dir = r"Data"
openai.api_key = os.getenv("OPENAI_API_KEY")

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
def initialize_db():
    #init db 

    # Initialize a MilvusClient instance
    client = MilvusClient(
        uri="https://in03-e5133527d396b05.api.gcp-us-west1.zillizcloud.com", # Cluster endpoint obtained from the console
        # - For a serverless cluster, use an API key as the token.
        # - For a dedicated cluster, use the cluster credentials as the token
        # in the format of 'user:password'.
            token = os.getenv("ZILLIZ_API_KEY")
    )

    # Create a collection
    client.create_collection(
        collection_name="investory_reports",
        dimension=1536
    )

    res = client.describe_collection(
        collection_name='investory_reports'
    )

    print(res)

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


        with open(filename, 'wb') as handle:
            pickle.dump(nodes_by_document, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return nodes_by_document

# Embed data and upsert it to Pinecone
def embed_data(nodes_by_document, client):
    print("starting embeddings")

    for file_name, nodes in nodes_by_document.items():
        data = []
        for i, node in enumerate(nodes):
            doc = node.text
            
            # Use OpenAI's API to generate the embedding
            embedding = openai.Embedding.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            vector = embedding["data"][0]["embedding"]

            data.append({ 
                'id' : i,  # Ensure id is an integer
                'vector': vector,  # No need to convert to list
                'filename': node.metadata['file_name']
            })

        # Insert multiple entities
        res = client.insert(
            collection_name="investory_reports",
            data=data
        )

        return vector

def semanticSearch(client, text_to_search, **searchFields):
    # Use OpenAI's API to generate the embedding for the text to search
    embedding_to_search = openai.Embedding.create(
        input=text_to_search,
        model="text-embedding-ada-002"
    )
    vector_to_search = embedding_to_search["data"][0]["embedding"]

    print(client.name)

    res = client.search(
        collection_name="investory_reports",
        data=[vector_to_search],  # Use the generated vector to perform the search
        output_fields=list(searchFields.values())
    )

    print(res)

def main():
    print("Initializing db.. ")
    client = initialize_db()
    print(client.name)


    # data = load_data(data_dir)
    # nodes = get_nodes(data)

    # print("Embedding and upserting data...")
    # vector = embed_data(nodes, client)
    # print("Upsert successful!")


    #perfom semantic search
    # print("performing semantic search ")
    # semanticSearch(client, "Michael Lewis's Flash Boys", field1="id", field2="vector")
    # semanticSearch(client, "Limited's investment approach", field1="id", field2="vector")

if __name__ == "__main__":
    main()