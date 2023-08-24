# Import necessary libraries
import os
import openai
import weaviate
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

import requests
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
def initialize_db(delete_db):
    #init db 
    client = weaviate.Client(
        url="https://weaviate001-jily9a6t.weaviate.network",  
        auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")), 
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY") 
        }
    )

    class_name = "updated_documents".lower()

    # Delete the class if it already exists
    if delete_db:
        if class_name in (k['class'].lower() for k in client.schema.get()['classes']):
            client.schema.delete_class(class_name)

    # Check if the class already exists
    if class_name not in (k['class'].lower() for k in client.schema.get()['classes']):
        print("Class {class_name} doesnt exsist yet creating it..")
        class_obj = {
            "class": class_name,
            "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            "moduleConfig": {
                "text2vec-openai": {},
                "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
            },
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                },
                {
                    "name": "filename",
                    "dataType": ["text"],
                },
            ]
        }

        client.schema.create_class(class_obj)

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

    client.batch.configure(batch_size=100)  # Configure batch

    with client.batch as batch:  # Configure a batch process
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
                    "entities": node.metadata.get('entities')
                }

                print(f"importing document: {i+1}")
                response = batch.add_data_object(
                    data_object=properties,
                    class_name="updated_documents",
                    vector=vector  # Add custom vector
                )
                print(response)

    print('finished embeddings starting upsert')


#performs similarity search on our vectors
def similarity_search(client, concepts):
    response = (
        client.query
        .get("updated_documents", ["text", "filename"])
        .with_near_text({"concepts": concepts})
        .with_limit(2)
        .do()
    )

    return json.dumps(response, indent=4)

# Performs generative search on your vectors
def generative_search(client, concepts, prompt):
    response = (
        client.query
        .get("updated_documents", ["text", "filename"])
        .with_near_text({"concepts": concepts})
        .with_generate(single_prompt=prompt)
        .with_limit(2)
        .do()
    )

    return response

def get_all_documents(client):
    response = (
        client.query
        .get("updated_documents", ["text", "filename"])
        .with_limit(100)  # Adjust this number based on how many documents you expect
        .do()
    )

    return json.dumps(response, indent=4)


def generative_search(client, concepts, prompt):
    response = (
        client.query
        .get("updated_documents", ["text", "filename"])
        .with_near_text({"concepts": concepts})
        .with_generate(grouped_task="Write a tweet with emojis about these facts. Max character count of 500")
        .with_limit(2)
        .do()
    )

    return json.dumps(response, indent=4)


def main():
    # print("initlizing db.. ")
    # client = initialize_db(True)

    # data = load_data(data_dir)
    # nodes = get_nodes(data)

    # print("Embeding and upseting data...")
    # embed_data(nodes, client)
    # print("upsert successful!")

    client = initialize_db(False)
    # concepts = ["high frequency trading", "Lindsell Train Limited's investment approach"]
    # print(similarity_search(client, concepts))

    # Generative search
    # concepts = ["average turnover rate", "implied holding period", "Lindsell Train Limited's Global Equity Fund"]
    # prompt = "Explain {text} as you might to a five-year-old."
    # print(generative_search(client, concepts, prompt))

    # Generative search
    concepts = ["high frequency trading", "Michael Lewis's Flash Boys", "Lindsell Train", "Limited's investment approach"]
    prompt = "How has the practice of high frequency trading, as described in Michael Lewis's Flash Boys, contrasted with Lindsell Train Limited's investment approach?"

    response = generative_search(client, concepts, prompt)
    response_dict = json.loads(response)  # Convert the JSON string to a dictionary

    print(response_dict)
    
    #get all queery
    # client = initialize_db(False)
    # print(get_all_documents(client))


if __name__ == "__main__":
    main()
