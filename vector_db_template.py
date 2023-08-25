# Import necessary libraries
import os
import openai
import pickle
#import vector db
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

    pass


    #return index

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
def embed_data(nodes_by_document, index):
    pinecone_vectors = []
    print("starting embeddings")

    for filename, nodes in nodes_by_document.items():
        for i, node in enumerate(nodes):
            doc = node.text
            
            embedding = openai.Embedding.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            vector = embedding["data"][0]["embedding"]
            
            ### CREATE VECTORS and APPEND####
            # pinecone_vector = (str(i), vector, {"text": doc, "source": filename})
            # pinecone_vectors.append(pinecone_vector)

    print('finished embeddings starting upsert')
    # UPSERT Vectors to db
    # upsert_response = index.upsert(vectors=pinecone_vectors)
    print("upsert finished, Length of pinecone vectors", len(pinecone_vectors))

def main():
    print("initlizing db.. ")
    index = initialize_db()

    data = load_data(data_dir)
    nodes = get_nodes(data)

    print("Embeding and upseting data...")
    embed_data(nodes, index)
    print("upsert successful!")

if __name__ == "__main__":
    main()