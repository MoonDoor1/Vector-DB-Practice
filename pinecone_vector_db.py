# Import necessary libraries
# import ...
import os
import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
#llama imports 
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader
import pinecone

#load env vars
load_dotenv()
data_dir = r"Data"
openai.api_key = os.getenv("OPENAI_API_KEY")

#Set up extractor and parser
entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=True,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

#define extractor 
metadata_extractor = MetadataExtractor(extractors=[entity_extractor])
#define node parser
node_parser = SimpleNodeParser.from_defaults(metadata_extractor=metadata_extractor)

#load the data
def load_data(data_dir):
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise Exception(f"Directory {data_dir} does not exist")

    # List all PDF files in the data directory
    pdf_files = [os.path.join(data_dir, pdf_file) for pdf_file in os.listdir(data_dir) if pdf_file.endswith('.pdf')]

    # Load the PDF files using SimpleDirectoryReader
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    

    return documents
# Initialize the database
def initialize_db():
    #Init the pinecone index   
    pinecone.init(      
        api_key= os.getenv("PINECONE_API_KEY"),      
        environment='us-west1-gcp-free'      
    ) 

    index_name = 'pinecone001'
    index = pinecone.Index(index_name)

    return index

def get_nodes(documents):
    nodes_by_document = {}

    nodes = node_parser.get_nodes_from_documents(documents)


    for node in nodes:
        file_name = node.metadata['file_name']

        if file_name not in nodes_by_document:
            nodes_by_document[file_name] = []

        nodes_by_document[file_name].append(node)

    return nodes_by_document

#using open AI embeddings to embed our chunked data.
def embed_data(nodes_by_document, index):
    """Embeds data using a language model.

    Args:
        nodes_by_document: A dictionary where each key is a filename and each value is a list of Node objects.
    """
    pinecone_vectors = []
    print("starting embeddings")

    for filename, nodes in nodes_by_document.items():
        for i, node in enumerate(nodes):
            doc = node.text  # Extracting the text content from the Node object
            
            embedding = openai.Embedding.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            vector = embedding["data"][0]["embedding"]
            
            # Create a Pinecone vector with the embedding and additional metadata
            pinecone_vector = (str(i), vector, {"text": doc, "source": filename})
            pinecone_vectors.append(pinecone_vector)

    print('finished embeddings starting upsert')
    # All vectors can be upserted to Pinecone in one go
    upsert_response = index.upsert(vectors=pinecone_vectors)
    print("upsert finished, Length of pinecone vectors", len(pinecone_vectors))

def main():
    # Initialize the database
    print("initlizing db.. ")
    index = initialize_db()

    # Load your data
    # print("loading and preparing data..")
    data = load_data(data_dir)  # Replace with your data loading code
    nodes = get_nodes(data)

    #embeds the data and upserts it to pinecone
    print("Embeding and upseting data...")
    embed_data(nodes, index)
    print("upsert successful!")


if __name__ == "__main__":
    main()