# Import necessary libraries
import os
import openai
import pickle
import lancedb
import re
import pickle
import requests
import zipfile
from pathlib import Path
#import vector db
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

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
# def embed_data(nodes_by_document):
#     print("Text splitter...")
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )

#     print(nodes_by_document)
#     # documents = text_splitter.split_documesnts(nodes_by_document)
#     embeddings = OpenAIEmbeddings()
    
#     return documents

def embed_data(nodes_by_document):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    for file_name, nodes in nodes_by_document.items():
        for node in nodes:
            text = node.text
            metadata = node.metadata
            # Using create_documents on a single text, so passing [text] as an argument
            chunked_texts = text_splitter.create_documents([text], [metadata])
            for chunk_text in chunked_texts:
                chunks.append(chunk_text)

    print(chunks)
    return chunks


def create_database(documents):
    embeddingsfunction = OpenAIEmbeddings()
    db = lancedb.connect('lancedb')
    table = db.create_table("pandas_docs", data=[
        {"vector": embeddingsfunction.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")
    docsearch = LanceDB.from_documents(documents, embeddingsfunction, connection=table)

    return docsearch
    
    

def queryDocs(question, docsearch):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    query = "1. How has the practice of high frequency trading, as described in Michael Lewis's Flash Boys, contrasted with Lindsell Train Limited's investment approach?"
    print(qa.run(query))

def main():
    print("Initializing db.. ")
    initialize_db()

    print("Loading data...")
    data = load_data(data_dir)

    print("Getting nodes...")
    nodes = get_nodes(data, 'newdata.pkl')

    print("Embedding and upserting data...")
    embededData = embed_data(nodes)
    print("Upsert successful!")

    print("set up the database")
    docsearch = create_database(embededData)

    print("Querying documents...")
    question = "What are the major differences in pandas 2.0?"
    queryDocs(question, docsearch)

if __name__ == "__main__":
    main()