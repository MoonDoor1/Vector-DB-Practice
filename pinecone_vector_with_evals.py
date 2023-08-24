# Import necessary libraries
# import ...
import os
import openai
import pickle
from dotenv import load_dotenv
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    EntityExtractor,
)
#llama imports 
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import SimpleDirectoryReader
from llama_index import (SimpleDirectoryReader, 
                         ServiceContext, 
                         LLMPredictor, 
                         GPTVectorStoreIndex)
from langchain.chat_models import ChatOpenAI
import pinecone


from llama_index_evals import get_query_responce_nodes, generate_questions_and_save, create_and_save_index

#load env vars
load_dotenv()
data_dir = r"Data"
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    )

service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=3000
    )



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
#evalualte
def evaluate_vector_database(tessting_index, questions, responses, service_context):
    # Initialize the query engine
    query_engine = tessting_index.as_query_engine(similarity_top_k=3, service_context=service_context)

    i = 0
    for question in questions:
        # Evaluate the response
        eval_result = get_query_responce_nodes(question, responses[i], service_context)

        # Print the evaluation result
        print(eval_result)
        i += 1

def generate_questions_and_responses(documents, service_context, index):
    # Generate questions
    print("generating questions..")
    questions = generate_questions_and_save(documents, service_context)
    print("questions generated", questions[:3])

    # Create query engine
    print("creating query index")
    query_engine = index.as_query_engine(similarity_top_k=3, service_context=service_context)

    # Check if responses file exists
    responses_file = 'responses.pkl'
    if os.path.exists(responses_file):
        # Load responses from file
        print("loading responses from file")
        with open(responses_file, 'rb') as f:
            responses = pickle.load(f)
    else:
        # Generate responses
        print("generating responses")
        responses = []
        for i, question in enumerate(questions):
            print(f"Generating response for question {i+1} of {len(questions)}")
            response = query_engine.query(question)
            print(f"Response: {response}")
            responses.append(response)

        # Save responses to a file
        print("saving responses to file")
        with open(responses_file, 'wb') as f:
            pickle.dump(responses, f)

    print("responses generated", responses[:3])

    return questions, responses

def main():
    # Initialize the database
    # print("initlizing db.. ")
    # index = initialize_db()

    # # Load your data
    # # print("loading and preparing data..")
    # data = load_data(data_dir)  # Replace with your data loading code
    # nodes = get_nodes(data)

    # #embeds the data and upserts it to pinecone
    # print("Embeding and upseting data...")
    # embed_data(nodes, index)
    # print("upsert successful!")

    # # Generate questions
    # tessting_index = create_and_save_index(data, "vector_index", 'storage')
    # questions,responses = generate_questions_and_responses(data, service_context, tessting_index)  # Replace with your question generation code
    

    # # Evaluate performance
    # print("Evaluating performance..")
    # evaluate_vector_database(tessting_index, questions, responses, service_context)
    # Load questions from pickle file
    # Load questions from pickle file
    with open('questions.pkl', 'rb') as f:
        questions = pickle.load(f)

        # Print the first 5 questions
        for i in range(5):
            print(questions[i])

    # Load responses from pickle file
    with open('responses.pkl', 'rb') as f:
        responses = pickle.load(f)

        # Print the first 5 responses
        for i in range(5):
            print(responses[i])

if __name__ == "__main__":
    main()