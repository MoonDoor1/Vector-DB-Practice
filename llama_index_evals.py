from llama_index.evaluation import DatasetGenerator, ResponseEvaluator, QueryResponseEvaluator
from llama_index import (SimpleDirectoryReader, 
                         ServiceContext, 
                         LLMPredictor, 
                         GPTVectorStoreIndex, 
                         load_index_from_storage, 
                         StorageContext)
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import pickle
import openai

#load vars from env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_documents(path):
    # Check if documents have already been loaded and saved
    if os.path.exists('documents.pkl'):
        print("Loading documents from cache")
        with open('documents.pkl', 'rb') as f:
            documents = pickle.load(f)
    else:
        print("Reading documents")
        reader = SimpleDirectoryReader(path)
        documents = reader.load_data()
        # Save the loaded documents to a file
        with open('documents.pkl', 'wb') as f:
            pickle.dump(documents, f)
    return documents

def generate_questions_and_save(documents, service_context):
    # Check if questions have already been generated and saved
    if os.path.exists('questions.pkl'):
        print("Loading questions from cache")
        with open('questions.pkl', 'rb') as f:
            questions = pickle.load(f)
    else:
        print("Generating questions")
        data_generator = DatasetGenerator.from_documents(documents, service_context=service_context)
        questions = data_generator.generate_questions_from_nodes()
        # Save the generated questions to a file
        with open('questions.pkl', 'wb') as f:
            pickle.dump(questions, f)
    return questions

def create_and_save_index(documents, index_id, persist_dir):
    # Check if index has already been created and saved
    if os.path.exists(f'{persist_dir}/{index_id}'):
        print("Loading index from storage")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, index_id=index_id)
    else:
        print("Creating index")
        index = GPTVectorStoreIndex.from_documents(documents)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir)
    return index


def get_responce_sourceCode(response, service_context):
    #get responce + source code 
    # define evaluator
    evaluator = ResponseEvaluator(service_context=service_context)

    # evaluate using the response object
    eval_result = evaluator.evaluate(response)
    return eval_result

def get_query_responce_sourceCode(questions, response, service_context):
    # define evaluator
    evaluator = QueryResponseEvaluator(service_context=service_context)

    # evaluate using the response object
    eval_result = evaluator.evaluate(questions, response)
    return eval_result

def get_query_responce_nodes(questions, response, service_context):
    # define evaluator
    evaluator = QueryResponseEvaluator(service_context=service_context)

    # evaluate using the response object
    eval_result = evaluator.evaluate_source_nodes(questions, response)
    return eval_result
def main():
    # Load documents
    documents = load_documents(".Data")

    print("Setting up LLM predictor")
    # Setup GPT-4 model
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # Generate questions
    questions = generate_questions_and_save(documents, service_context)

    print("Questions Generated!")

    # Create index 
    index = create_and_save_index(documents, "vector_index", 'storage')

    print("Index Built")

    # Query the index
    query_engine = index.as_query_engine(similarity_top_k=3, service_context=service_context)
    response = query_engine.query(questions[0])

    print("Responses Built")

    print(response)

    # Call the new functions and print their responses
    print("Evaluating response source code")
    eval_result = get_responce_sourceCode(response)
    print(eval_result)

    print("Evaluating query response source code")
    eval_result = get_query_responce_sourceCode(questions[0], response, service_context)
    print(eval_result)

    print("Evaluating query response nodes")
    eval_result = get_query_responce_nodes(questions[0], response, service_context)
    print(eval_result)

if __name__ == "__main__":
    main()