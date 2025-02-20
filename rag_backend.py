from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS
import requests
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.docstore.document import Document
from langchain_community.utilities import ApifyWrapper
from langchain_cohere import CohereEmbeddings, CohereRagRetriever, ChatCohere
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes
# Load environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
API_KEY = '<your API key>'
# Check if environment variables are loaded properly
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set.")

# Initialize Apify client wrapper (if needed)
# apify = ApifyWrapper()

# Function to fetch articles from API endpoint and map to Document objects
def fetch_articles_and_map_to_documents():
    try:
        api_url = 'http://localhost:1337/api/articles?populate=*'
        headers = {'Authorization': f'Bearer {API_KEY}'}

        # Fetch data from API endpoint
        response = requests.get(api_url, headers=headers)
        data = response.json().get('data', [])

        # Map data to Document objects
        documents = []
        for item in data:
            document = Document(
                page_content=format_content(item.get('body', [])),
                metadata={"source": item.get('title', '')}
            )
            documents.append(document)

        return documents

    except Exception as e:
        print('Error fetching articles from API endpoint:', e)
        raise

# Function to format content from API response
def format_content(content):
    # Customize as per your content structure
    formatted_content = []
    for item in content:
        if item.get('type') == 'paragraph':
            formatted_content.append(item.get('children')[0].get('text'))
        elif item.get('type') == 'list':
            # Handle list formatting if needed
            pass
        # Add more conditions as per your content structure

    return '\n\n'.join(formatted_content)

# Define the embedding function using Cohere
embedding_function = CohereEmbeddings(cohere_api_key=cohere_api_key)

# Route to handle RAG QA queries
@app.route('/rag-qa', methods=['POST'])
async def rag_qa():
    data = request.get_json()
    user_query = data['question']
    chat_llm = ChatCohere(model="command-r")
    print(user_query)
    try:
        # Fetch articles from API endpoint and map to Document objects
        print("fetchting documents")
        documents = fetch_articles_and_map_to_documents()
        print("got the documents")
        
        print("Faissing documents")
        try:
            db = FAISS.from_documents(documents, embedding_function)
            print("similarity search")
            docs = db.similarity_search(user_query)
            # Query the vector store index for relevant documents
            results = docs[0].page_content
            
            results_metadata = docs[0].metadata['source']
            
            messages = [
                SystemMessage(content=f'please keep the response very short. {results}'),
                HumanMessage(content=user_query),
            ]
            llm_response = chat_llm.invoke(messages)
            print(f'llm_response: {llm_response}')
            print(results_metadata)
        except Exception as e:
            print(e)

        
        results = results[:200]
        return jsonify({'response': llm_response.content, 'metadata_title': results_metadata})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
