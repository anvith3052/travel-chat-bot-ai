from flask import Flask, request, jsonify
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
import re

from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import cassio
from flask import Flask, request, render_template
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:SzpXOwPzBCArDAZuPWPZHBkJ:08d2b64ba79cc21bf3a11fc64722e5b3f5462eb56603a7dd58498b43315862ee"
ASTRA_DB_ID = "70140abb-1253-4c1f-bd5c-c48fd3bad02c" # enter your Database ID
OPENAI_API_KEY ="sk-proj-6MJfPIj5KDBPBxzCuxBBT3BlbkFJmL198mgSzQLxuVh1yB3y" # enter your OpenAI key
PDF_FILE_PATH = 'tourist_attractions_combined.pdf'

pdfreader = PdfReader("tourist_attractions_combined.pdf")
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-instruct")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
import pandas as pd

# Initialize an empty list to store data
data = []

# Flag to check if it's the first question
first_question = True

# Counter for serial number
slno = 1

# Route to handle user queries
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data['query']

    # Perform the query
    answer = astra_vector_index.query(user_query, llm=llm).strip()
    lines = []
    for line in re.split(r'[-?.]', answer):
        lines.append(line.strip())

    # Return each line as a separate response in the JSON object
    return jsonify({'response': lines})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
