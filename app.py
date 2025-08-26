# import langchain dependencies
#from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# bring in stramlit for UI development
import streamlit as st
# bring in watsonx interface
from watsonxlangchain import LangChainInterface


creds = {
    'api_key': '',
    'url': 'https://eu-de.ml.cloud.ibm.com'
}

# create LLM using langchain
llm = LangChainInterface(
    credentials = creds,
    model = 'meta-llama/llama-2-70b-chat',
    params = {
        'decoding_method':'sample',
        'max_new_tokens':200,
        'temperature':0.5
    },
    project_id=''
)

# load local DB (PDF's)
@st.cache_resource
def load_pdf():
    pdf_name = 'Hyprog_LLM\dataset\LOTR.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    # create vector database (chromaDB)
    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MinLM-L12-v2'),
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overload=0)
    ).from_loaders(loaders)
    #return vector db
    return index

# load er on up
index = load_pdf()

# create a Q&A chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# setuo the app title
st.title('Ask watsonx')

# setup a session state message var to store all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages=[]
# display them
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# build a prompt input template to display the prompts
prompt = st.chat_input('Pass your prompt here')

# if the user hits enter here
if prompt:
    # display the prompt
    st.chat_message('user').markdown('prompt')
    # store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content':prompt})
    # send the promp to the PDF fed chain
    response = chain.run(prompt)
    # show the llm response
    st.chat_message('assistant').markdown(response)
    # store also the llm responses in the state
    st.session_state.messages.append({'role':'assistant', 'content':response})

    
