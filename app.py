import os
import sys
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    StorageContext,
    load_index_from_storage
)
from langchain import OpenAI
import gradio as gr

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Configuration parameters
MAX_INPUT_SIZE = 4096
NUM_OUTPUTS = 512
MAX_CHUNK_OVERLAP = 20
CHUNK_SIZE_LIMIT = 600
DIRECTORY_PATH = "docs"


# Initialize and configure components
prompt_helper = PromptHelper(
    MAX_INPUT_SIZE, NUM_OUTPUTS, MAX_CHUNK_OVERLAP, chunk_size_limit=CHUNK_SIZE_LIMIT
)
llm_predictor = LLMPredictor(
    llm=OpenAI(
        temperature=0, model_name="text-davinci-003", max_tokens=NUM_OUTPUTS
    )
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)


# Load documents and create index
documents = SimpleDirectoryReader(DIRECTORY_PATH).load_data()
index = GPTVectorStoreIndex.from_documents(
    documents=documents, service_context=service_context
)
index.storage_context.persist(persist_dir="./index.json")




# Process input query and return the response
def process_query(input_text):
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./index.json")
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(input_text)
        return (
            str(response.response) + "\n\n" + str(response.get_formatted_sources())
        )


    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return "An error occurred. Please try again."


# Create Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-Trained AI",
)


# Launch the interface
iface.launch(server_name="127.0.0.1", server_port=3000, share=False)