import gradio as gr
from llama_index import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.postprocessor import NvidiaNeMoRerank
from llama_index.readers import (
    Reader,
    # SimpleWebPageReader,
    HTMLTagReader,
    DocxReader,
    PDFReader,
    PresentationReader,
    ImageReader,
    CSVReader,
)

# Initialize LLM, embedding model, and text splitter
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)
Settings.service_context = ServiceContext.from_defaults(
    llm=Settings.llm, embed_model=Settings.embed_model, text_splitter=Settings.text_splitter
)

# Import Nemo modules

from nemoguardrails import LLMRails, RailsConfig

# Define a RailsConfig object
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)


# Initialize global variables for the index and query engine
index = None
query_engine = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs, url=None):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        # Create reader classes for different file types
        reader_classes = {
            ".pdf": PDFReader,
            ".docx": DocxReader,
            ".pptx": PresentationReader,
            ".csv": CSVReader,
            ".jpg": ImageReader,
            ".jpeg": ImageReader,
        }

        documents = []
        for file in file_objs:
            reader_class = reader_classes.get(file.name.lower()[-5:], Reader)
            reader = reader_class()
            documents.extend(reader.load_data(file))

        if url:
            reader = HTMLTagReader()
            documents.extend(reader.load_data(urls=[url]))

        if not documents:
            return f"No documents found in the selected files."

        # Create index from documents
        def create_index(documents):
            vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024,
            collection_name="vectorstore",
            gpu_id=0,  # Specify the GPU ID to use
            output_fields=["field1","field2"]
        )
        # vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True) #For CPU only vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create index and query engine
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return "Documents loaded successfully!"

# Function to handle chat interactions
def moderated_chat(message, history):
    global query_engine, rails
    if query_engine is None:
        return history + [("Please upload a file first.", None)]
    try:
        # Get response from query engine
        response = query_engine.query(message)
        # Provide relevant context to guardrails
        validated_response = rails.generate(context={"history": history, "response": response.response, "message": message}, prompt=message)
        history.append((message, validated_response.generated_text))
        yield history
    except Exception as e:
        yield history + [(message, f"An error occurred: {str(e)}")]

# Function to stream responses
def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("Please upload a file first.", None)]
        return

    try:
        # Get response from query engine
        response = query_engine.query(message)
        # Provide relevant context to guardrails
        validated_response = rails.generate(context={"history": history, "response": response.response, "message": message}, prompt=message)
        partial_response = ""
        for text in validated_response.response_gen:
            partial_response += text
            yield history + [(message, partial_response)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

with gr.Blocks() as iface:
    gr.Markdown("# Multi-document RAG Chatbot 🦀 ")  # Title using Markdown
    gr.Markdown("Upload various documents or HTML URL for Q&A")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                file_count="multiple", label="Upload documents")
        with gr.Column():
            url_input = gr.Textbox(label="Enter URL")
    with gr.Row():
        load_button = gr.Button("Load Documents & URL")
    status_output = gr.Textbox(label="Status")

    load_button.click(
        fn=load_documents,
        inputs=[file_input, url_input],
        outputs=status_output,
    )

    with gr.Row():
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your question")
        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        clear_button = gr.Button("Clear")
        clear_button.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
