# IPC Chatbot

An intelligent chatbot that answers questions about the Indian Penal Code (IPC) using Retrieval-Augmented Generation (RAG) with a locally hosted Large Language Model.

## Overview

This project creates a conversational AI assistant that can answer legal queries about the Indian Penal Code by retrieving relevant information from the official IPC document and generating contextual responses using the Falcon-7B language model.

## Features

- **Document Processing**: Loads and processes the IPC PDF document
- **Semantic Search**: Uses FAISS vector store with sentence transformers for efficient similarity search
- **Local LLM**: Runs Falcon-7B-Instruct model with 4-bit quantization for memory efficiency
- **Conversational Interface**: Interactive command-line chat interface
- **Context-Aware Responses**: Retrieves relevant IPC sections before generating answers

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector database for semantic search
- **Hugging Face Transformers**: For loading and running the LLM
- **Sentence Transformers**: For creating document embeddings
- **PyPDF**: For PDF document processing
- **BitsAndBytes**: For model quantization

## Requirements

```
faiss-cpu
torch
pypdf
langchain-community
langchain_huggingface
sentence_transformers
bitsandbytes
transformers
```

## Installation

1. Install required packages:
```bash
pip install faiss-cpu torch pypdf
pip install -U langchain-community
pip install langchain_huggingface
pip install sentence_transformers
pip install bitsandbytes
```

2. Mount Google Drive (if using Google Colab):
```python
from google.colab import drive
drive.mount("/content/drive")
```

3. Ensure you have the IPC PDF document available at the specified path.

## Usage

### Running the Chatbot

```python
# Start the interactive chat session
chat()
```

### Example Queries

```
You: what is the punishment for murder?
IPC Chatbot: The punishment for murder is death or imprisonment for life, and the offender shall also be liable to fine.

You: what is section 302?
IPC Chatbot: Section 302 states that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.
```

### Exiting the Chat

Type `exit`, `quit`, or `bye` to end the conversation.

## Architecture

### 1. Document Loading
- Loads the IPC PDF using `PyPDFLoader`
- Splits the document into manageable chunks (1000 chars with 200 char overlap)

### 2. Embedding & Indexing
- Uses `all-MiniLM-L6-v2` model to create embeddings
- Stores embeddings in FAISS vector store for fast retrieval

### 3. Language Model
- **Model**: Falcon-7B-Instruct
- **Optimization**: 4-bit quantization for reduced memory usage
- **Framework**: Hugging Face Transformers pipeline

### 4. Retrieval Chain
- Uses `RetrievalQA` chain with "stuff" strategy
- Retrieves relevant document chunks based on query similarity
- Passes context to LLM for answer generation

## Configuration

### Model Settings
```python
model_name = "tiiuae/falcon-7b-instruct"
torch_dtype = torch.float16
load_in_4bit = True
max_new_tokens = 100
```

### Text Splitting
```python
chunk_size = 1000
chunk_overlap = 200
```

### Embedding Model
```python
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

## Memory Requirements

- **GPU**: Recommended for faster inference (T4 or better)
- **RAM**: Minimum 12GB (with 4-bit quantization)
- **Storage**: ~14GB for Falcon-7B model

## Limitations

- Response quality depends on the LLM's understanding
- Limited to information present in the IPC document
- May occasionally generate incorrect or incomplete answers
- Response length capped at 100 tokens

## Troubleshooting

### Common Issues

1. **Out of Memory Error**: Reduce `max_new_tokens` or use a smaller model
2. **Slow Response Time**: Ensure GPU is being utilized (`device_map="auto"`)
3. **Poor Answers**: Check if the query relates to content in the IPC document

## License

This project is for educational purposes. The Indian Penal Code is a government document in the public domain.

## Acknowledgments

- Falcon-7B by Technology Innovation Institute (TII)
- Sentence Transformers by UKP Lab
- LangChain framework
- Hugging Face model hub

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.


**Note**: This chatbot is for informational purposes only and should not be considered as legal advice. Always consult with a qualified legal professional for legal matters.
