# ragchat

Demo Retrieval Augmented Chatbot AI using open source components and [SemaDB](https://www.semafind.com/products/semadb).

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/) as the main LLM:

```bash
https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
```

## Usage

Start the Ollama server:

```bash
ollama serve
```

and install the desired model:

```bash
ollama pull llama2
# Test using
ollama run llama2
```

then run the `main.py` entry point with the SemaDB API Key:

```bash
SEMADB_API_KEY=... python3 main.py
```

### Built With

- [LangChain](https://www.langchain.com/) Used for gluing together components
- [Gradio](https://www.gradio.app/) Web interface
- [Ollama](https://ollama.com/) Main large language model library
- [Sentence Transformers](https://www.sbert.net/) The embedding model