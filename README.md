# ARENA Tutor

RAG-enhanced AI tutor for ARENA 3.0: AI Safety Fundamentals curriculum.

## Quick Start

```bash
# Install dependencies
npm install

# Pull required Ollama models
ollama pull gemma3:12b
ollama pull nomic-embed-text

# Generate embeddings (first time only)
node server/embeddings.js

# Start the tutor API
node server/tutor-api.js

# In another terminal, serve the UI
python3 -m http.server 8000

# Open http://localhost:8000/ui/app.html
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web UI        │────▶│   Tutor API     │────▶│   Ollama        │
│   (app.html)    │     │   (port 3002)   │     │   (gemma3:12b)  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  RAG Retrieval  │
                        │  (nomic-embed)  │
                        │  104 chunks     │
                        └─────────────────┘
```

## Features

- **RAG-enhanced responses**: Retrieves relevant curriculum content before answering
- **Persona adaptation**: Tyla (CS), Aaliyah (Bootcamp), Maneesha (ID)
- **Mental model grounding**: Collapse, Thermostat, Routing, Dimensionality
- **Visual network diagram**: Progressive understanding model
- **LangSmith tracing**: Full observability with graceful flush on exit

## Curriculum Structure

- **Chapter 0**: Fundamentals (Weeks 1-2)
- **Chapter 1**: Transformer Interpretability (Weeks 3-5)
- **Chapter 2**: Reinforcement Learning (Weeks 6-7)
- **Chapter 3**: LLM Evaluations (Weeks 8-9)

**Capstone**: Sycophancy Detection and Mitigation

## Environment Variables

```bash
LANGSMITH_API_KEY=lsv2_...  # Optional: Enable tracing
LANGSMITH_PROJECT=arena-tutor
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
PORT=3002
```

## Assessment Gap Analysis

See `data/assessment-gap-analysis.json` for Gagné's Nine Events coverage and implementation roadmap.
