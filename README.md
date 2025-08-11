# Research Agent

LLM-powered researcher that combines OpenAI chat models with Google results via SerpAPI. The agent asks the model to emit all needed searches at once, runs them concurrently, feeds snippets back, and returns a well‑cited answer. Includes a simple CLI.

## Features
- OpenAI Chat Completions with function calling
- Batches 2–50 `search_web` tool calls in one turn
- Concurrent Google searches via SerpAPI
- Optional JSON trace (`--outfile`) with steps and final answer

## Requirements
- Python 3.9+ (3.10+ recommended)
- OPENAI_API_KEY
- SERPAPI_API_KEY

## Install
```bash
git clone https://github.com/vladm-serpapi/web-research-agent
cd web-research-agent
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Setup API keys
Option A — export in your shell (recommended):
```bash
export OPENAI_API_KEY="sk-..."
export SERPAPI_API_KEY="..."
```
Option B — .env file (don’t commit this file):
```bash
# .env
export OPENAI_API_KEY="sk-..."
export SERPAPI_API_KEY="..."
# load it
source .env
```
Security: Never share or commit your keys.

## Quick start
```bash
python research_agent.py -q "What are the latest approaches to retrieval‑augmented generation in 2025?"
# Save full JSON trace
python research_agent.py -q "State of LLM reasoning benchmarks in 2025" --outfile trace.json
```

## CLI
```bash
python research_agent.py -h
# usage: research_agent.py [-h] -q QUERY [-m {o3,o4-mini,gpt-4o}] [-n TOPN] [-o OUTFILE] [-d]
#   -q, --query        Research question (required)
#   -m, --model        o3 (default) | o4-mini | gpt-4o
#   -n, --topn         Organic results per search (default: 10)
#   -o, --outfile      Write JSON trace to file
#   -d, --debug        Print debug logs
```

## How it works (brief)
- System prompt asks the model to emit all `search_web` calls first
- Agent executes all requested Google searches concurrently (SerpAPI)
- Results are passed back as tool messages; model produces a final, cited answer
- Note on model tool behavior: o3 / o4-mini reasoning models prefer to output single tool call per prompt, so gpt-4o is preferred when many queries are required

## Example
```bash
python research_agent.py -q "Compare FAISS vs. Milvus vs. Qdrant for RAG (2025)" -m o3 -n 8 -o rag_db_trace.json
```

### With debug mode
```bash
 python research_agent.py -q "airlines industry trend 2025, compare multiple trends by impact and research each deeper to provide a comphrehensive picture" --outfile trace.json --debug --model gpt-4o
```

### Sample output

```bash
 python research_agent.py -q "research the nuclear energy sector in 2025 and build a comprehensive thesis / report on it. I want this 
report to cover AI, uranium, energy, etc. Financial projections, key players, companies, etc. Do the research in iterative fashion, after each round of searches and getting new info
rmation, do another round of searches to dive deeper into each specific topic. Don't stop on surface findings. Think and analyze what data are you missing, and proceed to research it deeper." --outfile trace.json --debug --model gpt-4o
[DEBUG] → OpenAI chat.completions.create request …
[DEBUG] → SerpAPI query: 'nuclear energy sector 2025 overview'
[DEBUG] → SerpAPI query: 'AI in nuclear energy 2025'
[DEBUG] → SerpAPI query: 'uranium market 2025'
[DEBUG] → SerpAPI query: 'key companies in nuclear energy 2025'
[DEBUG] → OpenAI chat.completions.create request …
[DEBUG] → SerpAPI query: 'financial projections nuclear energy 2025'
[DEBUG] → SerpAPI query: 'nuclear energy policies 2025'
[DEBUG] → SerpAPI query: 'AI-driven nuclear technologies 2025'
[DEBUG] → SerpAPI query: 'key innovations in nuclear technology 2025'
[DEBUG] → OpenAI chat.completions.create request …
…
```

## JSON trace example (with --outfile)
```json
{
  "question": "...",
  "answer": "...",
  "steps": [
    { "type": "tool_call", "query": "first search" },
    { "type": "tool_result", "content": "- Title: snippet ..." },
    { "type": "assistant_answer", "content": "final answer text" }
  ]
}
```

## Programmatic use
```python
from research_agent import ResearchAgent

agent = ResearchAgent(model="o3", topn=10, debug=False)
result = agent.run("Summarize the most cited papers on RAG.")
print(result["answer"])  # final answer
print(len(result["steps"]))
```

## Troubleshooting
- "OPENAI_API_KEY and SERPAPI_API_KEY must be set." → export both keys or source your .env
- Model not available → switch to a supported one (o3, o4-mini, gpt-4o)
- Empty/failed searches → check SerpAPI key/quota and network settings

## License
MIT License — see LICENSE.