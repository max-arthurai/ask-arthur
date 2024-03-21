# Ask Arthur

## Setup

From the main directory of this repository, install the requirements from the terminal:
```bash
pip install -r requirements.txt
```

Before running anything, export API keys for any providers you will be accessing in your terminal session
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

Run with default settings
```bash
python run.py
```

Run with specific LLM
```bash
python run.py --llm="claude-3-haiku-20240307"
```

Run with specific prompt
```bash
python run.py --prompt="What is the difference between shield and chat?"
```

Run with DSPy framework
```bash
python run.py --prompt="What is the difference between shield and chat?"
```

Putting it all together
```bash
python run.py --llm="claude-3-haiku-20240307" --prompt="What is the difference between shield and chat?"
```
