# LLM Query Tool

## Overview
A comprehensive tool for integrating Large Language Models (LLMs) with data processing tasks.

## Features
- Multi-model support (OpenAI, Anthropic, Local models)
- Tasks:
  * Text Summarization
  * Sentiment Classification
  * Data Cleaning
  * Fuzzy Joining

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/llm_query.git
cd llm_query
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up API keys
- Create a `.env` file in the project root
- Add your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

### Command Line Arguments
```bash
python main.py --task summarize \
               --input data.csv \
               --column text_column \
               --models openai anthropic \
               --output results.csv
```

### Tasks
- `summarize`: Generate concise summaries
- `classify`: Perform sentiment analysis
- `clean`: Clean and preprocess data
- `join`: Perform fuzzy joins

## Examples

### Summarization
```bash
python main.py --task summarize --input reviews.csv --column review_text
```

### Sentiment Classification
```bash
python main.py --task classify --input tweets.csv --column tweet_text
```

## Performance Tracking
- Detailed performance metrics are logged
- Comparison reports generated for multi-model runs

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License