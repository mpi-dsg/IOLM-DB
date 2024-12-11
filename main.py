import argparse
import logging
import pandas as pd

from llm_query.core import LLMQueryManager
from llm_query.evaluators import ModelComparator

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='llm_query.log'
    )

def main():

    parser = argparse.ArgumentParser(description='LLM Query Tool')

    parser.add_argument('--query', type=str, 
                        default='Summarize the following text in exactly 5 words:',
                        help='Type of NLP query to perform')
    
    parser.add_argument('--input', type=str, 
                        default="/OS/MLeeF/nobackup/All_Beauty.jsonl",
                        help='Path to input data file (CSV, JSON, etc.)')

    parser.add_argument('--model_name', type=str, 
                        default="/NS/llm-1/nobackup/shared/huggingface_cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct", 
                        help='Path to model or Hugging Face model name')

    parser.add_argument('--output', type=str, 
                        default='llm_query_results.csv', 
                        help='Path to output results')

    parser.add_argument('--column', type=str, default='text', 
                        help='Column to apply the task on')

    parser.add_argument('--max-rows', type=int, default=10, 
                        help='Limit number of rows for processing')
    
    parser.add_argument('--llm-judge', choices=['openai', 'anthropic', 'off'], default='anthropic',
                        help='Type of judgement to use for LLM comparison')

    parser.add_argument('--time', action='store_true',
                        help='Use timed version of query functions')
    
    args = parser.parse_args()

    print(args)
    
    setup_logging()
    logger = logging.getLogger('LLMQuery')
    
    try:

        if not args.input:
            logger.error("No input file provided")
            return
        
        df = pd.read_csv(args.input) if args.input.endswith('.csv') else \
             pd.read_json(args.input, lines=True) if args.input.endswith('.jsonl') else \
             pd.read_excel(args.input)
        
        if args.max_rows:
            df = df.head(args.max_rows)
        
        query_manager = LLMQueryManager(model_name=args.model_name, llm_judge=args.llm_judge)
        
        if args.query:
            results = query_manager.execute_query_llm(df, args.query, column=args.column, measure_time=args.time)
            logger.info(f"Query results: {results}")
        
        logger.info(query_manager.performance_metrics)
        comparator = ModelComparator()
        comparison_report = comparator.compare_models(results)
        if 'semantic_score_mean' in comparison_report.columns:
            logger.info(comparison_report['semantic_score_mean'].mean())
        
        comparison_report.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()