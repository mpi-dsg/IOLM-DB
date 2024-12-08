import os
import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
import openai
import anthropic
from vllm import LLM
from transformers import AutoTokenizer
from functools import wraps


def _time_query(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(instance: Any, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(instance, *args, **kwargs)
        end_time = time.time()
        
        # Store total execution time
        instance.performance_metrics[func.__name__] = {
            'total_execution_time': end_time - start_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'row_count': len(args[0]) if isinstance(args[0], pd.DataFrame) else 0,
            'average_per_row': (end_time - start_time) / len(args[0]) if isinstance(args[0], pd.DataFrame) else 0
        }
        return result
    return wrapper

def _time_single_inference(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(instance: Any, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(instance, *args, **kwargs)
        end_time = time.time()
        
        # Store single inference time
        method_name = f"{func.__name__}_single"
        instance.performance_metrics[method_name] = {
            'single_inference_time': end_time - start_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return result
    return wrapper


class LLMQueryManager:
    def __init__(self, model_name: str = None, llm_judge: str = 'openai'):
        super().__init__()
        self._cache_openai = self.load_cache('openai_cache.json') if 'openai' in llm_judge else {}
        self._cache_anthropic = self.load_cache('anthropic_cache.json') if 'anthropic' in llm_judge else {}
        
        self.llm_judge = llm_judge
        self.model_name = model_name
        
        self.openai_client = openai.OpenAI() if 'openai' in self.llm_judge else None
        self.anthropic_client = anthropic.Anthropic() if 'anthropic' in self.llm_judge else None
        
        self.performance_metrics = {}
        self.system_prompt = "You are a helpful assistant"

        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = LLM(model=self.model_name)

    def load_cache(self, cache_file: str):
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self, cache_file: str, cache: dict):
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

    @_time_query
    def execute_query(self, df: pd.DataFrame, query: str, column: str = None) -> pd.DataFrame:
        df[f'local_result'] = df[column].apply(lambda x: self._local_query(x, query))
        return df

    @_time_query
    def execute_query_llm(self, df: pd.DataFrame, query: str, column: str = None) -> pd.DataFrame:
        if self.llm_judge == 'openai':
            df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._openai_query(x, query))
        elif self.llm_judge == 'anthropic':
            df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._anthropic_query(x, query))
        return df

    @_time_single_inference
    def _openai_query(self, data: str, query: str) -> str:
        prompt = f"{query}:\n\n{data}\n\n"

        if prompt in self._cache_openai:
            return self._cache_openai[prompt]

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        self._cache_openai[prompt] = response.choices[0].message.content.strip()
        self.save_cache('openai_cache.json', self._cache_openai)

        return response.choices[0].message.content.strip()

    @_time_single_inference
    def _anthropic_query(self, data: str, query: str) -> str:
        prompt = f"{query}:\n\n{data}\n\n"

        if prompt in self._cache_anthropic:
            return self._cache_anthropic[prompt]

        response = self.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        self._cache_anthropic[prompt] = response.choices[0].message.content.strip()
        self.save_cache('anthropic_cache.json', self._cache_anthropic)

        return response.choices[0].message.content.strip()

    @_time_single_inference
    def _local_query(self, data: str, query: str) -> str:
        prompt = f"{query}:\n\n{data}\n\n"

        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt =  self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        outputs = self.model.generate(formatted_prompt)

        output = outputs[0].outputs[0].text.strip()

        return output
    
   
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for executed queries.
        
        :return: Dictionary of performance metrics
        """
        return {
            'metrics': self.performance_metrics,
            'total_queries': len(self.performance_metrics),
            'average_execution_time': np.mean(
                [metric['total_execution_time'] for metric in self.performance_metrics.values()]
            ) if self.performance_metrics else 0
        }