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

    def execute_query_llm(self, df: pd.DataFrame, query: str, column: str = None, measure_time: bool = False) -> pd.DataFrame:
        
        if measure_time:
            total_start_time_local = time.time()
            df[f'local_result'] = df[column].apply(lambda x: self._timed_local_query(x, query))
            total_end_time_local = time.time()
            total_inference_time_local = total_start_time_local - total_end_time_local
            self.performance_metrics['total_inference_time_local'] = total_inference_time_local
        else:
            df[f'local_result'] = df[column].apply(lambda x: self._local_query(x, query))

        
        if self.llm_judge == 'openai':
            if measure_time:
                total_start_time_openai = time.time()
                df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._timed_openai_query(x, query))
                total_end_time_openai = time.time()
                total_inference_time_openai = total_start_time_openai - total_end_time_openai
                self.performance_metrics['total_inference_time_openai'] = total_inference_time_openai
            else:
                df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._openai_query(x, query))
        elif self.llm_judge == 'anthropic':
            if measure_time:
                total_start_time_anthropic = time.time()
                df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._timed_anthropic_query(x, query))
                total_end_time_anthropic = time.time()
                total_inference_time_anthropic = total_start_time_anthropic - total_end_time_anthropic
                self.performance_metrics['total_inference_time_anthropic'] = total_inference_time_anthropic
            else:
                df[f'{self.llm_judge}_result'] = df[column].apply(lambda x: self._anthropic_query(x, query))
        return df

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

        total_queries = sum(len(times) for times in self.performance_metrics.get('single_inference_times_local', []))
        
        average_execution_time_local = np.mean(
            [entry['time'] for entry in self.performance_metrics.get('single_inference_times_local', [])]
        ) if total_queries > 0 else 0

        average_execution_time_openai = np.mean(
            [entry['time'] for entry in self.performance_metrics.get('single_inference_times_openai', [])]
        ) if total_queries > 0 else 0

        average_execution_time_anthropic = np.mean(
            [entry['time'] for entry in self.performance_metrics.get('single_inference_times_anthropic', [])]
        ) if total_queries > 0 else 0
        
        return {
            'metrics': self.performance_metrics,
            'total_queries': total_queries,
            'average_execution_time_local': average_execution_time_local,
            'average_execution_time_openai': average_execution_time_openai,
            'average_execution_time_anthropic': average_execution_time_anthropic
        }


    def _time_query(self, query_func: Callable, data: str, query: str, query_name: str) -> str:
        start_time = time.time()
        result = query_func(data, query)
        end_time = time.time() 
        single_inference_time = end_time - start_time
        
        self.performance_metrics.setdefault(f'single_inference_times_{query_name}', []).append({
            'query': query[:5],
            'time': single_inference_time
        })
        
        return result
    
    def _timed_local_query(self, data: str, query: str) -> str:
        return self._time_query(self._local_query, data, query, "local")
    
    def _timed_openai_query(self, data: str, query: str) -> str:
        return self._time_query(self._openai_query, data, query, "openai")

    def _timed_anthropic_query(self, data: str, query: str) -> str:
        return self._time_query(self._anthropic_query, data, query, "anthropic")