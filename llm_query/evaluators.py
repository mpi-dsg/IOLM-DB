import pandas as pd
import numpy as np
from typing import Dict, Any, List
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from rouge_score import rouge_scorer

class ModelComparator:

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)


    def calculate_semantic_score(self, original_text: str, summary: str) -> float:

        rouge_scores = self.rouge_scorer.score(original_text, summary)
        return np.mean([
            rouge_scores['rouge1'].fmeasure, 
            rouge_scores['rouge2'].fmeasure
        ])

    def compare_models(self, df: pd.DataFrame) -> pd.DataFrame:

        if 'openai_result' in df.columns:
            df['semantic_score_openai'] = df.apply(
                lambda row: self.calculate_semantic_score(row['openai_result'], row['local_result']), 
                axis=1
            )

        if 'anthropic_result'in df.columns:
            df['semantic_score_anthropic'] = df.apply(
                lambda row: self.calculate_semantic_score(row['anthropic_result'], row['local_result']), 
                axis=1
            )
        
        if 'openai_result' in df.columns and 'anthropic_result' in df.columns:
            df['semantic_score_mean'] = df[['semantic_score_openai', 'semantic_score_anthropic']].mean(axis=1)

        return df