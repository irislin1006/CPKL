from typing import List, Dict
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import math
import pandas as pd
from agents.user_config import UserAgent


def load_json_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as fp:
        data = json.load(fp)
    return data

def load_data(file_path: str) -> List[Dict]:
    return load_json_data(file_path)

def classify_links(agent, test_data, BATCH_SIZE):
    all_responses = []
    split_size = math.ceil(len(test_data) / BATCH_SIZE)
    for batch_idx in np.array_split(range(len(test_data)), split_size):
        batch_inputs = [test_data[i] for i in batch_idx]
        responses = agent.classify_link(batch_inputs)
        all_responses.extend(responses)
    return all_responses

def evaluate(responses, test_data):
    answers = [td['gold_reference'] for td in test_data]
    f1 = f1_score(answers, responses, average='binary')
    acc = accuracy_score(answers, responses)
    return f1, acc

if __name__ == "__main__":
    BATCH_SIZE = 1
    agent = UserAgent()
    
    data_path = 'dimiss_items/data/eval/task2_data_subset.json'
    test_data_raw = load_data(data_path)
    test_data = test_data_raw['data']
    # breakpoint()
    responses = classify_links(agent, test_data, BATCH_SIZE)
    f1, acc = evaluate(responses, test_data)

    print("(1.1) Prediction F1 Score:", f1)
    print("(1.2) Prediction Accuracy:", acc)

    df = pd.DataFrame(test_data)
    df['prediction'] = responses
    model_name = agent.__class__.__name__
    df.to_csv(f'dimiss_items/data/eval/task2_data_subset_{model_name}_predictions.csv', index=False)
