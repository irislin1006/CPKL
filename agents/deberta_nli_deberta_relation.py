from typing import List, Dict
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re


def replace_phrases(sentence, person_a_tag):
    sentence = sentence.lower()
    # Pattern to find "I am" and replace with "person a is"
    sentence = re.sub(r"\bi am\b", f"{person_a_tag} is", sentence, flags=re.IGNORECASE)
    # Pattern to find "I was" and replace with "person a was"
    sentence = re.sub(r"\bi was\b", f"{person_a_tag} was", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bi\b", person_a_tag, sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bmy\b", f"{person_a_tag}'s", sentence, flags=re.IGNORECASE)
    sentence = sentence.replace('person a', person_a_tag)
    sentence = sentence.replace('personx', person_a_tag)
    return sentence

def get_relation_thresholding(relation, default_threshold=0.4):
    relation = relation.lower()
    relation_thres = {
        # 'routine_habit': 0.45, # 0.23232323232323235, (45 bad)
        # 'characteristic': 0.45, # 0.13131313131313133, (30 bad, 45better but bad)
        # 'characteristic_relationship': 0.45, # 0.27272727272727276, (45 no change)
        # 'experience': 0.3, # 0.16161616161616163, (30 good)
        # 'goal_plan': 0.45, # 0.13131313131313133, (30 bad, 45better but bad)
        # 'routine_habit_relationship': 0.30, # 0.25252525252525254, (45 bad, 30 bad)
        # 'experience_relationship': 0.30, # 0.7676767676767677,(45 good
        # 'goal_plan_relationship': 0.45, # 0.7777777777777778
    }
    return relation_thres.get(relation, default_threshold)


class DebertaNliRelationDeberta(object):
    def __init__(self):
        """ Load your model(s) here """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model_path = "agents/models/relation/deberta_v3_large_sample_False"
        model_path = "dimiss_items/model/nli_relation/deberta-v3-large-1720917183/deberta_v3_large_sample_False"
        # print(f"Loading head model from {head_model_path}")

        self.tokenizer, self.model = self.load_model(model_path)
        # print(f"Loaded head model from {head_model_path} and tail model from {tail_model_path}")
        # print("Two model loaded")
        self.use_tag = True
        self.pa_tag, self.pb_tag = 'Person A', 'Person B'
        
    def load_model(self, model_path: str):
        # need to add tok into final_model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model
    
    def get_head_conv_with_tag_from_comfact(self, example):
        def format_utter(utter, tag):
            if utter.strip() == '':
                return ''
            return f"{tag}: {utter}"
        
        center_utter = format_utter(example["ut"], self.pa_tag)
        post_utters = [format_utter(example["ut-2"], self.pa_tag), format_utter(example["ut-1"], self.pb_tag)]
        future_utters = [format_utter(example["ut+1"], self.pb_tag), format_utter(example["ut+2"], self.pa_tag)]

        convs_list = '\n'.join(post_utters + [center_utter] + future_utters)
        return convs_list
    
    def get_tail_conv_with_tag_from_comfact(self, example):
        return self.get_head_conv_with_tag_from_comfact(example) #.lower()

    def transform_df(self,df):
        df['head_conv'] = df.apply(self.get_head_conv_with_tag_from_comfact, axis=1)
        df['tail_conv'] = df.apply(self.get_tail_conv_with_tag_from_comfact, axis=1)
        df['head'] = df['head'].apply(lambda x: replace_phrases(x, self.pa_tag))
        df['tail'] = df['tail'].apply(lambda x: x)
        # df['fact_text'] = df.apply(lambda x: f"{x['head']} and {x['tail']}", axis=1)
        # df['fact_text'] = df.apply(lambda x: f"{x['peacok_relation']} {x['head_fact_text']} and {x['tail_fact_text']}", axis=1)
            
        df['fact_text'] = df.apply(lambda x: f"{x['relation']} {x['head']} and {x['tail']}", axis=1)
        # print('fact_text:', df['fact_text'].iloc[0])
        return df
    
    def get_predictions(self, model, tokenizer, conv_text, fact_text, threshold=0.4):
        inputs = tokenizer(conv_text, fact_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = (probs[:, 0] >= threshold).int() #.cpu().numpy()
        preds = [True if pred == 1 else False for pred in predicted_ids]
        return preds, probs

    def classify_link(self, test_data: List[Dict]) -> List[bool]:
        """
        Input 1 -
        [
            {"ut-2": ..., "ut+2": ... "head": ..., "relation": ..., "tail": ... }, # conversation window 1
            ...
            {"ut-2": ..., "ut+2": ... "head": ..., "relation": ..., "tail": ... }  # conversation window 250
        ]

        Model should return a list of True or False for each conversation window

        """
        res = []
        for i, t_data in enumerate(test_data):
            relation = t_data["relation"]
            test_df = pd.DataFrame([t_data])
            test_df = self.transform_df(test_df)
            test_ds = Dataset.from_pandas(test_df)
            with torch.no_grad():
                threshold = get_relation_thresholding(relation, default_threshold=0.31)
                # print('threshold', threshold)
                preds, probs = self.get_predictions(self.model, self.tokenizer, test_ds["head_conv"], test_ds["fact_text"], threshold=threshold)

                # final_probs = convert_sklearn_to_numpy_formula(h_probs[:, 0], t_probs[:, 0])
                # print(i,'[Conv]\n', test_ds["head_conv"])
                # print(i, 'fact', test_ds["fact_text"], )
                # print(i, 'preds', preds, probs)
                # print(i, 'label', test_ds["gold_reference"])
            # print()
            res.extend(preds)
        return res