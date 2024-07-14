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


class DebertaNluHeadDebertaNluTailAgent(object):
    def __init__(self):
        """ Load your model(s) here """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        head_model_path = "dimiss_items/model/head/deberta-v3-large-1720879116/deberta_v3_large_sample_False"
        tail_model_path = "dimiss_items/model/tail/deberta-v3-large-1720894757/deberta_v3_large_sample_False"

        self.head_tokenizer, self.head_model = self.load_model(head_model_path)
        self.tail_tokenizer, self.tail_model = self.load_model(tail_model_path)

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
            test_df = pd.DataFrame([t_data])
            test_df = self.transform_df(test_df)
            test_ds = Dataset.from_pandas(test_df)
            with torch.no_grad():
                head_preds, h_probs = self.get_predictions(self.head_model, self.head_tokenizer, test_ds["head_conv"], test_ds["head"], threshold=0.45)
                tail_preds, t_probs = self.get_predictions(self.tail_model, self.tail_tokenizer, test_ds["tail_conv"], test_ds["tail"], threshold=0.35)

            res.extend([h and t for h, t in zip(head_preds, tail_preds)])
        return res