import os
cache_dir="agents/cache/"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ['HF_HOME'] = cache_dir

from agents.deberta_nlu_deberta_relation import DebertaNluRelationDeberta

UserAgent = DebertaNluRelationDeberta