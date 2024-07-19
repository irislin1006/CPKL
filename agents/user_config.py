import os
cache_dir="agents/cache/"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ['HF_HOME'] = cache_dir

from agents.deberta_nlu_deberta_relation import DebertaNluRelationDeberta
from agents.deberta_nli_deberta_relation import DebertaNliRelationDeberta
from agents.deberta_nlu_deberta_tail import DebertaNluTailDeberta
from agents.deberta_nlu_deberta_head import DebertaNluHeadDeberta
from agents.deberta_nlu_deberta_full import DebertaNluFullDeberta
from agents.deberta_nlu_head_deberta_tail_agent import DebertaNluHeadDebertaNluTailAgent
from agents.comfact_relation_agnostic_baseline_agent import ComfactRelationAgnosticBaselineClassifierAgent

UserAgent = DebertaNluHeadDeberta