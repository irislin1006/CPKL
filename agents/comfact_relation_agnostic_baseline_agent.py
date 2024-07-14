from typing import List, Dict
import random

import argparse
import logging
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import json
from argparse import Namespace
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    DebertaV2ForSequenceClassification,
    get_linear_schedule_with_warmup
)
from dimiss_items.model.comfact_relation_agnostic_baseline.dataset import (
    FactLinkingDataset,
    FactGenerationDataset,
    FactGenerationEvalDataset,
    SPECIAL_TOKENS,
    ADD_TOKENS_VALUES
)
from dimiss_items.model.comfact_relation_agnostic_baseline.utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from dimiss_items.model.comfact_relation_agnostic_baseline.utils.metrics import (
    BLEU, METEOR, ROUGE
)
from dimiss_items.model.comfact_relation_agnostic_baseline.utils.model import (
    run_batch_generation_train,
    run_batch_generation_eval,
    run_batch_linking,
    softmax
)

logger = logging.getLogger(__name__)

class ComfactRelationAgnosticBaselineClassifierAgent(object):
    def __init__(self):
        """ Load your model(s) here """

        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument("--params_file", type=str, help="JSON configuration file")
        parser.add_argument("--eval_only", action="store_true",
                            help="Perform evaluation only")
        parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
        parser.add_argument("--max_tokens", type=int, default=-1,
                            help="Maximum length of input tokens, will override that value in config")
        parser.add_argument("--dataroot", type=str, default="data",
                            help="Path to dataset")
        parser.add_argument("--eval_dataset", type=str, default="test",
                            help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
        parser.add_argument("--no_labels", action="store_true",
                            help="Read a dataset without labels.json. This option is useful when running "
                                 "knowledge-seeking turn detection on test dataset where labels.json is not available")
        parser.add_argument("--labels_file", type=str, default=None,
                            help="If set, the labels will be loaded not from the default path, but from this file instead, "
                            "useful to take the outputs from the previous task in the pipe-lined evaluation")
        parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file")
        parser.add_argument("--exp_name", type=str, default="",
                            help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
        parser.add_argument("--eval_desc", type=str, default="",
                            help="Optional description to be listed in eval_results.txt")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="Device (cuda or cpu)")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="Local rank for distributed training (-1: not distributed)")
        args = parser.parse_args()

        # set args directly
        args.eval_only = True
        args.checkpoint = 'dimiss_items/model/comfact_relation_agnostic_baseline/all-deberta-large-nlu-entity'
        args.params_file = 'dimiss_items/model/comfact_relation_agnostic_baseline/all-deberta-large-nlu-entity/params-deberta-large.json'
#        args.dataroot = './data/${portion}/${task}/${window}'
        args.dataroot = './tmp'
        args.no_labels = True
        args.labels_file = None

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )

        verify_args(args, parser)

        # load args from params file and update the args Namespace
        with open(args.params_file, "r") as f:
            params = json.load(f)
            args = vars(args)

            update_additional_params(params, args)
            args.update(params)
            args = Namespace(**args)

        args.params = params  # used for saving checkpoints
        set_default_params(args)
        #
        #  dataset_args={'max_utterances': 20, 'max_tokens': 512, 'dataroot': 'data/all/entity/nlu'},
        #  Namespace(dataroot='data', max_tokens=512, max_utterances=20)
        #
        self.dataset_args = Namespace(**args.dataset_args)
        set_default_dataset_params(self.dataset_args)
        self.dataset_args.local_rank = args.local_rank
        self.dataset_args.task = args.task
        self.dataset_args.model_name_or_path = args.model_name_or_path
#        self.dataset_args.window = args.dataroot.split("/")[3]
        self.dataset_args.window = 'nlu'

        # Setup CUDA, GPU & distributed training
        args.distributed = (args.local_rank != -1)
        if not args.distributed:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(rank_to_device[args.local_rank])
            device = torch.device("cuda", rank_to_device[args.local_rank])
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
        args.device = device

        # Set seed
        self.set_seed(args)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        self.dataset_class_train, \
        self.dataset_class_eval, \
        self.model_class, \
        self.run_batch_fn_train, \
        self.run_batch_fn_eval = self.get_classes(args.task)

        if args.eval_only:
            args.output_dir = args.checkpoint
            self.model = self.model_class.from_pretrained(args.checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
#        else:
#            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#            tokenizer.add_special_tokens(SPECIAL_TOKENS)
#            tokenizer.add_tokens(ADD_TOKENS_VALUES)
#            model = model_class.from_pretrained(args.model_name_or_path)
#            model.resize_token_embeddings(len(tokenizer))

        self.model.to(args.device)

        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

        # keep args
        self.args = args

    def get_classes(self, task):
        if task.lower() == "generation":
            return FactGenerationDataset, FactGenerationEvalDataset, AutoModelForSeq2SeqLM, \
                   run_batch_generation_train, run_batch_generation_eval
        elif task.lower() == "linking":
            return FactLinkingDataset, FactLinkingDataset, DebertaV2ForSequenceClassification, \
                   run_batch_linking, run_batch_linking
        else:
            raise ValueError("args.task not in ['linking', generation'], got %s" % task)

    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def evaluate(self, args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
        eval_output_dir = None
        if args.local_rank in [-1, 0]:
            eval_output_dir = args.output_dir
            os.makedirs(eval_output_dir, exist_ok=True)

        # eval_batch_size for selection must be 1 to handle variable number of candidates
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=eval_dataset.collate_fn
        )

        # multi-gpu evaluate
        if args.n_gpu > 1 and eval_dataset.args.eval_all_snippets:
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

        # linking
        data_infos = []
        all_preds = []
        all_labels = []

        # generation
        metrics = [BLEU(), METEOR(), ROUGE()]
        all_output_texts = []
        all_infos = {"context_ids": [], "turn_ids": [], "head_ids": []}
        cur_context_id = -1
        cur_turn_id = -1
        cur_head_id = -1
        cur_gen_beam = []
        cur_refs = []

        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            with torch.no_grad():
                if args.task == "generation":
                    gen_text, truth_text, data_info = run_batch_fn(args, model, batch, tokenizer)
                    for batch_id, gen_beam in enumerate(gen_text):
                        if data_info["context_ids"][batch_id] == cur_context_id and \
                                data_info["turn_ids"][batch_id] == cur_turn_id and \
                                data_info["head_ids"][batch_id] == cur_head_id:
                            pass
                        else:
                            if len(cur_refs) > 0:
                                for metric in metrics:
                                    for gen_single in cur_gen_beam:
                                        if gen_single:
                                            if args.dataroot.split("/")[2] != "all_linked_head":
                                                metric.update((gen_single, cur_refs))

                            cur_refs = []
                            cur_gen_beam = deepcopy(gen_beam)
                            cur_context_id = data_info["context_ids"][batch_id]
                            cur_turn_id = data_info["turn_ids"][batch_id]
                            cur_head_id = data_info["head_ids"][batch_id]

                            all_output_texts.append(cur_gen_beam)
                            all_infos["context_ids"].append(cur_context_id)
                            all_infos["turn_ids"].append(cur_turn_id)
                            all_infos["head_ids"].append(cur_head_id)

                        cur_refs.append(truth_text[batch_id])
                else:
                    _, _, mc_logits, mc_labels = run_batch_fn(args, model, batch, tokenizer)
                    data_infos.append(batch[-1])
                    all_preds.append(mc_logits.detach().cpu().numpy())
                    all_labels.append(mc_labels.detach().cpu().numpy())

        if args.task == "generation":
            if len(cur_refs) > 0:
                for metric in metrics:
                    for gen_single in cur_gen_beam:
                        if gen_single:
                            if args.dataroot.split("/")[2] != "all_linked_head":
                                metric.update((gen_single, cur_refs))

            if args.output_file:
                write_generation_preds(eval_dataset.dataset_walker, args.output_file, all_output_texts, all_infos)

            result = dict()
            if args.local_rank in [-1, 0]:
                if args.dataroot.split("/")[2] != "all_linked_head":
                    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results %s *****" % desc)
                        writer.write("***** Eval results %s *****\n" % desc)
                        for metric in metrics:
                            name = metric.name()
                            score = metric.compute()
                            result[name] = score
                            logger.info("  %s = %s", name, str(score))
                            writer.write("%s = %s\n" % (name, str(score)))
        else:
            all_labels = np.concatenate(all_labels)
            all_pred_ids = np.concatenate([np.argmax(logits, axis=1).reshape(-1) for logits in all_preds])
            all_pred_scores = np.concatenate([softmax(logits, axis=1) for logits in all_preds], axis=0)
            # predict 1 if the score is greater than 0.4
            all_pred_ids = (all_pred_scores[:, 1] > 0.4).astype(int)

#            accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
#            precision = precision_score(all_labels, all_pred_ids)
#            recall = recall_score(all_labels, all_pred_ids)
#            f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))
#            result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
#            print(result)
#            if args.output_file:
#                write_linking_preds(args.output_file, data_infos, all_pred_ids, all_pred_scores)
#
#            if args.local_rank in [-1, 0]:
#                output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
#                with open(output_eval_file, "a") as writer:
#                    logger.info("***** Eval results %s *****" % desc)
#                    writer.write("***** Eval results %s *****\n" % desc)
#                    for key in sorted(result.keys()):
#                        logger.info("  %s = %s", key, str(result[key]))
#                        writer.write("%s = %s\n" % (key, str(result[key])))
            result = list(map(lambda x: bool(x), all_pred_ids))

        return result

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

        # convert test_data and save to logs.json
        logs = []
        for idata in test_data:
            ilog = {}
#            ilog['cid'] = idata['dialog_id']
            ilog['cid'] = 'dummy'
            ilog['tid'] = 0
            ilog['text'] = []
            itext = {}
            itext['type'] = 'f_context'
            itext['utter'] = idata['ut-2']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'f_context'
            itext['utter'] = idata['ut-1']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'center'
            itext['utter'] = idata['ut']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'fact'
            itext['utter'] = idata['head']
            ilog['text'].append(itext)
            ilog['fid'] = -1
            logs.append(ilog)
        logs_file = os.path.join(self.args.dataroot, self.args.eval_dataset, 'logs.json')
        # breakpoint()
        if not os.path.exists(os.path.dirname(logs_file)):
            os.makedirs(os.path.dirname(logs_file))
        with open(logs_file, 'w') as f:
            json.dump(logs, f, indent=2)
        

        # do evaluation for head
        result_head = {}
        if self.args.local_rank in [-1, 0]:
            eval_dataset = self.dataset_class_eval(self.dataset_args, self.tokenizer, split_type=self.args.eval_dataset,
                                                   labels=not self.args.no_labels, labels_file=self.args.labels_file)
            result_head = self.evaluate(self.args, eval_dataset, self.model, self.tokenizer, self.run_batch_fn_eval, desc=self.args.eval_desc or "test")

        # convert test_data and save to logs.json
        logs = []
        for idata in test_data:
            ilog = {}
            ilog['cid'] = 'dummy'
            ilog['tid'] = 0
            ilog['text'] = []
            itext = {}
            itext['type'] = 'f_context'
            itext['utter'] = idata['ut-2']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'f_context'
            itext['utter'] = idata['ut-1']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'center'
            itext['utter'] = idata['ut']
            ilog['text'].append(itext)
            itext = {}
            itext['type'] = 'fact'
            itext['utter'] = idata['tail']
            ilog['text'].append(itext)
            ilog['fid'] = 0
            logs.append(ilog)
        logs_file = os.path.join(self.args.dataroot, self.args.eval_dataset, 'logs.json')
        with open(logs_file, 'w') as f:
            json.dump(logs, f, indent=2)

        # do evaluation for tail
        result_tail = {}
        if self.args.local_rank in [-1, 0]:
            eval_dataset = self.dataset_class_eval(self.dataset_args, self.tokenizer, split_type=self.args.eval_dataset,
                                                   labels=not self.args.no_labels, labels_file=self.args.labels_file)
            result_tail = self.evaluate(self.args, eval_dataset, self.model, self.tokenizer, self.run_batch_fn_eval, desc=self.args.eval_desc or "test")

        # get logical-and of predictions between head and tail
        result = np.logical_and(result_head, result_tail)
        # print('result_head', result_head)
        # print('result_tail', result_tail)
        # result = np.logical_or(result_head, result_tail)

        return [bool(r) for r in result]

