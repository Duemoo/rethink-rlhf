from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from datasets import load_metric
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from rl4lms.envs.text_generation.caption_metrics.cider import Cider
from rl4lms.envs.text_generation.caption_metrics.spice.spice import Spice
from gem_metrics.texts import Predictions
from rl4lms.envs.text_generation.summ_metrics.summa_c import SummaCConv, SummaCZS
from rl4lms.data_pools.task_utils.totto.eval_utils import compute_parent, compute_bleu
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from tqdm import tqdm
import copy
import rouge
import json


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class RLHFSimMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = 'en'
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        ref_texts = [ref[0] for ref in reference_texts]
        rejected_texts = [rej['rejected'] for rej in meta_infos]

        scores = 0

        for i in tqdm(range(len(generated_texts))):

            prediction = [generated_texts[i]]
            rejected = [rejected_texts[i]]
            reference = [ref_texts[i]]
            with torch.no_grad():
                rejected_metric_result = self._metric.compute(
                    predictions=prediction,
                    references=rejected,
                    lang=self._language,
                    device=self._last_gpu,
                )["f1"][0]
                chosen_metric_result = self._metric.compute(
                    predictions=prediction,
                    references=reference,
                    lang=self._language,
                    device=self._last_gpu,
                )["f1"][0]

            if rejected_metric_result < chosen_metric_result:
                scores += 1

        avg_score = scores / len(generated_texts)
        print(f"scores: {scores} | avg: {avg_score}")
        metric_dict = {"fluency_metrics/chosen-preference-bertscore": (None, avg_score)}

        return metric_dict


class RepMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self.measure_repetition_and_diversity(generated_texts)

        metric_dict = {"fluency_metrics/rep-2": (None, score[0]),
                        "fluency_metrics/rep-3": (None, score[1]),
                        "fluency_metrics/rep-4": (None, score[2]),
                        "fluency_metrics/rep-div": (None, score[3])
                        }

        return metric_dict
    
    def eval_text(self, text, ngram):
        token_list = text.strip().split()
        start_idx, end_idx = 0, ngram
        total_num = 0
        ngram_set = set()
        while end_idx < len(token_list):
            one_ngram_list = token_list[start_idx:end_idx]
            assert len(one_ngram_list) == ngram
            one_ngram = ' '.join(one_ngram_list)
            total_num += 1
            ngram_set.add(one_ngram)
            start_idx += 1
            end_idx += 1
        return len(ngram_set), total_num

    def eval_one_instance(self, text, ngram_list):
        res_dict = {}
        for n in ngram_list:
            n_unique, n_total = self.eval_text(text, n)
            res_dict[n] = {'unique':n_unique, 'total':n_total}
        unique_token_set = set(text.strip().split())
        return res_dict, unique_token_set

    def measure_repetition_and_diversity(self, text_list):
        '''
            text_list: the list of text
        '''
        ngram_list = [2,3,4]
        pred_res_dict = {}
        for n in ngram_list:
            pred_res_dict[n] = {}
            pred_res_dict[n]['unique'] = 0
            pred_res_dict[n]['total'] = 0
        
        pred_unique_token_set = set()
        for text in text_list:
            text = text.strip('\n').strip()
            one_pred_res_dict, one_pred_uni_token_set = self.eval_one_instance(text, ngram_list)

            # unique token set
            pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
            # ngram statistic
            for n in ngram_list:
                pred_res_dict[n]['unique'] += one_pred_res_dict[n]['unique']
                pred_res_dict[n]['total'] += one_pred_res_dict[n]['total']

        # prediction result
        pred_seq_2 = 1 - (pred_res_dict[2]['unique']/pred_res_dict[2]['total'])
        pred_seq_2 = round(pred_seq_2 * 100, 2)
        pred_seq_3 = 1 - (pred_res_dict[3]['unique']/pred_res_dict[3]['total'])
        pred_seq_3 = round(pred_seq_3 * 100, 2)
        pred_seq_4 = 1 - (pred_res_dict[4]['unique']/pred_res_dict[4]['total'])
        pred_seq_4 = round(pred_seq_4 * 100, 2)
        pred_div = (1 - pred_seq_2/100) * (1 - pred_seq_3/100) * (1 - pred_seq_4/100)
        return pred_seq_2, pred_seq_3, pred_seq_4, pred_div


class RougeMetric(BaseMetric):
    def __init__(self, use_single_ref: bool = True) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BERTScoreMetric(BaseMetric):
    def __init__(self, language: str) -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                device=self._last_gpu,
            )
            bert_scores = metric_results["f1"]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEUMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts
            )
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class BLEURTMetric(BaseMetric):
    def __init__(self, config_name: str = None) -> None:
        super().__init__()
        self._metric = load_metric("bleurt", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[split_name + prompt_text] = [gen_text]
        refs[split_name + prompt_text] = ref_text
    return preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class CIDERMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Cider()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)
        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/cider": (individual_scores, corpus_score)}
        return metric_dict


class SpiceMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Spice()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)

        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/spice": (individual_scores, corpus_score)}
        return metric_dict


class DiversityMetrics(BaseMetric):
    def __init__(self, window_size: int = 100) -> None:
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class SummaCZSMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCZS(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"consistency/summaczs": (metric_results["scores"], corpus_score)}
        return metric_dict


class SummaCConvMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCConv(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {
            "consistency/summacconv": (metric_results["scores"], corpus_score)
        }
        return metric_dict


class Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        encodings = tokenizer("\n\n".join(reference_texts), return_tensors="pt")

        device = self.get_device(model)

        nlls = []
        max_length = model.config.n_positions
        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # run on last device
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }


class ParentToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        parent_overall, parent_overlap, parent_non_overlap = compute_parent(
            generated_texts, tables
        )

        metric_results = {}
        metric_names = ["parent_overall", "parent_overlap", "parent_non_overlap"]
        metric_values = [parent_overall, parent_overlap, parent_non_overlap]
        for name, value in zip(metric_names, metric_values):
            metric_results[f"table_to_text/{name}/precision"] = (
                None,
                value["precision"],
            )
            metric_results[f"table_to_text/{name}/recall"] = (None, value["recall"])

            # individual f-scores - fetch only for overall since we don't know for which samples
            if name == "parent_overall":
                f_scores = value["all_f"]
            else:
                f_scores = None

            metric_results[f"table_to_text/{name}_f_score"] = (
                f_scores,
                value["f_score"],
            )
        return metric_results


class BLEUToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        bleu_overall, bleu_overlap, bleu_non_overlap = compute_bleu(
            generated_texts, tables
        )

        metric_results = {
            "table_to_text/bleu_overall": (None, bleu_overall),
            "table_to_text/bleu_overlap": (None, bleu_overlap),
            "table_to_text/bleu_non_overlap": (None, bleu_non_overlap),
        }
        return metric_results


class RougeLMax(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25)
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = load_metric("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict


class TERMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("ter")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/ter": (None, score)}
        return metric_dict


class chrFmetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("chrf")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/chrf": (None, score)}
        return metric_dict


class MAUVEMetric(BaseMetric):
    def __init__(self, debug=False) -> None:
        super().__init__()
        self._metric = load_metric("mauve")
        self.debug = debug

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        assert len(generated_texts) == len(reference_texts)
        references = []
        for idx, reference in enumerate(reference_texts):
            assert len(reference) == 1
            references.append(reference[0])
        try:
            score = self._metric.compute(
                predictions=generated_texts, references=references, device_id=2, verbose=False,
            ).mauve
        except: #score = 0.0
            print('error occurred during MAUVE score computation. Set to score=0')
            score = 0.0
        
        if self.debug:
            def split(a, n):
                k, m = divmod(len(a), n)
                return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
            split_gen = split(generated_texts, 100)
            split_ref = split(references, 100)
            error_cases = {'ref': [], 'gen': []}
            for i in range(len(split_gen)):
                try:
                    score = self._metric.compute(
                        predictions=split_gen[i], references=split_ref[i], device_id=1, verbose=False,
                    ).mauve
                except:
                    print('error occurred in MAUVE metric computation!')
                    print(f"reference: {split_ref[i]}")
                    print(f"generated_text: {split_gen[i]}")
                    error_cases['ref'].append(split_ref[i])
                    error_cases['gen'].append(split_gen[i])
            
            with open("/home/work/smlm/smlm/mauve_error_cases.json", "a") as f:
                json.dump(error_cases, f, indent=4)

        metric_dict = {"semantic/mauve": (None, score)}
        return metric_dict


class VerbalizerMetric(BaseMetric):
    def __init__(self, tokenizer_id) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._tokenizer.truncation_side = 'left'
        self._tokenizer.padding_side = 'left'
        self._max_length = 512 # hard-coded

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        ref_texts = [ref[0] for ref in reference_texts]
        rejected_texts = [rej['rejected'] for rej in meta_infos]
        options = list(zip(ref_texts, rejected_texts))
        score = 0

        for index, option in enumerate(tqdm(options)):
            prob_list = []
            with torch.no_grad():
                for opt_idx in range(len(option)):
                    option_ = self._tokenizer.encode_plus(option[opt_idx], max_length=self._max_length,
                                                    padding=True, truncation=True, return_tensors="pt")
                    source = self._tokenizer.encode_plus(prompt_texts[index], max_length=self._max_length,
                                                    padding=True, truncation=True, return_tensors="pt")
                    source_ids = source["input_ids"]
                    source_mask = source["attention_mask"]
                    lm_labels = option_["input_ids"].expand(len(source_ids), -1)
                    lm_labels[lm_labels[:, :] == self._tokenizer.pad_token_id] = -100
                    outputs = model(
                        input_ids=source_ids.cuda(),
                        attention_mask=source_mask.cuda(),
                        labels=lm_labels.cuda(),
                        decoder_attention_mask=option_["attention_mask"].cuda()
                    )

                    logits = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)
                    lm_labels=lm_labels.cuda().unsqueeze(-1)
                    seq_token_log_prob=torch.zeros(lm_labels.shape)
                    #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                    for i in range(lm_labels.shape[0]):
                        for j in range(lm_labels.shape[1]):
                            seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                    seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                    prob_list.append(seq_log_prob)
                    #print(f"option: {option[index]}\nseq_log_prob: {seq_log_prob}\nprob_list: {prob_list}")
                concat = torch.cat(prob_list).view(-1,len(source_ids))
                #print('\n', 'concat: ', concat, '\n\n\n')
                predictions = concat.argmax(dim=0)
                if predictions.item() == 0:
                    score += 1
                #dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)]
        
        avg_score = score / len(generated_texts)
        print(f"scores: {score} | avg: {avg_score}")
        metric_dict = {"fluency_metrics/chosen-preference-verbalizer": (None, avg_score)}
        return metric_dict


class IntentAccuracyDailyDialog(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = f"cuda:{torch.cuda.device_count() - 1}"
        self._model = self._model.to(self._device)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text

        # we have to extract the history utterances
        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


if __name__ == "__main__":
    prompt_texts = [""]
    gen_texts = ["Hello there general kenobi", "foo bar foobar"]
    reference_texts = [["Hello there general kenobi"], ["foo bar foobar"]]
    # metric = MeteorMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = RougeMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SacreBLEUMetric(tokenize="intl")
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = TERMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = chrFmetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BERTScoreMetric(language="en")
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEUMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEURTMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = DiversityMetrics()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft’s Internet Business Unit. In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft’s fastest-growing businesses, exceeding $2 billion in revenues. Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft’s enterprise cloud leadership. He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft’s mobile-first/cloud-first transformation and acquisitions. Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University."""
    # summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

    # metric = SummaCZSMetric(granularity="sentence",
    #                         use_ent=True,
    #                         use_con=False)
    # print(metric.compute([document], [summary], []))

    # metric = SummaCConvMetric(granularity="sentence")
    # print(metric.compute([document], [summary], []))

    prompt_texts = ["1", "2"]
    gen_texts = [
        "The dog is the boy's cat.",
        "A boy is picking apples from trees and put them into bags.",
    ]
    reference_texts = [
        ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
        ["A boy is picking apples from trees."],
    ]
    metric = CIDERMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))

    metric = SpiceMetric()
    print(metric.compute(prompt_texts, gen_texts, reference_texts))
