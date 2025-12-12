"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb
from uncertainty.models.huggingface_models import HuggingfaceModel
from tqdm import tqdm

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'


def main(args):

    # with open("data/train_multiquestion_generations.pkl", "rb") as generations_file:
    with open("data/combined_paraphrase_answers.pkl", "rb") as generations_file:
        generations = pickle.load(generations_file)

    # Model for p_true
    pt_model = model = HuggingfaceModel(
        'Llama-2-7b-chat', stop_sequences='default',
        max_new_tokens=200)

    with open('semantic_uncertainty/pt_fewshot_prompt.txt', 'r') as prompt_file:
        p_true_few_shot_prompt = prompt_file.read()
    print("PT Fewshot:", p_true_few_shot_prompt)

    with open("data/combined_uncertainty_measures_paraphrases.pkl", "rb") as uncertainty_measures_file:
        result_dict = pickle.load(uncertainty_measures_file)    

    p_trues = []
    p_trues_paraphrases = []
    count = 0  # pylint: disable=invalid-name

    # Loop over datapoints and compute validation embeddings and entropies.
    for ex_id in tqdm(result_dict['example_ids']):
        example = generations[ex_id]
        question = example['question']
        context = example['context']

        original_full_responses = example["responses"]
        all_full_responses = example["responses"] + example["paraphrase_responses"]
        most_likely_answer = example['most_likely_answer'][0] # Most likely answer to original question

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            
            original_responses = [fr[0] for fr in original_full_responses[:args.use_num_generations]]
            all_responses = [fr[0] for fr in all_full_responses[:args.use_num_generations]]
        else:
            original_responses = [fr[0] for fr in original_full_responses]
            all_responses = [fr[0] for fr in all_full_responses]

        p_true = p_true_utils.calculate_p_true(
            pt_model, question, most_likely_answer['response'],
            original_responses, p_true_few_shot_prompt,
            hint=False)
        p_trues.append(p_true)

        p_true_paraphrases = p_true_utils.calculate_p_true(
            pt_model, question, most_likely_answer['response'],
            all_responses, p_true_few_shot_prompt,
            hint=False)
        p_trues_paraphrases.append(p_true_paraphrases)

        count += 1

    result_dict['uncertainty_measures']['original_p_false'] = [1 - p for p in p_trues]
    result_dict['uncertainty_measures']['all_p_false'] = [1 - p for p in p_trues_paraphrases]

    result_dict['uncertainty_measures']['original_p_false_fixed'] = [1 - np.exp(p) for p in p_trues]
    result_dict['uncertainty_measures']['all_p_false_fixed'] = [1 - np.exp(p) for p in p_trues_paraphrases]

    # with open("data/semantic_uncertainty_paraphrases.pkl", "wb") as results_file:
    with open("data/combined_uncertainty_measures_paraphrases_with_P_false.pkl", "wb") as results_file:
        pickle.dump(result_dict, results_file)

if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
