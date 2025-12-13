"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

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

    with open("data/paraphrase_answers.pkl", "rb") as generations_file:
        generations = pickle.load(generations_file)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    # Model for p_true
    if args.compute_p_true_in_compute_stage:
        pt_model = model = HuggingfaceModel(
            'Llama-2-7b-chat', stop_sequences='default',
            max_new_tokens=200)

    # Restore outputs from `generate_answers.py` run.
    with open("data/uncertainty_paraphrase_answers.pkl", "rb") as uncertainty_measures_file:
        result_dict = pickle.load(uncertainty_measures_file)
    # result_dict = {}

    result_dict['example_ids'] = []

    # result_dict['semantic_ids'] = []
    result_dict['original_semantic_ids'] = [] # Only for original question
    result_dict['all_semantic_ids'] = []

    entropies = defaultdict(list)
    embeddings, is_true, answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(generations):
        result_dict['example_ids'].append(tid)
        example = generations[tid]
        question = example['question']
        context = example['context']
        # full_responses = example["responses"]
        original_full_responses = example["responses"]
        all_full_responses = example["responses"] + example["paraphrase_responses"]
        most_likely_answer = example['most_likely_answer'][0] # Most likely answer to original question

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            # responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
            
            original_responses = [fr[0] for fr in original_full_responses[:args.use_num_generations]]
            all_responses = [fr[0] for fr in all_full_responses[:args.use_num_generations]]
        else:
            original_responses = [fr[0] for fr in original_full_responses]
            all_responses = [fr[0] for fr in all_full_responses]

        is_true.append(most_likely_answer['accuracy'])

        answerable.append(is_answerable(example))
        embeddings.append(most_likely_answer['embedding'])
        logging.info('%i: is_true: %f', idx, is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                original_log_liks = [r[1] for r in original_full_responses[:args.use_num_generations]]
                all_log_liks = [r[1] for r in all_full_responses[:args.use_num_generations]]
            else:
                original_log_liks = [r[1] for r in original_full_responses]
                all_log_liks = [r[1] for r in all_full_responses]

            # for i in log_liks:
            #     assert i

            if args.condition_on_question and args.entailment_model == 'deberta':
                original_responses = [f'{question} {r}' for r in original_responses]
                all_responses = [f'{question} {r}' for r in all_responses]

            # Compute semantic ids.
            original_semantic_ids = get_semantic_ids(
                original_responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)
            all_semantic_ids = get_semantic_ids(
                all_responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['original_semantic_ids'].append(original_semantic_ids)
            result_dict['all_semantic_ids'].append(all_semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['original_cluster_assignment_entropy'].append(cluster_assignment_entropy(original_semantic_ids))
            entropies['all_cluster_assignment_entropy'].append(cluster_assignment_entropy(all_semantic_ids))

            # Length normalization of generation probabilities.
            original_log_liks_agg = [np.mean(log_lik) for log_lik in original_log_liks]
            all_log_liks_agg = [np.mean(log_lik) for log_lik in all_log_liks]

            # Compute naive entropy.
            entropies['original_regular_entropy'].append(predictive_entropy(original_log_liks_agg))
            entropies['all_regular_entropy'].append(predictive_entropy(all_log_liks_agg))

            # Compute semantic entropy.
            original_log_likelihood_per_semantic_id = logsumexp_by_id(original_semantic_ids, original_log_liks_agg, agg='sum_normalized')
            all_log_likelihood_per_semantic_id = logsumexp_by_id(all_semantic_ids, all_log_liks_agg, agg='sum_normalized')
            
            original_pe = predictive_entropy_rao(original_log_likelihood_per_semantic_id)
            all_pe = predictive_entropy_rao(all_log_likelihood_per_semantic_id)

            entropies['original_semantic_entropy'].append(original_pe)
            entropies['all_semantic_entropy'].append(all_pe)

            # pylint: disable=invalid-name
            # log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            # entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            # logging.info(80*'#')
            # logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            # logging.info('Context:')
            # logging.info(example['context'])
            # logging.info('Question:')
            # logging.info(question)
            # logging.info('True Answers:')
            # logging.info(example['reference'])
            # logging.info('Low Temperature Generation:')
            # logging.info(most_likely_answer['response'])
            # logging.info('Low Temperature Generation Accuracy:')
            # logging.info(most_likely_answer['accuracy'])
            # logging.info('High Temp Generation:')
            # logging.info([r[0] for r in full_responses])
            # logging.info('High Temp Generation:')
            # logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            # logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(is_true))
    is_false = [1.0 - is_t for is_t in is_true]
    result_dict['is_false'] = is_false

    unanswerable = [1.0 - is_a for is_a in answerable]
    result_dict['unanswerable'] = unanswerable
    logging.info('Unanswerable prop: %f', np.mean(unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    with open("data/combined_uncertainty_measures_paraphrases.pkl", "wb") as results_file:
        pickle.dump(result_dict, results_file)


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
