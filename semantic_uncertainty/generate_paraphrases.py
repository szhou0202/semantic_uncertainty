import torch
import os
from huggingface_hub import InferenceClient
from tqdm import tqdm
import pickle

from uncertainty.utils import utils
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta

def main(args):
    # For each question in dataset, produce a set of paraphrases
    # Check each paraphrase is semantically equivalent to original by checking bidirectional entailment of question+answer pairs
    # Save phrases

    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    args.model_max_new_tokens = 200
    # Initialize model.
    model = utils.init_model(args)
    print("Finished initializing main model. Start initializing entailment model...")
    # Initialize entailment model
    # entailment_model = EntailmentDeberta() # <- too large?
    entailment_model = "microsoft/deberta-v2-xlarge-mnli"
    entailment_client = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])

    # Load data
    train_data, val_data = load_ds(args.dataset, seed=1)
    print(f"Train data: {train_data.shape} ; Val data: {val_data.shape}")

    def check_implication(phrase1, phrase2):
        inf_result = entailment_client.text_classification(f"{phrase1}. {phrase2}", model=entailment_model)
        label = inf_result[0].label
        return label == "ENTAILMENT"

    print("Finished initializing entailment model.")
    num_generations = 3

    for dataset_split in ['train', 'validation']:
        dataset = train_data if dataset_split == 'train' else val_data

        generations = {}
        count = 0

        for i in range(args.paraphrase_start, len(dataset)):
            example = dataset[i]

            question, answer = example["question"], example['answers']['text'][0]
            print(count, "Original Q:", question)
            print("Answer:", answer)
            generations[example["id"]] = []

            prompt = f"For the following question Q, please provide {num_generations} paraphrases of the question that are semantically equivalent to it. Q: {question} Paraphrased questions:"
            # print(prompt)
            response, _, _ = model.predict(prompt, 1.0)
            # print(response)
            paraphrases = [res[3:] for res in response.split("? ")]
            paraphrases = [res if len(res)>0 and res[-1]=='?' else res+'?' for res in paraphrases]
            # print(paraphrases)
            
            original_qa_combined = question + " " + answer
            for q in paraphrases:
                print("Paraphrase:", q)
                qa_combined = q + " " + answer

                implication_1 = check_implication(original_qa_combined, qa_combined)
                implication_2 = check_implication(qa_combined, original_qa_combined)

                # Store semantically equivalent questions
                if implication_1 and implication_2:
                    print("Equivalent")
                    generations[example["id"]].append(q)
                else:
                    print("NOT Equivalent")
            
            count += 1
            if count == 1 or count % 100 == 0:
                with open(f'data/generated_paraphrases/{dataset_split}_paraphrases_{args.paraphrase_start}_{args.paraphrase_start+count}.pkl', 'wb') as f:
                    pickle.dump(generations, f)
        
        # Save generations for that split.
        with open(f'data/generated_paraphrases/{dataset_split}_paraphrases_{args.paraphrase_start}_{args.paraphrase_start+count}.pkl', 'wb') as f:
            pickle.dump(generations, f)


if __name__ == "__main__":
    
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    print("STARTING `generate_paraphrases`")
    main(args)
    print("FINISHED `generate_paraphrases`")
