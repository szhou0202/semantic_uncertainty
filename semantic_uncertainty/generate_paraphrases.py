from uncertainty.utils import utils
from uncertainty.data.data_utils import load_ds

def main(args):
    # For each question in dataset, produce a set of paraphrases
    # Check each paraphrase is semantically equivalent to original by checking bidirectional entailment of question+answer pairs
    # Save phrases
    train_data, val_data = load_ds(args.dataset, seed=1)
    print(train_data.shape, val_data.shape)

    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    args.model_max_new_tokens = 100
    # Initialize model.
    model = utils.init_model(args)
    
    num_generations = 3

    generations = {}
    for dataset_split in ['train', 'validation']:
        dataset = train_data if dataset_split == 'train' else val_data

        for example in dataset:
            question, answer = example["question"], example['answers']['text']

            generations[example["id"]] = []

            prompt = f"For the following question Q, please provide {num_generations} paraphrases of the question that are semantically equivalent to it. Q: {question} Paraphrased questions:"
            print(prompt)
            response, _, _ = model.predict(prompt, 1.0)
            print(response)
            paraphrases = [res[3:] for res in response.split("? ")]
            paraphrases = [res if res[-1]=='?' else res+'?' for res in paraphrases]
            print(paraphrases)
            break

if __name__ == "__main__":
    
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    print("STARTING `generate_paraphrases`")
    main(args)
    print("FINISHED `generate_paraphrases`")
