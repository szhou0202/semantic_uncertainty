# An investigation into question-based methods for uncertainty detection in LLMs

This repository builds on the original, now deprecated codebase for semantic uncertainty at [https://github.com/lorenzkuhn/semantic_uncertainty](https://github.com/lorenzkuhn/semantic_uncertainty).

## Software Dependencies

Our code relies on Python 3.11 with PyTorch-CUDA 11.8

We ran experiments on the Bouchet HPC cluster.

In [environment.yml](environment.yml) we list the exact versions for all Python packages used in our experiments.

Our experiments rely on Hugging Face for all LLM models and most of the datasets.
It may be necessary to set the environment variable `HF_TOKEN` to the token associated with your Hugging Face account.
Further, it may be necessary to [apply for access](https://huggingface.co/meta-llama) to use the official repository of Meta's LLaMa-2 models.


## Reproducing Results

To generate paraphrases, execute

```
python semantic_uncertainty/generate_paraphrases_keep_unentailed.py --model_name=Llama-2-7b-chat
```
Due to the large dataset, we did not run the script all the way through and instead added a check to save results (in pickle files) after each 100 iterations.

The final paraphrasing results file should be renamed "train_paraphrases.pkl" (in the "data" folder) and to be used for the next step.

Next, to generate answers to the original and paraphrased questions, execute
```
python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa --using_paraphrases --metric=llm
```

This should generate "paraphrase_answers.pkl" and "uncertainty_paraphrase_answers.pkl" (with baseline and other measures) in the data folder.

Finally, to calculate semantic entropy and paraphrase-augmented semantic entropy, execute
```
python semantic_uncertainty/compute_uncertainty_paraphrases.py
```

This should generate "combined_uncertainty_measures_paraphrases.pkl" in the data folder, which will contain a dictionary with several uncertainty measures, including P(false), semantic entropy, and paraphrase-augmented semantic entropy.

We visualize semantic entropy results in "paraphrase_eval.ipynb", in the "notebooks" folder. Some results at the bottom are for our ablation experiment, which used slightly altered scripts (available in this repository) and rely on different data files.
