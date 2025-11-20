import os
import numpy as np
from huggingface_hub import InferenceClient

client = InferenceClient(model='microsoft/deberta-v2-xlarge-mnli')

completion = client.text_classification("the weather is good[SEP]the weather is good and i like you")

print(completion[0].label)