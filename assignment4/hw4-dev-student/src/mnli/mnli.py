from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any


import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
#from olmo.utils import determine_device
import olmo_core
from olmo_core.utils import get_default_device
import sys
import re

MODEL_MAP = {
    "pretrained": "allenai/OLMo-7B-hf",
    "sft": "allenai/OLMo-7B-SFT-hf",
    "instruct": "allenai/OLMo-7B-Instruct-hf"
}

def load_mnli_train() -> Dataset:
    """you can use this for coming up with a few-shot prompt."""
    
    ds = load_dataset("facebook/anli", split="train_r3").take(100)
    return ds

def load_mnli_dev() -> Dataset:
    """Use this for picking your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="dev_r3").shuffle().take(50)
    return ds

def load_mnli_test() -> Dataset:
    """Use this only AFTER you have picked your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="test_r3").shuffle().take(100)
    return ds

def make_verbalizer(dev_ds:Dataset) -> str:
    """Should return a verbalizer string. You may choose to use examples from the dev set in the verbalizer."""

    # Define label mapping for entailment tasks
    label_map = {
        0: "Entailment",
        1: "Neutral",
        2: "Contradiction"
    }

    # Extract some examples from the dataset
    examples = []
    for example in dev_ds.select(range(3)):  # Select a few examples for the verbalizer
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = label_map[example['label']]

        examples.append(f'Premise: "{premise}"\nHypothesis: "{hypothesis}"\nLabel: {label}\n')

    # Create the verbalizer string with some formatted examples
    verbalizer_prompt = (
        "Given a premise and a hypothesis, classify their relationship as one of the following: "
        "Entailment, Neutral, or Contradiction. Here are some examples:\n\n" +
        "\n".join(examples) +
        "\nNow, classify the following:" +
        "\n "
    )

    return verbalizer_prompt

def make_prompt(verbalizer: str, 
                premise: str, 
                hypothesis:str) -> str:
    """Given a verbalizer, a premise, and a hypothesis, return the prompt."""
    
    return f"{verbalizer}\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\nReturn the predicted label of the question:"


def predict_labels(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        model_type
        ):
    
    """Should return a list of integer predictions (0, 1 or 2), one per prompt."""
    
    label_map = {
        "Entailment": 0,
        "Neutral": 1,
        "Contradiction": 2
    }
    
    
    predictions = []
    
    for prompt in prompts:
        
        # print(prompt)
        # print('#######################################')
        inputs = tokenizer(prompt, 
                           return_tensors = 'pt', 
                           return_token_type_ids = False
                           )
            
        # if model_type == "pretrained":
        #     inputs = tokenizer(prompt, 
        #                        return_tensors = 'pt', 
        #                        return_token_type_ids = False
        #                        )
        # else:
        #     inputs = tokenizer.apply_chat_template(prompt, 
        #                                            tokenize = True, 
        #                                            add_generation_prompt = True, 
        #                                            return_tensors = "pt"
        #                                            )
        # if model_type != "pretrained" and not isinstance(inputs, dict):
        #         inputs = {'input_ids': inputs}
                
        inputs = {k: v.to(device) for k, v in inputs.items()}

    
        response = model.generate(**inputs, 
                                  max_new_tokens = 300
                                  )
        response_decoded = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        # if model_type != "pretrained":
        #         response_decoded = response_decoded.split("\n<|assistant|>\n")[-1]
        
        match = re.search(r"Now, classify the following:.*", 
                          response_decoded,
                          re.DOTALL
                         )
                
        # Print the matched sentence if found
        if match:
            answer = match.group(0)
            for label in label_map.keys():
                if label in answer:
                    predicted_label = label_map[label]
                    predictions.append(predicted_label)
                    break
            else:
                print('match return signal is found, but predicted label cannot found')
                print(response_decoded)
                print('MATCH: ' + answer)
                print('################################')
                predictions.append('unpredicted')
        else:
            print("No matching sentence found.".upper())
            print(response_decoded)
            print('################################')
            predictions.append('unpredicted')


    return predictions

if __name__ == "__main__":
    
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    torch.cuda.empty_cache()  # Clears unused memory
    torch.cuda.ipc_collect()  # Collects unused memory from other processes

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    model_type = 'instruct'
    
    train_ds = load_mnli_train()
    dev_ds = load_mnli_dev()
    test_ds = load_mnli_test()
    
    verbalizer = make_verbalizer(train_ds)
    
    prompts = []
    true_labels = []
    for ex in dev_ds:
        prompt = make_prompt(verbalizer, 
                             ex["premise"], 
                             ex["hypothesis"]
                             )
        prompts.append(prompt)
        true_labels.append(ex["label"])
      
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    olmo_core.utils.DEFAULT_DEVICE = device
    print(get_default_device())
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_type], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_type])
    
    predicted_labels = predict_labels(model,
                                      tokenizer,
                                      prompts,
                                      model_type)
    
    num_correct = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    accuracy = num_correct / len(true_labels)
    print('Total examples: ', len(true_labels))
    print("Accuracy:", accuracy)