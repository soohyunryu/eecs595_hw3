import argparse
import os
import random
import torch
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_data(tokenizer, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    def tokenize_function(examples):
        tokenized_examples = tokenizer(...)
        return tokenized_examples

    dataset = load_dataset(params.dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataloader = DataLoader(...)
    eval_dataloader = DataLoader(...)
    test_dataloader = DataLoader(...)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return train_dataloader, eval_dataloader, test_dataloader


def finetune(model, train_dataloader, eval_dataloader, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    optimizer = ...
    lr_scheduler = ...

    for epoch in range(params.num_epochs):

        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


def test(model, test_dataloader, prediction_save='predictions.torch'):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    metric = evaluate.load(...)
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print('Test Accuracy:', score['accuracy'])
    torch.save(all_predictions, prediction_save)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, train_dataloader, eval_dataloader, params)

    test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)

    params, unknown = parser.parse_known_args()
    main(params)