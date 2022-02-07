
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from typing import List
import argparse
import shutil
import os
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main(dataset_main_dir:str, output_path:str, languages:List[str]=None):
    model_name="distilbert-base-multilingual-cased"
    batch_size=10
    
    def filter_langs(file_name, languages_to_filter):
        if languages_to_filter:
            if file_name.split(".")[0] in languages_to_filter:
                return True
            else:
                return False
        else:
            return True

    train_ds_path=os.path.join(dataset_main_dir,"train")
    train_ds_files = [os.path.join(train_ds_path,f) for f in os.listdir(train_ds_path) if os.path.isfile(os.path.join(train_ds_path,f)) and f.endswith(".csv") and filter_langs(f, languages)]

    test_ds_path=os.path.join(dataset_main_dir,"test")
    test_ds_files = [os.path.join(test_ds_path,f) for f in os.listdir(test_ds_path) if os.path.isfile(os.path.join(test_ds_path,f)) and f.endswith(".csv") and filter_langs(f, languages)]
    
    

    dataset = load_dataset("csv", data_files=train_ds_files)
    dataset["test"] = load_dataset("csv", data_files=test_ds_files)["train"]

    dataset["train"] = dataset["train"].shuffle()
    dataset["test"] = dataset["test"].shuffle()

    #dataset["train"] = dataset["train"].rename_column("lang", "labels")
    #dataset["test"] = dataset["test"].rename_column("lang", "labels")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len=tokenizer.max_len_single_sentence   

    internal_output_path = "./outputs/model/"
    if not os.path.exists(internal_output_path):
        os.makedirs(internal_output_path)
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    labels=[os.path.basename(file_name).split(".")[0] for file_name in test_ds_files]
    def preprocessing(examples):
         
         result = tokenizer( examples["sentences"],truncation=True, padding=True )
         ilabels = [ labels.index(l) for l in examples["lang"]]
         result["labels"] = ilabels
         return result

    train_dataset = dataset["train"].map(preprocessing, batched=True, batch_size=batch_size)
    
    test_dataset = dataset["test"].map(preprocessing, batched=True, batch_size=batch_size)
    
    

    num_labels = len(labels)
    
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label,
    )


    training_args = TrainingArguments(
        output_dir=internal_output_path+"checkpoints",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        learning_rate=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

        # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train model
    
    train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(internal_output_path) 
    trainer.save_state()

    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    metrics["eval_samples"] = len(eval_result)

    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)

    # evaluate model
    

    kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
    
    kwargs["dataset"] = "wikiann"
    trainer.create_model_card(**kwargs)
    
    if output_path:
        for fileName in os.listdir(internal_output_path):    
            if os.path.isfile(fileName):
                shutil.move(os.path.join(internal_output_path,fileName),output_path)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train model on dataset')
    parser.add_argument('--dataset_main_dir', help='dataset name', default="../open-subtitles-dataset")    
    parser.add_argument('--output_path', help='dataset name', default="../outputs/model")    
    args = parser.parse_args()
    
    main(**vars(args))

    