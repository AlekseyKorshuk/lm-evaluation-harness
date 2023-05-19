import argparse
import json
import logging
import fnmatch
import wandb
import pandas as pd

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


def main():
    limit = 1000
    model_type = "hf-causal-experimental"
    models = [
        "PygmalionAI/pygmalion-6b",
        "ChaiML/ak_edit_v0",
        "hakurei/lit-6B",
        "EleutherAI/gpt-j-6b",
    ]
    task_names = [
        # 'truthfulqa_mc',
        # 'arc_challenge',
        # 'hellaswag',
        # 'hh',
        'chai_davinci',
        'chai_synthetic',
        'chai_davinci_vs_lit',
    ]
    description_dict_path = None
    num_fewshot = 0
    check_integrity = False
    decontamination_ngrams_path = None
    provide_description = False
    batch_size = 4
    device = None
    output_path = None
    no_cache = False

    print(f"Selected Tasks: {task_names}")
    description_dict = {}
    if description_dict_path:
        with open(description_dict_path, "r") as f:
            description_dict = json.load(f)

    run = wandb.init(
        # Set the project where this run will be logged
        project="lmeh-chai",
        # Track hyperparameters and run metadata
        config=dict(
            models=models,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            no_cache=no_cache,
            limit=limit,
            description_dict=description_dict,
            decontamination_ngrams_path=decontamination_ngrams_path,
            check_integrity=check_integrity,
        )
    )
    metrics = ["acc", "acc_norm", "mc2"]
    stats = {}
    for model in models:
        model_args = f"pretrained={model},dtype=float16"
        results = evaluator.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            no_cache=no_cache,
            limit=limit,
            description_dict=description_dict,
            decontamination_ngrams_path=decontamination_ngrams_path,
            check_integrity=check_integrity,
        )
        stats[model] = results["results"]
        print(
            f"{model_type} ({model_args}), limit: {limit}, provide_description: {provide_description}, "
            f"num_fewshot: {num_fewshot}, batch_size: {batch_size}"
        )
        print(evaluator.make_table(results))

        dumped = json.dumps(results, indent=2)
        print(dumped)

    for metric in metrics:
        table_data = {}
        for model_name, results in stats.items():
            model_data = {}
            for task_name, model_metrics in stats[model_name].items():
                model_data[task_name] = model_metrics.get(metric)
            table_data[model_name] = model_data
        df = pd.DataFrame(table_data).transpose()
        df["Model"] = list(df.index)
        df['mean'] = df.mean(axis=1, numeric_only=True)
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]
        table = wandb.Table(dataframe=df)
        wandb.log(
            {
                f"tables/{metric}": table
            }
        )


if __name__ == "__main__":
    main()
