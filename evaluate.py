import argparse
import json
import logging
import fnmatch
import wandb

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def main():
    limit = 100
    model_type = "hf-causal-experimental"
    models = [
        "hakurei/lit-6B",
        "PygmalionAI/pygmalion-6b",
        "ChaiML/ak_edit_v0",
    ]
    task_names = ['hh', 'chai_davinci', 'chai_synthetic', 'chai_davinci_vs_lit']
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
        print(
            f"{model_type} ({model_args}), limit: {limit}, provide_description: {provide_description}, "
            f"num_fewshot: {num_fewshot}, batch_size: {batch_size}"
        )
        print(evaluator.make_table(results))

        dumped = json.dumps(results, indent=2)
        print(dumped)

        if output_path:
            with open(output_path, "w") as f:
                f.write(dumped)

        wandb_result = {}
        for key, value in results["results"].items():
            wandb_result[key][model] = value
        wandb.log(
            wandb_result
        )


if __name__ == "__main__":
    main()
