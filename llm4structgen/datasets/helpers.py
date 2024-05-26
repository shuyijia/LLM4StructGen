import torch
from dataclasses import dataclass
import transformers

from llm4structgen.datasets.cartesian_dataset import CartesianDataset
from llm4structgen.datasets.internal_dataset import InternalCoordinatesDataset
from llm4structgen.constants import *

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances)
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances] 
                for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_datasets(args, tokenizer):
    format_options = {
        "permute_composition": args.format_permute_composition,
        "permute_structure": args.format_permute_structure,
    }

    if args.dataset_type in ["cif", "cartesian"]:
        dataset_class = CartesianDataset
    elif args.dataset_type in ["zmatrix", "internal"]:
        dataset_class = InternalCoordinatesDataset
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    datasets = {
        "train": dataset_class(
            str(args.data_path / "train.csv"), 
            format_options,
            llama_tokenizer=tokenizer,
            w_attributes=args.w_attributes,
            task_probabilities=args.task_probabilities,
            add_perturbed_example=args.add_perturbed_example,
            permutation_invariance=args.permutation_invariant
        ),
        "val": dataset_class(
            str(args.data_path / "val.csv"),
            format_options,
            llama_tokenizer=tokenizer,
            w_attributes=args.w_attributes,
            task_probabilities=args.task_probabilities,
            add_perturbed_example=args.add_perturbed_example,
            permutation_invariance=args.permutation_invariant
        ),
    }

    return datasets