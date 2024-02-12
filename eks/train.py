import click
import datasets
import evaluate
import json
import logging
import numpy as np
import os
import sys
import transformers

from eks.dataset import (
    MUCSUM_TRAIN,
    MUCSUM_DEV,
    MUCSUM_TEST,
)
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from functools import partial
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import PredictionOutput
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SUMMARIZATION_MODELS = {
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "google/pegasus-large",
    "google/pegasus-cnn_dailymail",
    "t5-large",
}
DEFAULT_MODEL = "facebook/bart-large-cnn"
TEMPLATE_TYPES = {
    "attack",
    "arson",
    "bombing",
    "forced work stoppage",
    "kidnapping",
    "robbery",
}
TEMPLATE_FIELDS = {
    "type": "event type",
    "completion": "stage of completion",
    "date": "date",
    "location": "location",
    "perpind": "individual perpetrators",
    "perporg": "organizations responsible",
    "target": "physical targets",
    "victim": "victims",
    "weapon": "weapons",
}

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")

MAX_SUMMARY_LENGTH = 256


@click.command()
@click.argument(
    "output_dir",
    type=str,
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(SUMMARIZATION_MODELS),
    default=DEFAULT_MODEL,
    help="the summarization model to train",
)
@click.option(
    "--num-epochs", type=click.INT, default=30, help="maximum training epochs"
)
@click.option(
    "--max-doc-len",
    type=click.INT,
    default=None,
    help="maximum number of tokens in the input document (all longer docs will be truncated)",
)
@click.option(
    "--min-new-tokens",
    type=click.INT,
    default=15,
    help="the minimum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=256,
    help="maximum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search (eval loop only)",
)
@click.option(
    "--gradient-checkpointing",
    is_flag=True,
    default=False,
    help="whether to use gradient checkpointing for training",
)
@click.option(
    "--input-format",
    type=click.Choice(["template_only", "document_only", "document_with_type", "template_and_document"]),
    default="template_and_document",
    help="the input format",
)
@click.option(
    "--include-slot-descriptions",
    is_flag=True,
    default=False,
    help="whether to include a verbal description of each slot in the linearized template (applies only when input_format = 'template_and_document')",
)
@click.option("--seed", type=int, default=1337, help="the random seed for training")
def train(
    output_dir,
    model,
    num_epochs,
    max_doc_len,
    min_new_tokens,
    max_new_tokens,
    num_beams,
    gradient_checkpointing,
    input_format,
    include_slot_descriptions,
    seed,
) -> None:
    """Train a summarization model for MUC-4

    :param output_dir: the directory where checkpoints will be saved
    :param model: a string indicating the HuggingFace base model to be fine-tuned
    :param num_epochs: the number of epochs for which training will be run
    :param max_doc_len: the maximum length of an input document (documents longer
        than this will be truncated)
    :param min_new_tokens: minimum number of tokens to generate in the summary (for evaluation)
    :param max_new_tokens: maximum number of tokens to generate in the summary (for evaluation)
    :param num_beams: number of beams to use for beam search (eval loop only)
    :param gradient_checkpointing: whether to use gradient checkpointing for training
    :param input_format: how the input should be formatted
    :param include_slot_descriptions: whether to include a description of each slot
        in the linearized template
    :param seed: the random seed to use
    :return: None
    """
    m = AutoModelForSeq2SeqLM.from_pretrained(model)
    if "t5" in model:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if model in {"t5-small", "t5-base"}:
            model_max_length = 512
        else:
            model_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            model, model_max_length=model_max_length
        )
        # required by T5
        prefix = "summarize: "
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        prefix = None

    train_data = MUCSUM_TRAIN
    dev_data = MUCSUM_DEV
    test_data = MUCSUM_TEST

    preprocess_fn = partial(
        preprocess,
        model=model,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
        input_format=input_format,
        prefix=prefix,
        include_slot_descriptions=include_slot_descriptions,
    )
    train_dataset = train_data.map(preprocess_fn, batched=True)
    eval_dataset = dev_data.map(preprocess_fn, batched=True)
    test_dataset = test_data.map(preprocess_fn, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=m)

    # Load model's default generation config, but
    # override with user-provided parameters
    assert m.generation_config is not None
    generation_config = m.generation_config
    generation_config.min_new_tokens = min_new_tokens
    generation_config.max_new_tokens = max_new_tokens
    generation_config.num_beams = num_beams

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        metric_for_best_model="rouge1",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=gradient_checkpointing,
        predict_with_generate=True,
        generation_config=generation_config,
        seed=seed,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.warning(f"(Min, Max) summary length: ({min_new_tokens}, {max_new_tokens})")
    logger.warning(f"Using beam size = {num_beams}")
    metrics = partial(compute_metrics, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=m,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    save_predictions(
        prediction_output, test_dataset, tokenizer, output_dir, input_format
    )


def save_predictions(
    prediction_output: PredictionOutput,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    model_path: str,
    input_format: str,
) -> None:
    decoded_preds = [
        p.lower()
        for p in tokenizer.batch_decode(
            prediction_output.predictions, skip_special_tokens=True
        )
    ]
    predictions = []
    assert len(decoded_preds) == len(test_dataset)
    for i, pred in enumerate(decoded_preds):
        predictions.append(
            {
                "instance_id": test_dataset[i]["instance_id"],
                "format": input_format,
                "document": test_dataset[i]["source"],
                "template": test_dataset[i]["template"],
                "prediction": pred,
                "reference": test_dataset[i]["target"],
            }
        )
    with open(os.path.join(model_path, "test_preds.jsonl"), "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(model_path, "test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    metrics = prediction_output.metrics
    with open(os.path.join(model_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def preprocess(
    examples: LazyBatch,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    input_format: str,
    include_slot_descriptions: bool = False,
    prefix: Optional[str] = None,
    max_doc_len: Optional[int] = None,
) -> BatchEncoding:
    """Preprocess MUC-4 summarization data

    Code is taken with light adaptation from the example at the following URL:
    https://huggingface.co/docs/transformers/tasks/summarization#preprocess

    :param examples: the examples to be preprocessed
    :param model: a string denoting the HuggingFace model associated with the `tokenizer`
    :param tokenizer: the tokenizer that will be used to tokenize each example
    :param input_format: how the input should be formatted
    :param include_slot_descriptions: whether to include the descriptions of the slots
        in the linearized template (applies only when input_format = "template_and_document")
    :param prefix: an optional prefix to prepend to the document text (necessary for
        certain pretrained models, like T5)
    :param max_doc_len: the maximum length of an input document (defaults to the
        maximum model length)
    :param template_only: whether to include only the template in the prompt
    :return: the preprocessed data
    """

    def format_template(template: Dict[str, str | List[str]]) -> str:
        # a special token for separating different slots
        # we try to use a token that does not overload the sep_token,
        # since we use the sep_token to separate the documetn and
        # the template. sometimes this isn't possible.
        if tokenizer.additional_special_tokens:
            slot_sep_token = tokenizer.additional_special_tokens[0]
        elif tokenizer.sep_token:
            # BART: just fall back to regular sep_token
            slot_sep_token = tokenizer.sep_token
        else:
            raise ValueError(f"Could not configure slot_sep_token for model!")

        # each template is formatted as follows:
        # <slot_sep> slot1_filler1, ..., slot1_fillerN <slot_sep> slot2_filler1, ..., slot2_fillerM <slot_sep> ...
        template_str = [slot_sep_token]
        for slot, description in sorted(TEMPLATE_FIELDS.items()):
            if slot in template:
                if slot in {"type", "completion"}:
                    value = template[slot]
                elif template[slot]:
                    value = ", ".join(template[slot])
                if include_slot_descriptions:
                    value = f"{description}: {value}"
                template_str.append(value)
            template_str.append(slot_sep_token)
        return "".join(template_str)

    model_input_dim = tokenizer.model_max_length
    assert (
        not max_doc_len or max_doc_len < model_input_dim
    ), f"Maximum document length ({max_doc_len}) > model input dimension ({model_input_dim})"
    if max_doc_len:
        logger.warning(f"Maximum document length: {max_doc_len}")
    else:
        logger.warning(f"Maximum document length: {model_input_dim}")
    prefix = prefix if prefix else ""
    if input_format == "template_only":
        templates = [prefix + format_template(t) for t in examples["template"]]
        model_inputs = tokenizer(
            templates, max_length=max_doc_len, padding="max_length", truncation=True
        )
    elif input_format == "document_only":
        docs = [prefix + doc for doc in examples["source"]]
        model_inputs = tokenizer(
            text=docs,
            padding="max_length",
            truncation=True
        )
    elif input_format == "document_with_type":
        docs = [prefix + doc for doc in examples["source"]]
        template_types = [t["type"] for t in examples["template"]]
        # input structure is <document> [SEP] <template_type>
        # if truncation is needed, the document is what gets truncated
        model_inputs = tokenizer(
            text=docs,
            text_pair=template_types,
            padding="max_length",
            truncation="only_first",
        )
    else:
        # input structure is <document> [SEP] <formatted_template>
        # as above, if truncation is needed, the document is what gets truncated
        assert (
            input_format == "template_and_document"
        ), f"Unrecognized input format '{input_format}'"
        docs = [prefix + doc for doc in examples["source"]]
        template_types = [format_template(t) for t in examples["template"]]
        model_inputs = tokenizer(
            text=docs,
            text_pair=template_types,
            padding="max_length",
            truncation="only_first",
        )

    # save the raw input format to a slot in the dataset (used during inference)
    input_str = tokenizer.batch_decode(model_inputs["input_ids"])
    labels = tokenizer(text_target=examples["target"], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["input_str"] = input_str
    return model_inputs


def compute_metrics(
    eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizerBase
) -> Dict[str, float]:
    """Compute summarization metrics

    Taken from the example at the following URL:
    https://huggingface.co/docs/transformers/tasks/summarization#evaluate

    :param eval_pred: the prediction to be evaluated
    :param tokenizer: the tokenizer associated with the model
    :return: a dictionary containing various metrics, including loss, rouge-1
        rouge-2, rouge-L, and the mean length of the generated summaries
    """
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # we lowercase because the original MUC data is lowercased
    decoded_preds = [
        p.lower() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ]
    decoded_labels = [
        l.lower() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)
    ]

    # automatic metrics include ROUGE-{1,2,L}, METEOR, and BERTscore
    result = ROUGE.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result["meteor"] = METEOR.compute(
        predictions=decoded_preds, references=decoded_labels
    )["meteor"]
    bertscore = BERT_SCORE.compute(
        predictions=decoded_preds, references=decoded_labels, lang="en"
    )
    for m, metric in zip(["p", "r", "f1"], ["precision", "recall", "f1"]):
        result["bertscore_" + m] = np.mean(bertscore[metric])

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["mean_summary_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    train()
