import click
import evaluate
import json
import logging
import os
import sys

from functools import partial
from pprint import pprint
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Dict, List

from eks.dataset import MUCSUM_TEST
from eks.train import preprocess, MAX_SUMMARY_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")


@click.command()
@click.argument("model_path", type=str)
@click.argument("hub_name", type=str)
@click.option(
    "--device",
    type=click.INT,
    default=0,
    help="the device to on which to load the model",
)
@click.option("--batch-size", type=click.INT, default=1, help="the batch size to use")
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search decoding",
)
@click.option(
    "--num-return-sequences",
    type=click.INT,
    default=1,
    help="number of summaries to generate",
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
    default=MAX_SUMMARY_LENGTH,
    help="maximum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search (eval loop only)",
)
@click.option(
    "--input-format",
    type=click.Choice(
        [
            "template_only",
            "document_only",
            "document_with_type",
            "template_and_document",
        ]
    ),
    default="template_and_document",
    help="the input format",
)
@click.option(
    "--include-slot-descriptions",
    is_flag=True,
    default=False,
    help="whether to include a verbal description of each slot in the linearized template (applies only when input_format = 'template_and_document')",
)
def inference(
    model_path,
    hub_name,
    device,
    batch_size,
    num_beams,
    min_new_tokens,
    max_new_tokens,
    num_return_sequences,
    max_doc_len,
    input_format,
    include_slot_descriptions,
) -> None:
    """Run inference for MUC-4 summarization

    :param model_path: path to the model with which to run inference
    :param hub_name: the name of the underlying model on HuggingFace Hub
    :param device: The GPU on which the model will be loaded
    :param batch_size: The batch size to use
    :param num_beams: The number of beams to use for beam seearch
    :param min_new_tokens: minimum number of tokens to generate in the summary
    :param max_new_tokens: maximum number of tokens to generate in the summary
    :param max_doc_len: maximum number of tokens in the input document (all longer docs will be truncated)
    :param input_format: how the input should be formatted
    :param include_slot_descriptions: whether to include descriptions of the slots
        in the linearized templates
    :param num_return_sequences: The number of summaries to generate for each
        example
    :returns None:
    """
    logger.warning(f"Loading model {model_path} for inference...")
    pipe = pipeline("summarization", model=model_path, device=device)
    logger.warning("...done.")
    if "t5" in hub_name:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if hub_name in {"t5-small", "t5-base"}:
            model_max_length = 512
        else:
            model_max_length = 1024
        # required by T5
        prefix = "summarize: "
    else:
        prefix = None

    preprocess_fn = partial(
        preprocess,
        model=hub_name,
        tokenizer=pipe.tokenizer,
        max_doc_len=max_doc_len,
        input_format=input_format,
        prefix=prefix,
        include_slot_descriptions=include_slot_descriptions,
    )

    test_data = MUCSUM_TEST.map(
        preprocess_fn,
        batched=True,
    )
    preds = []
    for out in tqdm(
        pipe(
            KeyDataset(test_data, "input_str"),
            batch_size=batch_size,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            return_text=True,
        ),
        desc="Evaluating...",
    ):
        preds += list(map(lambda x: x["summary_text"], out))

    scores_dict = score(preds, test_data["target"])
    pprint(scores_dict)

    predictions = []
    for ex, pred in zip(test_data, preds):
        predictions.append(
            {
                "instance_id": ex["instance_id"],
                "format": input_format,
                "document": ex["document"],
                "template": ex["template"],
                "prediction": pred,
                "reference": ex["target"],
            }
        )

    predictions_file = os.path.join(model_path, "../test_preds.jsonl")
    with open(predictions_file, "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(model_path, "../test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    scores_file = os.path.join(model_path, "../test_metrics.json")
    with open(scores_file, "w") as f:
        json.dump(scores_dict, f, indent=2)


def score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE metrics for generated summaries

    :param predictions: the predicted summaries
    :param references: the reference summaries
    :returns: a dictionary of rouge-1, rouge-2, and rouge-L scores
    """
    result = ROUGE.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    inference()
