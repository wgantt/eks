# Model Outputs

This directory contains test set predictions from all models reported in the paper. This includes both:

- **Fine-Tuned Models**: BART-large, T5-large, and PEGASUS-large fine-tuned on the MUCSUM training split. Results in the paper reflect macro-averages across three separate training runs, each using a different random seed (1337, 1338, 1339). Predictions from each run are included in JSONlines format as `{seed}_test_preds.jsonl`
- **Zero-Shot Models**: gpt-3.5-turbo and gpt4 evaluated zero-shot. Results in the paper reflect macro-averages across three slightly different prompts (`prompt1`, `prompt2`, `prompt3`). Predictions using each prompt are included in JSONL format as `prompt{N}_test_preds.jsonl`.

The fine-tuned models are evaluated in three different settings. In all cases, they model must generate a summary keyed to a *specific* event as output:

- **Template Only** (`temp_only`): Only the linearized template for the target event is provided as input.
- **Document Only** (`doc_only`): Only the document describing the target event is provided as input.
- **Template and Document** (`temp_and_doc`): Both the document and the template are provided as input.

Alongside the model predictions, for all combinations of model and setting, we include two types of metrics files:

- `*metrics.json`: BERTScore, ROUGE-1, ROUGE-2, and ROUGE-L scores for the whole test set
- `*metrics_per_example.json`: BERTScore, ROUGE-1, ROUGE-2, and ROUGE-L scores by example

The aggregate test set scores are computed automatically when either `train.py` or `inference.py` is run. Instructions for computing the other metrics (CEAF-REE and the NLI metrics) will be added soon.