# Event-Keyed Summarization

This is the official repository for the MUCSUM dataset and for the following paper:

>*Event-Keyed Summarization.* William Gantt, Alexander Martin, Pavlo Kuchmiichuk, and Aaron Steven White.

If you use MUCSUM or any of the code or resources in this repository, we ask that you please cite this paper.

## Project Structure

This project contains the following directories:

- `data`: contains the MUCSUM dataset, as well as results from our human evaluation study
- `eks`: source code for running training and inference on MUCSUM
- `model_outputs`: model predictions and metric scores for both the fine-tuned and the few-shot models described in the paper

## Getting Started

If you just wish to use the MUCSUM data and do not need any other resources from this repository, no setup is required. The MUCSUM data can be found in `data/mucsum`. Otherwise, please follow the instructions below.

This project relies on [poetry](https://python-poetry.org/) for dependency installation and management. If you do not already have poetry installed, you can find instructions for doing so [here](https://python-poetry.org/docs/#installation). Poetry relies on a virtual environment to isolate dependencies. We used Conda for this purpose (example below), but you can use whatever tool you find most convenient. We developed this project using Python 3.10.13, so, using Conda, you can create a new virtual environment as follows.

```
conda create --name eks python=3.10.13
```

This project also uses PyTorch and one shortcoming of poetry is that using it to install PyTorch can be something of a pain. We wrote our code using PyTorch version 2.0.1 and CUDA 11.7 on a Linux machine, and have found it easiest to simply specify the PyTorch dependency in the `pyproject.toml` file via a URL to the appropriate PyTorch wheel, as specified in "Option 1" [here](https://github.com/python-poetry/poetry/issues/6409). As such, we suggest you change the URL in the `torch` dependency line in the `pyproject.toml` line to point to the wheel that matches your operating system and CUDA version (we recommend trying to keep the PyTorch version as 2.0.1 if possible).

After you have made the above adjustment (if necessary), and with your virtual environment activated, run the following from the project root to install all necessary dependencies:

```
poetry install
```

Before running training or inference, please be sure to set the `PROJECT_ROOT` variable at the top of `eks/dataset.py`

## Training

To replicate our training runs, you can run the following command from the project root

```
python eks/train.py $MODEL_DIR --model $MODEL_NAME --input-format $INPUT_FORMAT --seed $SEED --include-slot-descriptions [--gradient-checkpointing]
```

where:
- `$MODEL_DIR` is a path to the directory where you would like your trained model to be saved
- `$MODEL_NAME` is one of `facebook/bart-large`, `google/pegasus-large`, or `t5-large`
- `$INPUT_FORMAT` is one of `template_only`, `document_only`, or `template_and_document` (see paper)
- `$SEED` is any non-negative integer (we use 1337, 1338, and 1339 in our experiments)

Depending on the GPU you are using, you may find it helpful to enable [gradient checkpointing](https://huggingface.co/docs/transformers/v4.18.0/en/performance) (`--gradient-checkpointing`) to prevent memory errors. Additional training parameters can be seen in `train.py`, though the ones specified above are all you need to replicate our results.

## Inference

Invoking `train.py` as described above will also run inference on the test set when training has finished. However, if you need to run inference again on a saved model for any reason, you can do so by running the following command from the project root:

```
python eks/inference.py $CHECKPOINT_DIR $MODEL_NAME --device $DEVICE_NUM --input-format $INPUT_FORMAT --include-slot-descriptions
```

where:
- `$CHECKPOINT_DIR` is the path to your saved model (should be of the form `$MODEL_DIR/checkpoint-<N>`, where `$MODEL_DIR` is as described in [training](#training) above)
- `$MODEL_NAME` is the name of the pretrained Transformer model used in training your saved model (one of `facebook/bart-large`, `google/pegasus-large`, `t5-large`)
- `DEVICE_NUM` is an ordinal value indicating the device on which you want to run inference
- `INPUT_FORMAT` is one of `template_only`, `document_only`, or `template_and_document` (see paper)

## TODOs

We are still in the process of uploading source code for computing certain metrics reported in the paper (CEAF-REE, NLI metrics).

## Questions

If you have any further questions or encounter any issues, please create an issue on GitHub and we will try to respond promptly.