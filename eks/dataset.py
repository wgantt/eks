import json
import os

from datasets import Dataset
from functools import partial

# TODO: set this to the root of the project
PROJECT_ROOT = "/brtx/605-nvme1/wgantt/eks"
if not PROJECT_ROOT:
    raise ValueError(
        "Please set the PROJECT_ROOT variable to the root of the project directory"
    )

OFFICIAL_ROOT = os.path.join(PROJECT_ROOT, "data/mucsum")
TRAIN_DATA_OFFICIAL = os.path.join(OFFICIAL_ROOT, "train.json")
DEV_DATA_OFFICIAL = os.path.join(OFFICIAL_ROOT, "dev.json")
TEST_DATA_OFFICIAL = os.path.join(OFFICIAL_ROOT, "test.json")
OFFICIAL_DATA_BY_SPLIT = {
    "train": TRAIN_DATA_OFFICIAL,
    "dev": DEV_DATA_OFFICIAL,
    "test": TEST_DATA_OFFICIAL,
}

DATA_BY_TYPE = {
    "official": OFFICIAL_DATA_BY_SPLIT,
}


def gen(split: str, type: str):
    assert (
        type in DATA_BY_TYPE
    ), f"Invalid choice ({type}) for 'type' parameter. Choices are: {sorted(DATA_BY_TYPE.keys())}"
    data_by_split = DATA_BY_TYPE[type]
    with open(data_by_split[split]) as f:
        d = json.load(f)

    for k, v in sorted(d.items()):
        for item in v:
            doc_id = k
            instance_id = item["instance_id"]
            document = " ".join(item["document"])
            summary = " ".join(item["summary"])
            template = item["template"]
            ex = {
                "doc_id": doc_id,
                "instance_id": instance_id,
                "source": document,
                "target": summary,
                "template": template,
            }
            yield ex


MUCSUM_TRAIN = Dataset.from_generator(
    partial(gen, split="train", type="official")
)
MUCSUM_DEV = Dataset.from_generator(partial(gen, split="dev", type="official"))
MUCSUM_TEST = Dataset.from_generator(partial(gen, split="test", type="official"))