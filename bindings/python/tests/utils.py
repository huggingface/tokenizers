import os
import requests
import pytest

DATA_PATH = os.path.join("tests", "data")


def download(url):
    filename = url.rsplit("/")[-1]
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(1024):
                f.write(chunk)
    return filepath


@pytest.fixture(scope="session")
def data_dir():
    assert os.getcwd().endswith("python")
    exist = os.path.exists(DATA_PATH) and os.path.isdir(DATA_PATH)
    if not exist:
        os.mkdir(DATA_PATH)


@pytest.fixture(scope="session")
def roberta_files(data_dir):
    return {
        "vocab": download(
            "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json"
        ),
        "merges": download(
            "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt"
        ),
    }


@pytest.fixture(scope="session")
def bert_files(data_dir):
    return {
        "vocab": download(
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
        ),
    }


@pytest.fixture(scope="session")
def openai_files(data_dir):
    return {
        "vocab": download(
            "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json"
        ),
        "merges": download(
            "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt"
        ),
    }
