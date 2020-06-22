from multiprocessing import Process
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


def encode_decode_in_subprocess(tokenizer):
    # It's essential to this test that we call 'encode' or 'encode_batch'
    # before the fork. This causes the main process to "lock" some resources
    # provided by the Rust "rayon" crate that are needed for parallel processing.
    tokenizer.encode("Hi")
    tokenizer.encode_batch(["hi", "there"])

    def encode():
        encoding = tokenizer.encode("Hi")
        tokenizer.decode(encoding.ids)

    p = Process(target=encode)
    p.start()
    p.join(timeout=1)

    # At this point the process should have successfully exited.
    # If the subprocess is still alive, the test have failed.
    # But we want terminate that process anyway otherwise pytest might hang forever.
    if p.is_alive():
        p.terminate()
        assert False, "tokenizer in sub process caused dead lock"

    assert p.exitcode == 0
