import multiprocessing as mp
import os
import requests
import pytest

DATA_PATH = os.path.join("tests", "data")


def download(url, with_filename=None):
    filename = with_filename if with_filename is not None else url.rsplit("/")[-1]
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


@pytest.fixture(scope="session")
def train_files(data_dir):
    big = download("https://norvig.com/big.txt")
    small = os.path.join(DATA_PATH, "small.txt")
    with open(small, "w") as f:
        with open(big, "r") as g:
            for i, line in enumerate(g):
                f.write(line)
                if i > 100:
                    break
    return {
        "small": small,
        "big": big,
    }


@pytest.fixture(scope="session")
def albert_base(data_dir):
    return download(
        "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-tokenizer.json"
    )


@pytest.fixture(scope="session")
def doc_wiki_tokenizer(data_dir):
    return download(
        "https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-quicktour/tokenizer.json",
        "tokenizer-wiki.json",
    )


@pytest.fixture(scope="session")
def doc_pipeline_bert_tokenizer(data_dir):
    return download(
        "https://s3.amazonaws.com/models.huggingface.co/bert/anthony/doc-pipeline/tokenizer.json",
        "bert-wiki.json",
    )


def multiprocessing_with_parallelism(tokenizer, enabled: bool):
    """
    This helper can be used to test that disabling parallelism avoids dead locks when the
    same tokenizer is used after forking.
    """
    # It's essential to this test that we call 'encode' or 'encode_batch'
    # before the fork. This causes the main process to "lock" some resources
    # provided by the Rust "rayon" crate that are needed for parallel processing.
    tokenizer.encode("Hi")
    tokenizer.encode_batch(["hi", "there"])

    def encode(tokenizer):
        tokenizer.encode("Hi")
        tokenizer.encode_batch(["hi", "there"])

    # Make sure this environment variable is set before the fork happens
    os.environ["TOKENIZERS_PARALLELISM"] = str(enabled)
    p = mp.Process(target=encode, args=(tokenizer,))
    p.start()
    p.join(timeout=1)

    # At this point the process should have successfully exited, depending on whether parallelism
    # was activated or not. So we check the status and kill it if needed
    alive = p.is_alive()
    if alive:
        p.terminate()

    assert (alive and mp.get_start_method() == "fork") == enabled
