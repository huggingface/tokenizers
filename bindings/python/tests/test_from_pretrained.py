import shutil
from pathlib import Path

import huggingface_hub
import pytest

from conftest import WIKI
from tokenizers import Padding, Tokenizer, __version__, from_pretrained

TEXT = "Hello there"


@pytest.fixture
def recorded_download(monkeypatch):
    """Stands in for `huggingface_hub.hf_hub_download`: hands back the wiki fixture, records the call."""
    seen = {}

    def hf_hub_download(**kwargs):
        seen.update(kwargs)
        return str(WIKI)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return seen


def test_is_also_reachable_as_a_tokenizer_staticmethod():
    assert Tokenizer.from_pretrained is from_pretrained


@pytest.mark.parametrize("load", [from_pretrained, Tokenizer.from_pretrained], ids=["module", "staticmethod"])
def test_downloads_tokenizer_json_from_the_model_repo(recorded_download, wiki, load):
    tokenizer = load("some-org/some-model")

    assert tokenizer.encode(TEXT).ids == wiki.encode(TEXT).ids
    assert recorded_download == {
        "repo_id": "some-org/some-model",
        "filename": "tokenizer.json",
        "revision": "main",
        "token": None,
        "cache_dir": None,
        "force_download": False,
        "local_files_only": False,
        "subfolder": None,
        "library_name": "tokenizers",
        "library_version": __version__,
    }


@pytest.mark.parametrize("token", ["hf_abc", True])
def test_forwards_every_download_option(recorded_download, tmp_path, token):
    from_pretrained(
        "some-org/some-model",
        revision="v2",
        token=token,
        cache_dir=tmp_path,
        force_download=True,
        local_files_only=True,
        subfolder="tokenizer",
    )

    assert recorded_download["revision"] == "v2"
    assert recorded_download["token"] == token
    assert Path(recorded_download["cache_dir"]) == tmp_path
    assert recorded_download["force_download"] is True
    assert recorded_download["local_files_only"] is True
    assert recorded_download["subfolder"] == "tokenizer"


def test_takes_a_padding_like_from_file(recorded_download):
    tokenizer = from_pretrained("some-org/some-model", padding=Padding(length=8))

    assert tokenizer.padding == Padding(length=8)


def test_reads_the_hub_cache_without_network(tmp_path, wiki):
    # The cache layout `hf_hub_download` documents: `refs/<branch>` names a commit, the file
    # sits under `snapshots/<commit>/`.
    commit = "0" * 40
    repo = tmp_path / "models--some-org--some-model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "snapshots" / commit).mkdir(parents=True)
    shutil.copy(WIKI, repo / "snapshots" / commit / "tokenizer.json")

    tokenizer = from_pretrained("some-org/some-model", cache_dir=tmp_path, local_files_only=True)

    assert tokenizer.encode(TEXT).ids == wiki.encode(TEXT).ids


def test_reports_a_repo_missing_from_the_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_pretrained("some-org/does-not-exist", cache_dir=tmp_path, local_files_only=True)


@pytest.mark.network
def test_honours_the_revision():
    text = "Hey there dear friend!"
    # `main` is a lowercasing WordPiece tokenizer, the `gpt-2` branch a byte-level BPE.
    lowercasing = from_pretrained("anthony/tokenizers-test")
    byte_level = from_pretrained("anthony/tokenizers-test", revision="gpt-2")

    assert lowercasing.decode(lowercasing.encode(text).ids) == "hey there dear friend!"
    assert byte_level.decode(byte_level.encode(text).ids) == "Hey there dear friend!"
