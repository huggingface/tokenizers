"""
Python bindings for Hugging Face's Tokenizers rust library

Encode text to token ids and decode token ids back to text
"""

from pathlib import Path

import huggingface_hub

from .tokenizers import Encoding, Padding, Tokenizer, __version__

__all__ = ["Encoding", "Padding", "Tokenizer", "__version__", "from_pretrained"]


def from_pretrained(
    identifier: str,
    revision: str = "main",
    token: str | bool | None = None,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    subfolder: str | None = None,
    padding: Padding | None = None,
) -> Tokenizer:
    """
    Instantiate a new `Tokenizer` from an existing file on the Hugging Face Hub.

    Defers downloading and caching to `huggingface_hub.hf_hub_download`.

    Args:
        identifier (`str`):
            The *model id* of a repo hosted on huggingface.co that contains a `tokenizer.json` file,
            e.g., `"openai-community/gpt2"`.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            since we use a git-based system for storing models and other artifacts on huggingface.co,
            so `revision` can be any identifier allowed by git.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the
            token generated when running `hf auth login`. If `False`, will send no token. If `None`,
            will use the stored token when there is one.
        cache_dir (`str` or `Path`, *optional*):
            Path to a directory in which the downloaded file should be cached if the standard cache
            should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to download the file again and override the cached version if it exists.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether or not to only rely on the cache and not to attempt to download anything.
        subfolder (`str`, *optional*):
            In case `tokenizer.json` is located inside a subfolder of the model repo on huggingface.co,
            specify it here.
        padding (`Padding`, *optional*):
            Replaces the padding configuration the file declares. `None` keeps the file's.

    Returns:
        `Tokenizer`: The tokenizer the file describes.

    Examples:

    ```python
    # Download tokenizer.json from huggingface.co and cache it.
    tokenizer = Tokenizer.from_pretrained("openai-community/gpt2")

    # Pin a revision: a branch name, a tag name or a commit id.
    tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")

    # A private or gated repo, with the token `hf auth login` stored.
    tokenizer = Tokenizer.from_pretrained("my-org/my-model", token=True)

    # Read the cache without contacting the Hub.
    tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
    ```
    """
    path = huggingface_hub.hf_hub_download(
        repo_id=identifier,
        filename="tokenizer.json",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        subfolder=subfolder,
        library_name="tokenizers",
        library_version=__version__,
    )
    return Tokenizer.from_file(path, padding=padding)


Tokenizer.from_pretrained = staticmethod(from_pretrained)