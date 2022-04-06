## How to release

# Before the release

Simple checklist on how to make releases for `tokenizers`.

- Freeze `master` branch.
- Run all tests (Check CI has properly run)
- If any significant work, check benchmarks:
  - `cd tokenizers && cargo bench` (needs to be run on latest release tag to measure difference if it's your first time)
- Run all `transformers` tests. (`transformers` is a big user of `tokenizers` we need
  to make sure we don't break it, testing is one way to make sure nothing unforeseen
  has been done.
  - Run all fast tests at the VERY least (not just the tokenization tests).
- **If any breaking change has been done**, make sure the version can safely be increased for transformers users (`tokenizers` version need to make sure users don't upgrade before `transformers` has). [link](https://github.com/huggingface/transformers/blob/main/setup.py#L154)
  For instance `tokenizers>=0.10,<0.11` so we can safely upgrade to `0.11` without impacting
  current users
- Then start a new PR containing all desired code changes from the following steps.
- You will `Create release` after the code modifications are on `master`.

# Rust

- `tokenizers` (rust, python & node) versions don't have to be in sync but it's
  very common to release for all versions at once for new features.
- Edit `Cargo.toml` to reflect new version
- Edit `CHANGELOG.md`:
    - Add relevant PRs that were added (python PRs do not belong for instance).
    - Add links at the end of the files.
- Go to [Releases](https://github.com/huggingface/tokenizers/releases)
- Create new Release:
    - Mark it as pre-release
    - Use new version name with a new tag (create on publish) `vX.X.X`.
    - Copy paste the new part of the `CHANGELOG.md`
- !! Click on `Publish release`. This will start the whole process of building a uploading
  the new version on `crates.io`, there's no going back after this
- Go to the [Actions](https://github.com/huggingface/tokenizers/actions) tab and check everything works smoothly.
- If anything fails, you need to fix the CI/CD to make it work again. Since your package was not uploaded to the repository properly, you can try again.


# Python

- Edit `bindings/python/setup.py` to reflect new version.
- Edit `bindings/python/py_src/tokenizers/__init__.py` to reflect new version.
- Edit `CHANGELOG.md`:
    - Add relevant PRs that were added (node PRs do not belong for instance).
    - Add links at the end of the files.
- Go to [Releases](https://github.com/huggingface/tokenizers/releases)
- Create new Release:
    - Mark it as pre-release
    - Use new version name with a new tag (create on publish) `python-vX.X.X`.
    - Copy paste the new part of the `CHANGELOG.md`
- !! Click on `Publish release`. This will start the whole process of building a uploading
  the new version on `pypi`, there's no going back after this
- Go to the [Actions](https://github.com/huggingface/tokenizers/actions) tab and check everything works smoothly.
- If anything fails, you need to fix the CI/CD to make it work again. Since your package was not uploaded to the repository properly, you can try again.
- This CI/CD has 3 distinct builds, `Pypi`(normal), `conda` and `extra`. `Extra` is REALLY slow (~4h), this is normal since it has to rebuild many things, but enables the wheel to be available for old Linuxes

# Node

- Edit `bindings/node/package.json` to reflect new version.
- Edit `CHANGELOG.md`:
    - Add relevant PRs that were added (python PRs do not belong for instance).
    - Add links at the end of the files.
- Go to [Releases](https://github.com/huggingface/tokenizers/releases)
- Create new Release:
    - Mark it as pre-release
    - Use new version name with a new tag (create on publish) `node-vX.X.X`.
    - Copy paste the new part of the `CHANGELOG.md`
- !! Click on `Publish release`. This will start the whole process of building a uploading
  the new version on `npm`, there's no going back after this
- Go to the [Actions](https://github.com/huggingface/tokenizers/actions) tab and check everything works smoothly.
- If anything fails, you need to fix the CI/CD to make it work again. Since your package was not uploaded to the repository properly, you can try again.


# Testing the CI/CD for release


If you want to make modifications to the CI/CD of the release GH actions, you need
to : 
- **Comment the part that uploads the artifacts** to `crates.io`, `PyPi` or `npm`.
- Change the trigger mecanism so it can trigger every time you push to your branch.
- Keep pushing your changes until the artifacts are properly created.
