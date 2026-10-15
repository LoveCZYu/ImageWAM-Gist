# Dependency Preparation

This document tracks the external source-code and model dependencies needed by ImageWAM.

## Summary

ImageWAM ships its own training, evaluation, dataset, ActionDiT, and model-adapter code in this repository. Some external projects are still required as runtime source dependencies or benchmark environments.

| Dependency | Release strategy |
| --- | --- |
| FLUX.2 | User clones upstream at the pinned commit. Not vendored. |
| OmniGen2 | User uses the ImageWAM OmniGen2 fork containing one extra patch commit. Not vendored. |
| DIM | User clones upstream at the pinned commit. Optional baseline. Not vendored. |
| RoboTwin | Vendored under `third_party/RoboTwin`; `assets/` contains only `_download.py`, not the actual asset payload. |
| Ovis-U1 HF code/config | Vendored under `third_party/ovis_u1_hf`, weights excluded. |
| LIBERO | Installed by `scripts/setup/_install_libero_env.sh`. Not vendored. |
| LIBERO-plus | Installed by `scripts/setup/_install_libero_plus_env.sh`. Not vendored. |

## FLUX.2

Use the upstream repository directly:

```bash
git clone https://github.com/black-forest-labs/flux2
cd flux2
git checkout 50fe5162777813d869182b139e83b10743caef15
```

Set:

```bash
export FLUX2_SRC=/path/to/flux2
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
```

Current audit result:

- Upstream: `https://github.com/black-forest-labs/flux2`
- Pinned commit: `50fe5162777813d869182b139e83b10743caef15`
- Local checkout matched upstream at audit time.
- No tracked, staged, or untracked local source modifications were found.

## OmniGen2

Use the ImageWAM OmniGen2 fork rather than the plain upstream checkout. The fork should contain exactly one ImageWAM-specific commit on top of upstream:

- Upstream base: `VectorSpaceLab/OmniGen2@18e6f9d5271b517fcb32e999f10df943ae9b8f20`
- ImageWAM patch commit: `3307397d74a90a6e22c595c6f5750f8dc71a8202`
- Patch summary: `Allow custom OmniGen2 attention head dim`
- Modified file in the external repo: `omnigen2/models/transformers/transformer_omnigen2.py`

Set:

```bash
export OMNIGEN2_SRC=/path/to/ImageWAM-OmniGen2-fork
export OMNIGEN2_MODEL_PATH=/path/to/OmniGen2/model
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

Current audit result:

- Local checkout was one commit ahead of `origin/main`.
- No tracked, staged, or untracked local source modifications were found.

## DIM

DIM is an optional baseline. Use the upstream repository directly:

```bash
git clone https://github.com/showlab/DIM.git
cd DIM
git checkout 28e644f5c3adf6f4d468792de4391de84ba19034
```

Set:

```bash
export DIM_SRC=/path/to/DIM
export DIM_MODEL_PATH=/path/to/DIM-4.6B-Edit
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

Current audit result:

- Upstream: `https://github.com/showlab/DIM.git`
- Pinned commit: `28e644f5c3adf6f4d468792de4391de84ba19034`
- Local checkout matched upstream at audit time.
- No tracked or staged local source modifications were found.
- Only ignored/generated Python cache files were observed as untracked noise.

## RoboTwin

RoboTwin is vendored because ImageWAM evaluation needs benchmark code and a project-specific policy adapter.

Release layout:

```text
third_party/RoboTwin/
├── README.vendor.md
├── LICENSE
├── envs/
├── script/
├── task_config/
├── description/
├── policy/
└── ...
```

Do not ship the RoboTwin asset payload in git. The release keeps only `third_party/RoboTwin/assets/_download.py`, which downloads the required RoboTwin assets into that directory.

Vendor metadata:

- Upstream: `https://github.com/RoboTwin-Platform/RoboTwin`
- Upstream commit recorded by the existing vendor note: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT

ImageWAM-specific policy code should use ImageWAM naming, e.g. `policy/imagewam_policy`.

## Ovis-U1 HF Code / Config

Ovis-U1 baseline support vendors only Hugging Face code/config/tokenizer files:

```text
third_party/ovis_u1_hf/
```

Weights are intentionally excluded and must be downloaded by the user.

Source metadata from `MANIFEST.json`:

- Source repo: `AIDC-AI/Ovis-U1-3B`
- Source base URL: `https://huggingface.co/AIDC-AI/Ovis-U1-3B/resolve/main`
- Included files: modeling/config/tokenizer files only
- Excluded files: safetensors weights

The vendored directory should include `README.vendor.md` before release.

## LIBERO

LIBERO is not vendored. Install it with:

```bash
bash scripts/setup/_install_libero_env.sh
```

The setup script clones upstream LIBERO and applies the small compatibility patch needed by the evaluation environment.

Relevant environment variables:

```bash
export LIBERO_DIR=/path/to/LIBERO
export LIBERO_CONFIG_DIR=$HOME/.libero
```

## LIBERO-plus

LIBERO-plus is not vendored. Install it with:

```bash
bash scripts/setup/_install_libero_plus_env.sh
```

The setup script clones upstream LIBERO-plus, downloads assets, and applies compatibility patches required by ImageWAM evaluation.

Relevant environment variables:

```bash
export LIBERO_PLUS_DIR=/path/to/LIBERO-plus
export LIBERO_CONFIG_DIR=$HOME/.libero
```

## Local Configuration

For local experiments, copy:

```bash
cp .env.example .env.local
```

Then fill in local paths. `.env.local` is ignored by git.

