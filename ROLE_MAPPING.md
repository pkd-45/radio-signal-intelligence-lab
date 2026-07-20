# Mapping to Oxford Intelligence Engineer (vacancy 187345)

This repository is a small, inspectable demonstration aligned to the advertised work. It does not claim to be Breakthrough Listen production software.

| Advertised need | Inspectable evidence in this repository |
|---|---|
| Transform raw research data into actionable datasets | Deterministic ON/OFF dynamic-spectrum generator, balanced manifest, leakage-safe group splits, optional `blimpy` adapter, candidate ranking CSV |
| Small purpose-built open-source models | Compact multimodal CNN with fewer than 30k trainable parameters and TorchScript export |
| Large, complex and multi-modal data | Two waterfall channels plus transparent metadata side branch; design extends to tiled `.fil`/`.h5` data |
| Architecture, design and deployment advice | Model card, explicit data contract, CLI, CI, scripted model export, limitations |
| GPU-based systems and optimisation | PyTorch implementation runs on CUDA when available and CPU otherwise; deliberately small for reproducibility |
| Robustness and performance | Group-safe splits, balanced classes, macro-F1, balanced accuracy, Brier score, confusion matrix |
| Signal-processing algorithms | Synthetic drifting lines, persistent narrowband RFI, bursts and combs; ON/OFF cadence logic |
| Technical documentation and research outputs | README, model card, validation report, machine-readable metrics and ranking |
| Multidisciplinary collaboration | Interfaces are deliberately separable: astronomy I/O, signal simulation, model, transparent triage |

## How it complements existing public repositories

- `ml-radio-recon`: deeper PyTorch radio-interferometric image reconstruction and anomaly detection.
- `radio-pipeline-control-lab`: reliable execution, restart/resume, integrity checks, telemetry, Slurm planning, Kafka/etcd adapter boundaries.
- `dsa-hi-survey-ops`: spectral-line planning, RFI-aware selection, FITS-cube injection/recovery, science-product QA.
- `Horn-antenna-21-cm`: early hands-on radio hardware, 21-cm observing and analysis experience.

The combined portfolio demonstrates a progression from radio instrumentation and astronomy, through science pipelines, to compact AI and operationally reliable research software.
