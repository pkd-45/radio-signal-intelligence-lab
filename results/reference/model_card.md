# Tiny Multimodal CNN model card

- Parameters: 6,705
- Best validation macro-F1: 1.000
- Inputs: two 64x64 ON/OFF waterfall channels plus six transparent metadata features.
- Output classes: noise, candidate drift, persistent narrowband RFI, broadband burst, spectral-comb RFI.
- Scope: controlled synthetic benchmark only. Not a production technosignature detector.
- Intended use: demonstrate dataset construction, compact multimodal modelling, evaluation, export, and candidate triage.
