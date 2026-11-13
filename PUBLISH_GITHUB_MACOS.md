# macOS reproducibility check

Use Python 3.11–3.13. For Python environments without a compatible
PyTorch wheel, create an isolated Conda environment:

```bash
conda create -n radio_intel python=3.12 -y
conda activate radio_intel
git clone https://github.com/pkd-45/radio-signal-intelligence-lab.git
cd radio-signal-intelligence-lab
chmod +x run_tests_macos.sh
PYTHON_BIN=python ./run_tests_macos.sh
```

Review the reference synthetic figures in
`results/reference/radio_signal_intelligence_summary.png` and compare
machine-readable outputs against the documented reference benchmark.

The repository is already public. These steps do not recreate a release,
reset Git history, or claim any observation of a real technosignature.
