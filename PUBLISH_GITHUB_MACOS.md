# Publish from macOS

Use Python 3.11--3.13. On a Mac with a Python 3.14 base environment, create a separate Conda environment:

```bash
conda create -n radio_intel python=3.12 -y
conda activate radio_intel
```

Validate:

```bash
chmod +x run_tests_macos.sh
PYTHON_BIN=python ./run_tests_macos.sh
open results/reference/radio_signal_intelligence_summary.png
```

Publish only after the tests and demo succeed:

```bash
git init
git branch -M main
git add .
git diff --cached --check
git commit -m "Initial public release of radio-signal-intelligence-lab"

gh repo create pkd-45/radio-signal-intelligence-lab \
  --public \
  --description "Compact multimodal AI benchmark for radio technosignature candidate triage." \
  --source=. \
  --remote=origin \
  --push
```

After CI passes:

```bash
git tag -a v0.1.0 -m "Application methods release"
git push origin v0.1.0

gh release create v0.1.0 \
  --title "radio-signal-intelligence-lab v0.1.0" \
  --notes "Controlled synthetic benchmark for balanced ON/OFF dynamic-spectrum data, a compact multimodal PyTorch model, TorchScript export, and transparent candidate ranking. No real-data technosignature claim."
```
