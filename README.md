## Installation

```sh
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .[all,dev,notebooks] 
```

## Training

```sh
python trainVits.py --output-path "results" \
--meta-file-train "dataset/ljspeech/metadata.csv" \
--dataset-path "dataset/ljspeech"
```

## Inference

```bash
tts --model_path "model.pth" \
--config_path "config.json" \
--text पीनवक्षाविशालाक्षोलक्ष्मीवान्शुभलक्षणः \
--out_path test_out.wav
```