# AudioJailbreak
<!-- Under construction. Stay tuned! -->

## Environment setup
```shell
git clone https://github.com/ictnlp/LLaMA-Omni
cd LLaMA-Omni

conda create -n llama-omni python=3.10
conda activate llama-omni
pip install pip==24.0
pip install -e .

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e . --no-build-isolation

pip install flash-attn --no-build-isolation
```

## Model & dataset preparation
1. Download the `Llama-3.1-8B-Omni` model from 🤗[Huggingface](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni). 

2. Download the `Whisper-large-v3` model.

```shell
import whisper
model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
```
3. Download the `HarmBench-Llama-2-13b-cls` from 🤗[Huggingface](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls). 

4. Download the dataset

- 4.1 Download the dataset [AdvBench-harmful_behaviors_custom](https://drive.google.com/file/d/1Wmo4NxoCnQnsJcAreW14FZQ5HXTMCgn6/view?usp=drive_link)

- 4.2 unzip
```shell
unzip AdvBench-harmful_behaviors_custom.zip
```

## Generate and test jailbreak audio
```shell
python generate.py
```

```shell
python get_rsp.py -type no_attack; python eval.py -type no_attack; python analyse.py -type no_attack
```

```shell
python get_rsp.py; python eval.py; python analyse.py
```