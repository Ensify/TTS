import os
import argparse
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


parser = argparse.ArgumentParser(description='TTS Training')
parser.add_argument('--output-path', type=str, default="outputs", help='Path to save output files')
parser.add_argument('--meta-file-train', type=str, default="dataset/metadata.csv", help='Path to metadata file for training')
parser.add_argument('--dataset-path', type=str, default="dataset", help='Path to dataset directory')
args = parser.parse_args()

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train=args.meta_file_train, language="sa", path=args.dataset_path
)

audio_config = VitsAudioConfig(
    sample_rate=24000, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0, 
    mel_fmax=None
)

characters_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    phonemes = None,
    characters = "ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९॰ॱॲॳॴॵॶॷॸॹॺॻॼॽॾॿ꣠꣡꣢꣣꣤꣥꣦꣧꣨꣩꣪꣫꣬꣭꣮꣯꣰꣱ꣲꣳꣴꣵꣶꣷ꣸꣹꣺ꣻ꣼ꣽꣾꣿ",
    punctuations = "|।–!,-. ?।॥"
)


config = VitsConfig(
    audio=audio_config,
    characters=characters_config,
    run_name="vits_ljspeech",
    batch_size=12,
    eval_batch_size=4,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=5,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=args.output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.2,
)

model = Vits(config, ap, tokenizer, speaker_manager=None)
trainer = Trainer(
    TrainerArgs(),
    config,
    args.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()