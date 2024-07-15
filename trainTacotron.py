import os
import argparse

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.vits import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
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

audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)


config = Tacotron2Config(
    audio=audio_config,
    batch_size=12,
    eval_batch_size=4,
    num_loader_workers=2,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    ga_alpha=0.0,
    decoder_loss_alpha=0.25,
    postnet_loss_alpha=0.25,
    postnet_diff_spec_alpha=0,
    decoder_diff_spec_alpha=0,
    decoder_ssim_alpha=0,
    postnet_ssim_alpha=0,
    r=2,
    attention_type="dynamic_convolution",
    double_decoder_consistency=False,
    epochs=5,
    text_cleaner="phoneme_cleaners",
    use_phonemes=False,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=args.output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
model = Tacotron2(config, ap, tokenizer)

trainer = Trainer(
    TrainerArgs(), config, args.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()