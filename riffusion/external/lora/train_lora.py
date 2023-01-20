from lora_diffusion.cli_lora_pti import train
from lora_diffusion.dataset import STYLE_TEMPLATE

MODEL_NAME = "riffusion/riffusion-model-v1"
INSTANCE_DIR = "/tmp/sample_clips_xzv8p57g/images"
OUTPUT_DIR = "./lora_output_acoustic"

if __name__ == "__main__":
    entries = [
        "music in the style of {}",
        "sound in the style of {}",
        "vibe in the style of {}",
        "audio in the style of {}",
        "groove in the style of {}",
    ]
    for i in range(len(STYLE_TEMPLATE)):
        STYLE_TEMPLATE[i] = entries[i % len(entries)]
    print(STYLE_TEMPLATE)

    train(
        pretrained_model_name_or_path=MODEL_NAME,
        instance_data_dir=INSTANCE_DIR,
        output_dir=OUTPUT_DIR,
        train_text_encoder=True,
        resolution=512,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        scale_lr=True,
        learning_rate_unet=1e-4,
        learning_rate_text=1e-5,
        learning_rate_ti=5e-4,
        color_jitter=False,
        lr_scheduler="linear",
        lr_warmup_steps=0,
        placeholder_tokens="<s1>|<s2>",
        use_template="style",
        save_steps=100,
        max_train_steps_ti=1000,
        max_train_steps_tuning=1000,
        perform_inversion=True,
        clip_ti_decay=True,
        weight_decay_ti=0.000,
        weight_decay_lora=0.001,
        continue_inversion=True,
        continue_inversion_lr=1e-4,
        device="cuda:0",
        lora_rank=1,
        use_face_segmentation_condition=False,
    )
