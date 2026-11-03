"""Four-bit NF4 QLoRA supervised fine-tuning for fuel-graph extraction."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local pretrained model path or model identifier')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing train.jsonl and val.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import os
    import sys
    import site
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    MODEL_PATH = args.model
    DATA_DIR = str(args.data_dir)
    OUTPUT_DIR = str(args.output_dir)
    TRAIN_FILE = f'{DATA_DIR}/train.jsonl'
    VAL_FILE = f'{DATA_DIR}/val.jsonl'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Please run this script on a GPU node.')
    major, _ = torch.cuda.get_device_capability()
    use_bf16 = major >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'use_bf16: {use_bf16}')
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory // 1024 ** 3
        if total_gib >= 75:
            max_memory[i] = '72GiB'
        elif total_gib >= 39:
            max_memory[i] = '34GiB'
        else:
            max_memory[i] = f'{max(10, total_gib - 4)}GiB'
    print('max_memory:', max_memory)
    dataset = load_dataset('json', data_files={'train': TRAIN_FILE, 'validation': VAL_FILE})

    def convert_to_prompt_completion(example):
        messages = example['messages']
        return {'prompt': messages[:-1], 'completion': [messages[-1]]}
    dataset = dataset.map(convert_to_prompt_completion, remove_columns=dataset['train'].column_names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map='auto', max_memory=max_memory, trust_remote_code=True)
    model.config.use_cache = False
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    training_args = SFTConfig(output_dir=OUTPUT_DIR, max_length=20480, packing=False, num_train_epochs=1, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=0.0002, lr_scheduler_type='cosine', warmup_ratio=0.03, logging_steps=5, eval_strategy='no', save_steps=50, save_total_limit=2, bf16=use_bf16, fp16=not use_bf16, gradient_checkpointing=True, optim='paged_adamw_8bit', report_to='none')
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset['train'], peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f'Saved LoRA adapter to: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()
