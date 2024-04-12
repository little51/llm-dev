import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,\
    BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig


def load_Model():
    model_path = "./dataroot/models/alpindale/gemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map={"": 0})
    return model, tokenizer


def load_data(tokenizer):
    dataset_path = "./dataroot/datasets/Abirate/english_quotes"
    data = load_dataset(dataset_path)
    data = data.map(lambda samples: tokenizer(samples["quote"]),
                    batched=True)
    return data


def generate(text):
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]


def train(model, data):
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj",
                        "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()


if __name__ == "__main__":
    print("torch cuda:", torch.cuda.is_available())
    model, tokenizer = load_Model()
    data = load_data(tokenizer)
    print(generate("Quote: Imagination is more"))
    train(model, data)
    print(generate("Quote: Imagination is more"))
