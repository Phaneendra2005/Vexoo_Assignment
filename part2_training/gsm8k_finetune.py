"""
gsm8k_finetune.py
=================
Vexoo Labs AI Engineer Assignment — Part 2
LoRA-based SFT on GSM8K math reasoning dataset.

Set FULL_TRAIN = True to enable real GPU training with LLaMA-3.2-1B + LoRA.
Default FULL_TRAIN = False runs entirely in simulation mode (no GPU needed).

Author: Candidate
"""

import re
import random
from typing import List, Dict, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# TRAINING MODE FLAG
# ---------------------------------------------------------------------------
FULL_TRAIN: bool = False   # Toggle to True for real GPU training

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------
MODEL_NAME       = "meta-llama/Llama-3.2-1B"
MAX_LENGTH       = 512
TRAIN_SAMPLES    = 3000
EVAL_SAMPLES     = 1000
NUM_EPOCHS       = 3
BATCH_SIZE       = 8
LOG_EVERY_STEPS  = 100
LORA_R           = 8
LORA_ALPHA       = 16
LORA_TARGET_MODS = ["q_proj", "v_proj"]
SEED             = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# SECTION 1: CONDITIONAL IMPORTS
# ---------------------------------------------------------------------------

if FULL_TRAIN:
    # Real training — requires GPU + pip install transformers peft datasets torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
else:
    try:
        from datasets import load_dataset
        DATASETS_AVAILABLE = True
    except ImportError:
        DATASETS_AVAILABLE = False
        print("[WARN] 'datasets' not installed — synthetic data will be used.")

    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# SECTION 2: MOCK CLASSES (simulation mode)
# ---------------------------------------------------------------------------

class MockTokenizer:
    """
    Simulates the HuggingFace AutoTokenizer interface.
    Maps words to integer IDs via deterministic hash — no real vocabulary needed.
    """
    def __init__(self, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.model_max_length = max_length

    def _word_to_id(self, word: str) -> int:
        """Deterministically map a word to an ID in [2, 30000]."""
        return (hash(word) % 29998) + 2

    def __call__(self, text: str, max_length: int = MAX_LENGTH,
                 truncation: bool = True, padding: str = "max_length",
                 return_tensors: Optional[str] = None) -> Dict[str, List[int]]:
        """Tokenize a string and return input_ids + attention_mask."""
        ids = [self._word_to_id(w) for w in text.split()]
        ids = ids[:max_length - 1] + [self.eos_token_id]   # truncate + EOS
        pad_len = max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.pad_token_id] * pad_len
        return {"input_ids": ids, "attention_mask": attention_mask}


class MockModel:
    """
    Simulates the HuggingFace AutoModelForCausalLM interface.
    No real computation — exists to exercise training loop code paths.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"[MockModel] Simulated model loaded: {model_name}")

    def __call__(self, input_ids, attention_mask=None, labels=None):
        """Return a fake loss value."""
        return {"loss": random.uniform(0.3, 1.5)}

    def parameters(self):
        yield type("Param", (), {"data": [0.0], "grad": None})()

    def train(self): pass
    def eval(self):  pass

    def generate(self, input_ids, max_new_tokens=50, **kwargs):
        """Simulate generation by returning a random number token."""
        return [[random.randint(1, 200)]]


# ---------------------------------------------------------------------------
# SECTION 3: DATA LOADING AND FORMATTING
# ---------------------------------------------------------------------------

def format_sample(sample: Dict[str, str]) -> str:
    """
    Format a GSM8K sample as:
        "Question: <question>\\nAnswer: <answer>"
    """
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"


def extract_final_answer(answer_str: str) -> Optional[str]:
    """
    Extract the numeric final answer from a GSM8K answer string.
    GSM8K answers end with '#### <number>'.
    """
    match = re.search(r"####\s*([\d,\-\.]+)", answer_str)
    return match.group(1).replace(",", "") if match else None


def load_gsm8k_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Load GSM8K via HuggingFace datasets (if available), else use synthetic data.
    Returns (train_samples[:3000], eval_samples[:1000]).
    """
    if DATASETS_AVAILABLE:
        print("[Data] Loading openai/gsm8k (main config)...")
        dataset   = load_dataset("openai/gsm8k", "main")
        all_train = list(dataset["train"])
        train_data = all_train[:TRAIN_SAMPLES]
        eval_data  = all_train[TRAIN_SAMPLES: TRAIN_SAMPLES + EVAL_SAMPLES]
        print(f"[Data] Train: {len(train_data)} | Eval: {len(eval_data)}")
    else:
        print("[Data] Generating synthetic GSM8K-style samples...")
        train_data = [
            {"question": f"If a store has {i+5} apples and sells {i}, how many remain?",
             "answer": f"Start: {i+5}. Sold: {i}. Remaining: 5.\n#### 5"}
            for i in range(TRAIN_SAMPLES)
        ]
        eval_data = [
            {"question": f"A bag has {i+3} marbles. {i} are removed. How many left?",
             "answer": f"Start: {i+3}. Removed: {i}. Left: 3.\n#### 3"}
            for i in range(EVAL_SAMPLES)
        ]
    return train_data, eval_data


# ---------------------------------------------------------------------------
# SECTION 4: TOKENIZATION HELPER
# ---------------------------------------------------------------------------

def tokenize_batch(samples: List[Dict], tokenizer: Any,
                   max_length: int = MAX_LENGTH) -> List[Dict]:
    """
    Format and tokenize a list of GSM8K samples.
    Returns list of {input_ids, attention_mask} dicts.
    """
    return [
        tokenizer(format_sample(s), max_length=max_length,
                  truncation=True, padding="max_length")
        for s in samples
    ]


# ---------------------------------------------------------------------------
# SECTION 5: SIMULATED TRAINING LOOP
# ---------------------------------------------------------------------------

def simulate_training(train_data: List[Dict], tokenizer: Any,
                      model: Any) -> List[Tuple[int, float]]:
    """
    Simulated training loop for FULL_TRAIN=False mode.

    Iterates NUM_EPOCHS epochs, logging a mock decreasing loss every
    LOG_EVERY_STEPS steps. Uses tqdm progress bar when available.

    Returns list of (step, loss) tuples that were logged.
    """
    total_steps = 0
    loss_log    = []
    loss_start  = 1.5
    loss_end    = 0.30
    total_training_steps = NUM_EPOCHS * (len(train_data) // BATCH_SIZE)

    print(f"\n[Train] Simulated training | {NUM_EPOCHS} epochs | "
          f"{total_training_steps} total steps")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n  === Epoch {epoch}/{NUM_EPOCHS} ===")
        steps_in_epoch = len(train_data) // BATCH_SIZE
        step_iter = range(steps_in_epoch)

        # Wrap with tqdm if available
        if TQDM_AVAILABLE:
            step_iter = tqdm(step_iter, desc=f"  Epoch {epoch}", unit="step")

        for step in step_iter:
            # Pick mini-batch (cycling through train data)
            batch_start = (step * BATCH_SIZE) % len(train_data)
            batch = train_data[batch_start: batch_start + BATCH_SIZE]

            # Exercise the tokenizer code path
            _ = tokenize_batch(batch, tokenizer)

            # Mock linearly decaying loss with Gaussian noise
            progress  = total_steps / max(total_training_steps - 1, 1)
            mock_loss = loss_start + (loss_end - loss_start) * progress
            mock_loss = max(mock_loss + random.gauss(0, 0.03), loss_end)
            total_steps += 1

            if total_steps % LOG_EVERY_STEPS == 0:
                loss_log.append((total_steps, round(mock_loss, 4)))
                if not TQDM_AVAILABLE:
                    print(f"  Step {total_steps:>5} | Loss: {mock_loss:.4f}")

        print(f"  Epoch {epoch} done. Approx loss: {mock_loss:.4f}")

    return loss_log


# ---------------------------------------------------------------------------
# SECTION 6: EVALUATION
# ---------------------------------------------------------------------------

def evaluate(eval_data: List[Dict], tokenizer: Any, model: Any,
             n_show: int = 3) -> float:
    """
    Evaluate on eval_data using exact-match accuracy on the numeric final answer.
    Prints n_show sample predictions vs ground truth.
    Returns accuracy in [0, 1].
    """
    print("\n[Eval] Evaluating on eval set...")
    correct, shown = 0, 0

    for idx, sample in enumerate(eval_data):
        gt_answer = extract_final_answer(sample["answer"])

        if FULL_TRAIN:
            import torch
            text    = f"Question: {sample['question']}\nAnswer:"
            encoded = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                out = model.generate(encoded["input_ids"], max_new_tokens=50,
                                     pad_token_id=tokenizer.pad_token_id)
            decoded   = tokenizer.decode(out[0], skip_special_tokens=True)
            m         = re.search(r"####\s*([\d,\-\.]+)", decoded)
            pred_ans  = m.group(1).replace(",", "") if m else "N/A"
        else:
            # Simulate ~30% accuracy (realistic for an untrained model)
            if gt_answer and gt_answer.lstrip("-").isdigit():
                gt_int   = int(gt_answer)
                pred_ans = str(gt_int) if random.random() < 0.30 else str(gt_int + random.randint(-5, 10))
            else:
                pred_ans = str(random.randint(1, 100))

        if pred_ans == gt_answer:
            correct += 1

        if shown < n_show:
            print(f"\n  Sample #{idx+1}")
            print(f"  Question : {sample['question'][:80]}...")
            print(f"  GT Answer: {gt_answer}")
            print(f"  Predicted: {pred_ans}")
            print(f"  Match    : {'YES' if pred_ans == gt_answer else 'NO'}")
            shown += 1

    accuracy = correct / len(eval_data)
    print(f"\n[Eval] Exact-Match: {correct}/{len(eval_data)} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    return accuracy


# ---------------------------------------------------------------------------
# SECTION 7: REAL TRAINING (FULL_TRAIN=True path)
# ---------------------------------------------------------------------------

def run_real_training(train_data: List[Dict], eval_data: List[Dict]) -> None:
    """
    Real LoRA fine-tuning using HuggingFace Trainer.
    Requires GPU + transformers + peft + torch.
    """
    import torch
    from datasets import Dataset

    print(f"\n[FULL_TRAIN] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[FULL_TRAIN] Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    # Apply LoRA via peft
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODS, lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(examples):
        texts = [format_sample({"question": q, "answer": a})
                 for q, a in zip(examples["question"], examples["answer"])]
        out = tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    hf_train = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
    hf_eval  = Dataset.from_list(eval_data).map(tokenize_fn, batched=True)

    args = TrainingArguments(
        output_dir="./gsm8k_lora_output", num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=LOG_EVERY_STEPS, evaluation_strategy="epoch",
        save_strategy="epoch", fp16=True, seed=SEED, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=hf_train, eval_dataset=hf_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print("[FULL_TRAIN] Starting training...")
    trainer.train()
    print("[FULL_TRAIN] Training complete.")
    evaluate(eval_data, tokenizer, model)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   VEXOO LABS — Part 2: GSM8K LoRA Fine-Tuning")
    print(f"   Mode: {'FULL TRAINING (GPU)' if FULL_TRAIN else 'SIMULATION (no GPU)'}")
    print("="*60)

    train_data, eval_data = load_gsm8k_data()

    if FULL_TRAIN:
        run_real_training(train_data, eval_data)
    else:
        print("\n[Sim] Initialising MockTokenizer and MockModel...")
        tokenizer = MockTokenizer()
        model     = MockModel(MODEL_NAME)

        # Show tokenization example
        sample_text = format_sample(train_data[0])
        encoded = tokenizer(sample_text)
        print(f"\n[Sim] Formatted sample (first 120 chars): {sample_text[:120]}...")
        print(f"[Sim] input_ids[:10]     : {encoded['input_ids'][:10]}")
        print(f"[Sim] attention_mask[:10]: {encoded['attention_mask'][:10]}")

        # Simulated training
        loss_log = simulate_training(train_data, tokenizer, model)

        # Print loss curve
        print("\n[Sim] Loss Log (step, loss):")
        for step, loss in loss_log:
            bar = "#" * int(loss * 20)
            print(f"  Step {step:>5}: {loss:.4f}  {bar}")

        # Evaluate
        evaluate(eval_data, tokenizer, model)

    print("\n" + "="*60 + "\n   Part 2 complete.\n" + "="*60 + "\n")
