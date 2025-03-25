import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ClassLabel, Features, Value, load_dataset
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# Define Pydantic data models
class LoraConfig(BaseModel):
    r: int = Field(16, description="LoRA rank")
    alpha: int = Field(32, description="LoRA alpha (scaling factor)")
    dropout: float = Field(0.1, description="LoRA dropout")
    target_modules: List[str] = Field(
        ["q_proj", "v_proj"], description="Modules to apply LoRA to"
    )
    bias: str = Field("none", description="Whether to train bias parameters")


class TrainingConfig(BaseModel):
    model_name: str = Field(
        "NovaSearch/stella_en_1.5B_v5", description="Base model to fine-tune"
    )
    num_labels: int = Field(3, description="Number of classification labels")
    learning_rate: float = Field(1e-4, description="Learning rate")
    batch_size: int = Field(8, description="Batch size")
    num_epochs: int = Field(20, description="Number of training epochs")
    warmup_ratio: float = Field(0.1, description="Warmup ratio")
    weight_decay: float = Field(0.01, description="Weight decay")
    max_length: int = Field(512, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(
        4, description="Gradient accumulation steps"
    )


# Model class with LoRA
class StellaForSequenceClassificationWithLoRA(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        instruction: str = "Classify this biotech press release based on what it means for the future of the company as either negative, neutral, or positive.",
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.instruction = instruction
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        print(self.model)
        print(type(self.model))

        # For stella model, we need to use the appropriate prompt template
        # Using the s2p prompt format as it's closer to our classification task
        self.query_prompt = f"Instruct: {instruction}\nQuery: "

        # Get the embedding dimension
        self.vector_dim = 1024
        self.vector_linear = nn.Linear(
            in_features=self.model.config.hidden_size, out_features=self.vector_dim
        )

        # Load the dimension projection weights if available
        try:
            vector_linear_dict = {
                k.replace("linear.", ""): v
                for k, v in torch.load(
                    f"NovaSearch/2_Dense_{self.vector_dim}/pytorch_model.bin"
                ).items()
            }
            self.vector_linear.load_state_dict(vector_linear_dict)
        except Exception as e:
            print(f"Could not load vector linear weights: {e}")
            print("Will use random initialization for vector projection.")

        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Classification head
        self.classifier = nn.Linear(self.vector_dim, num_labels)

        # Apply LoRA
        if lora_config:
            # Create PEFT LoRA config with base_model_name_or_path
            peft_config = PeftLoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                bias=lora_config.get("bias", "none"),
                base_model_name_or_path=model_name,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def forward(self, text=None, labels=None, max_length=512, **kwargs):
        # Handle the case where text is provided directly
        if text is not None:
            # Format the input with appropriate prompt
            formatted_text = [self.query_prompt + t for t in text]

            # Tokenize the input
            with torch.no_grad():
                input_data = self.tokenizer(
                    formatted_text,
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_data = {k: v.to(device) for k, v in input_data.items()}

                # Get the embeddings using the model
                attention_mask = input_data["attention_mask"]
                last_hidden_state = self.model(**input_data)[0]

                # Mask padding tokens
                last_hidden = last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )

                # Mean pooling
                embeddings = (
                    last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                )

                # Project to vector dimension
                embeddings = self.vector_linear(embeddings)

                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
        else:
            raise ValueError("This model requires 'text' input for classification")

        # Classification
        logits = self.classifier(embeddings)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


# Data processor
class DataProcessor:
    def __init__(self, data_file_path: str, model_name: str, max_length: int = 512):
        self.data_file_path = data_file_path
        self.model_name = model_name
        self.max_length = max_length

    def load_and_process(self, test_size: float = 0.1):
        # Check if file exists
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")

        features = Features(
            {
                "text": Value("string"),
                "labels": ClassLabel(names=["0.0", "1.0", "2.0"]),
            }
        )

        dataset = load_dataset(
            "csv",
            data_files=self.data_file_path,
            features=features,
        )

        dataset["train"] = dataset["train"].add_column(
            "id", range(len(dataset["train"]))
        )

        labels = np.array(dataset["train"]["labels"])
        indices = np.arange(len(labels))

        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels,
            random_state=42,
        )

        train_dataset = dataset["train"].select(train_indices)
        test_dataset = dataset["train"].select(test_indices)

        class_weights = self._compute_class_weights(dataset["train"]["labels"])

        return train_dataset, test_dataset, class_weights

    def _compute_class_weights(self, labels):
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels,
        )
        return torch.tensor(class_weights, dtype=torch.float)

    def create_collator(self):
        def collate_fn(examples):
            texts = [example["text"] for example in examples]
            labels = torch.tensor([example["labels"] for example in examples])

            return {
                "text": texts,
                "labels": labels,
            }

        return collate_fn


# Custom trainer with weighted loss
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.class_weights is not None and labels is not None:
            # Use weighted cross entropy loss
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device))
            loss = loss_fct(outputs["logits"], labels)
            outputs["loss"] = loss
        else:
            loss = outputs["loss"] if "loss" in outputs else None

        return (loss, outputs) if return_outputs else loss


# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", labels=[0, 1, 2]
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Main training function
def train_lora_model(data_file_path, training_config, lora_config):
    # Load and process data
    data_processor = DataProcessor(
        data_file_path=data_file_path,
        model_name=training_config.model_name,
        max_length=training_config.max_length,
    )

    train_dataset, test_dataset, class_weights = data_processor.load_and_process()
    data_collator = data_processor.create_collator()

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Class weights: {class_weights}")

    # Create model with LoRA
    model = StellaForSequenceClassificationWithLoRA(
        model_name=training_config.model_name,
        num_labels=training_config.num_labels,
        instruction="Classify this biotech press release based on what it means for the future of the company.",
        lora_config={
            "r": lora_config.r,
            "alpha": lora_config.alpha,
            "dropout": lora_config.dropout,
            "target_modules": lora_config.target_modules,
            "bias": lora_config.bias,
        },
    ).to(device)

    # Test model with a small batch to catch errors early
    try:
        dummy_batch = {
            "text": [
                "This is a test biotech press release.",
                "Another test press release about drug trials.",
            ],
            "labels": torch.tensor([0, 1]).to(device),
        }
        model(**dummy_batch)
        print("Model forward pass works correctly!")
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        raise

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        num_train_epochs=training_config.num_epochs,
        warmup_ratio=training_config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=(device.type == "cuda"),  # Only use fp16 with CUDA, not with MPS
        report_to="none",
    )

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Train model
    print("Let's start the training")
    trainer.train()

    print("Training complete")
    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save LoRA adapter only
    output_dir = "./lora_adapter"
    os.makedirs(output_dir, exist_ok=True)
    model.model.save_pretrained(output_dir)

    return model, eval_results


# Inference function
def predict_sentiment(model, text, max_length=512):
    model.eval()
    with torch.no_grad():
        outputs = model(text=[text], max_length=max_length)
        logits = outputs["logits"]
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        # Map class to meaning
        class_meanings = {
            0: "Negative outlook for company",
            1: "Neutral impact on company future",
            2: "Positive outlook for company",
        }

        meaning = class_meanings[predicted_class]

    return {
        "class": predicted_class,
        "meaning": meaning,
        "probabilities": probabilities.detach().cpu().numpy()[0],
    }


if __name__ == "__main__":
    # Data path
    data_file_path = "data/processed/finetuning_3_labels_topic_pruned.csv"

    # Set up configurations
    training_config = TrainingConfig(
        num_labels=3,
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=10,
        gradient_accumulation_steps=1,
        weight_decay=0.005,
    )

    lora_config = LoraConfig(
        r=16,
        alpha=32,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # Train model
    model, eval_results = train_lora_model(
        data_file_path=data_file_path,
        training_config=training_config,
        lora_config=lora_config,
    )

    # Example prediction
    test_text = "BioVie Announces Alignment with FDA on Clinical Trial to Assess Bezisterim in Parkinson's Disease"
    result = predict_sentiment(model, test_text)
    print(f"Predicted class: {result['class']} - {result['meaning']}")
    print(f"Confidence scores: {result['probabilities']}")
