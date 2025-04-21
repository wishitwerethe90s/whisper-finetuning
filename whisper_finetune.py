# Import necessary libraries
from dataclasses import dataclass
import torch
import logging
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from huggingface_hub import notebook_login
from typing import Any, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Step 1: Load and prepare dataset with error handling
def load_and_prepare_dataset(dataset_name="mozilla-foundation/common_voice_11_0", config="hi", split="test[:10%]"):
    """Load and prepare dataset with error handling"""
    try:
        logger.info(f"Loading dataset {dataset_name} with config {config}")
        dataset = load_dataset(dataset_name, config, split=split)
        
        logger.info("Removing unnecessary columns")
        dataset = dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        
        logger.info("Casting audio column to 16kHz sampling rate")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

try:
    dataset = load_and_prepare_dataset()
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    logger.info(f"Dataset split complete. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
except Exception as e:
    logger.error(f"Dataset preparation failed: {str(e)}")
    raise

# Step 2: Prepare processor components
try:
    model_name = "openai/whisper-tiny"
    logger.info(f"Loading model components for {model_name}")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Hindi", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="Hindi", task="transcribe")
    logger.info("Processor components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model components: {str(e)}")
    raise

# Step 3: Preprocess dataset with logging
def prepare_dataset(batch):
    try:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    except Exception as e:
        logger.error(f"Error processing sample {batch.get('path', 'unknown')}: {str(e)}")
        raise

try:
    logger.info("Preprocessing training dataset")
    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    logger.info("Preprocessing test dataset")
    test_dataset = test_dataset.map(
        prepare_dataset,
        remove_columns=test_dataset.column_names,
        num_proc=4
    )
    logger.info("Preprocessing completed successfully")
except Exception as e:
    logger.error(f"Preprocessing failed: {str(e)}")
    raise

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # print(f"input_features type: {type(input_features)}, shape: {[len(f) for f in input_features]}")
        # print(f"label_features type: {type(label_features)}, shape: {[len(f['input_ids']) for f in label_features]}")
        # print(f"Padded batch input_features type: {type(batch['input_features'])}, shape: {batch['input_features'].shape}")
        # print(f"Padded batch labels_batch type: {type(labels_batch['input_ids'])}, shape: {labels_batch['input_ids'].shape}")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # print(f"Labels after masked_fill type: {type(labels)}, shape: {labels.shape}")

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Step 4: Create data collator
try:
    # from transformers.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )
    logger.info("Data collator initialized")
except Exception as e:
    logger.error(f"Error initializing data collator: {str(e)}")
    raise

# Step 5: Enhanced evaluation metrics with logging
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    try:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        # Log sample predictions
        logger.info(f"\nSample Prediction: {pred_str[0]}")
        logger.info(f"Sample Reference: {label_str[0]}")
        logger.info(f"Current WER: {wer:.2f}")
        
        return {"wer": wer}
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {"wer": float("nan")}

# Step 6: Model initialization with logging
try:
    logger.info(f"Initializing model {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    logger.info("Model initialized successfully")
    
    # Log model architecture
    logger.debug(f"Model architecture: {model}")
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise

# Step 7: Configure training arguments with enhanced logging
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_dir="./logs",  # For TensorBoard
)

# Step 8: Trainer initialization with exception handling
try:
    logger.info("Initializing Trainer")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    logger.info("Trainer initialized successfully")
except Exception as e:
    logger.error(f"Trainer initialization failed: {str(e)}")
    raise

# Step 9: Training process with enhanced logging
try:
    logger.info("Starting training process")
    logger.info(f"Training arguments:\n{training_args}")
    
    train_result = trainer.train()
    
    logger.info("Training completed successfully")
    logger.info(f"Training metrics: {train_result.metrics}")
except KeyboardInterrupt:
    logger.warning("Training interrupted by user")
except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    raise

# Step 10: Model saving with error handling
try:
    logger.info("Saving final model")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")
    raise

# Step 11: Final evaluation with logging
try:
    logger.info("Starting final evaluation")
    metrics = trainer.evaluate(test_dataset)
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
        
except Exception as e:
    logger.error(f"Evaluation failed: {str(e)}")
    raise

logger.info("Fine-tuning pipeline completed successfully")