# Import necessary libraries
from dataclasses import dataclass
import torch
import logging
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Optional
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import soundfile as sf
from datasets import Dataset, Audio, DatasetDict, concatenate_datasets
import re
import gc
from tqdm.auto import tqdm

# Try to import optional dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available. Audio resampling will be limited.")

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

# Step 1: Load and prepare MUCS dataset
def load_mucs_dataset(data_dir, split="train", test_size=0.01):
    """
    Load the MUCS dataset from Kaldi-style files
    
    Args:
        data_dir: Root directory of MUCS dataset
        split: Dataset split to use ('train' by default)
        test_size: Proportion of data to use for testing (if splitting train data)
        
    Returns:
        DatasetDict containing train and test splits
    """
    try:
        logger.info(f"Loading MUCS dataset from {data_dir}")
        split_dir = os.path.join(data_dir, split)
        transcript_dir = os.path.join(split_dir, "transcripts")
        
        # Load text file (utterance -> transcription)
        text_file = os.path.join(transcript_dir, "text")
        with open(text_file, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
        
        utterance_texts = {}
        for line in text_lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utterance_id, text = parts
                utterance_texts[utterance_id] = text
        
        # Load wav.scp file (recording_id -> audio file path)
        wav_scp_file = os.path.join(transcript_dir, "wav.scp")
        with open(wav_scp_file, 'r', encoding='utf-8') as f:
            wav_lines = f.readlines()
        
        recording_wavs = {}
        for line in wav_lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                recording_id, wav_path = parts
                # For MUCS dataset, wav_path is just the filename
                # Save it as is, the load_audio function will handle path resolution
                recording_wavs[recording_id] = wav_path
        
        # Check if segments file exists for segmented audio
        segments_file = os.path.join(transcript_dir, "segments")
        utterance_segments = {}
        if os.path.exists(segments_file):
            with open(segments_file, 'r', encoding='utf-8') as f:
                segment_lines = f.readlines()
            
            for line in segment_lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    utt_id, recording_id, start_time, end_time = parts[:4]
                    utterance_segments[utt_id] = {
                        'recording_id': recording_id,
                        'start_time': float(start_time),
                        'end_time': float(end_time)
                    }
        
        # Create dataset
        data = []
        for utt_id in utterance_texts.keys():
            # For MUCS dataset, we need to use the segments file
            if utt_id in utterance_segments:
                seg_info = utterance_segments[utt_id]
                recording_id = seg_info['recording_id']
                
                if recording_id in recording_wavs:
                    entry = {
                        'id': utt_id,
                        'sentence': utterance_texts[utt_id],
                        'path': recording_wavs[recording_id],
                        'recording_id': recording_id,
                        'start_time': seg_info['start_time'],
                        'end_time': seg_info['end_time']
                    }
                    data.append(entry)
                else:
                    logger.warning(f"Recording ID {recording_id} for utterance {utt_id} not found in wav.scp")
            else:
                # Handle case where utterance might directly correspond to a full audio file
                # This is less likely in the MUCS dataset based on the samples provided
                pass
        
        logger.info(f"Loaded {len(data)} utterances")
        
        # Create Dataset object
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        
        # Split into train and test
        if split == "train":
            dataset_dict = dataset.train_test_split(test_size=test_size)
            logger.info(f"Split into {len(dataset_dict['train'])} train and {len(dataset_dict['test'])} test samples")
        else:
            dataset_dict = DatasetDict({"train": dataset})
        
        return dataset_dict
    
    except Exception as e:
        logger.error(f"Error loading MUCS dataset: {str(e)}")
        raise

# Function to load audio data
def load_audio(path, start_time=None, end_time=None, target_sr=16000):
    """Load audio file with optional segment information"""
    try:
        # For MUCS dataset, make sure to handle the path correctly
        # Check if path is absolute, otherwise make it relative to the data directory
        if not os.path.isabs(path):
            path = os.path.join(MUCS_DATA_DIR, "train", path)
        
        if not os.path.exists(path):
            # Try adding .wav extension if not present
            if not path.endswith('.wav'):
                path = path + '.wav'
            
            # If still not found, try looking in a different location
            if not os.path.exists(path):
                alt_path = os.path.join(MUCS_DATA_DIR, path)
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    logger.warning(f"Audio file not found at {path}")
                    # Return a silent audio segment as fallback
                    return {
                        "array": np.zeros(1600),  # 0.1s of silence
                        "sampling_rate": target_sr
                    }
        
        logger.debug(f"Loading audio from {path}")
        
        try:
            audio_array, sr = sf.read(path)
        except Exception as e:
            logger.warning(f"Failed to read audio file {path}: {str(e)}")
            return {
                "array": np.zeros(1600),
                "sampling_rate": target_sr
            }
        
        # Handle segmentation
        if start_time is not None and end_time is not None:
            try:
                # Load specific segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Ensure valid indices
                if start_sample >= len(audio_array):
                    start_sample = 0
                if end_sample > len(audio_array):
                    end_sample = len(audio_array)
                    
                audio_array = audio_array[start_sample:end_sample]
            except Exception as e:
                logger.warning(f"Error in segmenting audio {path}: {str(e)}")
                # If segmentation fails, use the entire file
        
        # Convert to mono if stereo
        try:
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
        except Exception as e:
            logger.warning(f"Error converting stereo to mono: {str(e)}")
            # Return a fallback array
            return {
                "array": np.zeros(1600),
                "sampling_rate": target_sr
            }
        
        # Resample if needed
        if sr != target_sr:
            try:
                if LIBROSA_AVAILABLE:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                else:
                    logger.warning(f"Librosa not available for resampling. Audio sampling rate {sr} differs from target {target_sr}")
            except Exception as e:
                logger.warning(f"Error resampling audio: {str(e)}")
        
        # Final check to ensure valid array
        if len(audio_array) == 0:
            logger.warning(f"Empty audio array for {path}")
            audio_array = np.zeros(1600)  # Use silent fallback
        
        return {
            "array": audio_array,
            "sampling_rate": sr
        }
    except Exception as e:
        logger.error(f"Error loading audio file {path}: {str(e)}")
        return {
            "array": np.zeros(1600),  # 0.1s of silence as a fallback
            "sampling_rate": target_sr
        }

def prepare_dataset_mucs(batch):
    """Prepare MUCS dataset samples for the model with special handling for Hindi-English code-switching"""
    try:
        # Load audio
        audio = load_audio(
            batch["path"], 
            batch.get("start_time", None), 
            batch.get("end_time", None)
        )
        
        # Skip samples that are too short or potentially corrupted
        if len(audio["array"]) < 1000:  # Very short audio, likely an issue
            logger.warning(f"Sample {batch.get('id', 'unknown')} has very short audio ({len(audio['array'])} samples), skipping")
            # Return a dummy batch that will be filtered out
            return {"skip_sample": True}
        
        # Check for NaN or Inf values in the audio array
        if np.isnan(audio["array"]).any() or np.isinf(audio["array"]).any():
            logger.warning(f"Sample {batch.get('id', 'unknown')} contains NaN or Inf values, skipping")
            return {"skip_sample": True}
        
        try:
            # Extract features
            features = feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            )
            
            batch["input_features"] = features.input_features[0]
        except Exception as e:
            logger.warning(f"Error extracting features for sample {batch.get('id', 'unknown')}: {str(e)}")
            return {"skip_sample": True}
        
        try:
            # Generate labels - special handling for Hindi-English code-switching
            if "sentence" not in batch or not batch["sentence"]:
                logger.warning(f"Missing transcription for sample {batch.get('id', 'unknown')}")
                return {"skip_sample": True}
                
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
        except Exception as e:
            logger.warning(f"Error tokenizing transcription for sample {batch.get('id', 'unknown')}: {str(e)}")
            return {"skip_sample": True}
        
        # Remove skip_sample marker if present
        if "skip_sample" in batch:
            del batch["skip_sample"]
            
        return batch
    except Exception as e:
        logger.error(f"Error processing sample {batch.get('id', 'unknown')}: {str(e)}")
        # Instead of raising, return a marker to filter this sample out
        return {"skip_sample": True}

def log_memory_usage():
    """Log current memory usage"""
    import psutil
    # Get the process memory info
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Convert to MB for readability
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    
    # Log the memory usage
    logger.info(f"Memory Usage - RSS: {rss_mb:.2f}MB, VMS: {vms_mb:.2f}MB")
    
    # GPU memory if torch and CUDA are available
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(f"CUDA Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")
        except:
            pass
    
    return rss_mb

def free_memory():
    """Explicitly free memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_dataset_in_batches(dataset, batch_size=1000):
    """Process the dataset in smaller batches to avoid memory issues"""
    logger.info(f"Processing dataset with {len(dataset)} samples in batches of {batch_size}")
    
    all_processed_datasets = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        logger.info(f"Processing batch {i+1}/{num_batches} (samples {start_idx}-{end_idx})")
        
        # Get the current batch
        batch_dataset = dataset.select(range(start_idx, end_idx))
        
        # Process the batch with a single process to avoid subprocess issues
        processed_batch = batch_dataset.map(
            prepare_dataset_mucs,
            remove_columns=batch_dataset.column_names,
            num_proc=1,  # Use single process for reliability
            desc=f"Batch {i+1}/{num_batches}",
        )
        
        # Filter out problematic samples
        if "skip_sample" in processed_batch.features:
            before_count = len(processed_batch)
            processed_batch = processed_batch.filter(lambda x: not x.get("skip_sample", False))
            after_count = len(processed_batch)
            logger.info(f"Filtered out {before_count - after_count} problematic samples in batch {i+1}")
        
        # Free memory
        free_memory()
        log_memory_usage()
        
        # Add to list of processed batches
        all_processed_datasets.append(processed_batch)
    
    # Combine all processed batches
    logger.info("Combining all processed batches")
    if all_processed_datasets:
        combined_dataset = concatenate_datasets(all_processed_datasets)
        logger.info(f"Successfully processed {len(combined_dataset)} samples")
        return combined_dataset
    else:
        logger.error("No samples were successfully processed")
        raise ValueError("Dataset processing failed - no valid samples")
    
def process_dataset_with_checkpoint(dataset, batch_size=1000, checkpoint_every=5, use_memory_mapping=True):
    """
    Process dataset in smaller batches with periodic checkpointing to avoid memory buildup
    
    Args:
        dataset: The dataset to process
        batch_size: Size of each batch to process
        checkpoint_every: Number of batches to process before saving to disk
    
    Returns:
        Processed dataset
    """
    logger.info(f"Processing dataset with {len(dataset)} samples with checkpoints")
    
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    processed_samples = 0
    checkpoint_counter = 0
    checkpoint_paths = []
    
    # Create temp directory for checkpoints
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="whisper_processing_")
    logger.info(f"Using temporary directory for checkpoints: {temp_dir}")
    
    try:
        # Process in groups of checkpoint_every batches
        for checkpoint_idx in range((total_batches + checkpoint_every - 1) // checkpoint_every):
            start_batch = checkpoint_idx * checkpoint_every
            end_batch = min((checkpoint_idx + 1) * checkpoint_every, total_batches)
            
            logger.info(f"Processing checkpoint group {checkpoint_idx+1}: batches {start_batch+1}-{end_batch} of {total_batches}")
            
            checkpoint_datasets = []
            
            # Process each batch in this checkpoint group
            for i in range(start_batch, end_batch):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(dataset))
                
                logger.info(f"Processing batch {i+1}/{total_batches} (samples {start_idx}-{end_idx})")
                
                # Get the current batch - use try-except for robustness
                try:
                    batch_dataset = dataset.select(range(start_idx, end_idx))
                except Exception as e:
                    logger.error(f"Error selecting batch {i+1}: {e}")
                    continue
                
                # Process the batch with a single process
                try:
                    processed_batch = batch_dataset.map(
                        prepare_dataset_mucs,
                        remove_columns=batch_dataset.column_names,
                        num_proc=1,
                        desc=f"Batch {i+1}/{total_batches}",
                    )
                    
                    # Filter out problematic samples
                    if "skip_sample" in processed_batch.features:
                        before_count = len(processed_batch)
                        processed_batch = processed_batch.filter(lambda x: not x.get("skip_sample", False))
                        after_count = len(processed_batch)
                        if before_count != after_count:
                            logger.info(f"Filtered out {before_count - after_count} problematic samples in batch {i+1}")
                    
                    # Add to list for this checkpoint
                    checkpoint_datasets.append(processed_batch)
                    processed_samples += len(processed_batch)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {e}")
                    continue
                
                # Force garbage collection after each batch
                free_memory()
                log_memory_usage()
            
            # If we processed any batches in this checkpoint, save them
            if checkpoint_datasets:
                # Combine batches in this checkpoint
                try:
                    combined_checkpoint = concatenate_datasets(checkpoint_datasets)
                    
                    # Save this checkpoint to disk
                    checkpoint_path = os.path.join(temp_dir, f"checkpoint_{checkpoint_idx}.arrow")
                    combined_checkpoint.save_to_disk(checkpoint_path)
                    checkpoint_paths.append(checkpoint_path)
                    logger.info(f"Saved checkpoint {checkpoint_idx+1} with {len(combined_checkpoint)} samples to {checkpoint_path}")
                    
                    # Clear the batch list to free memory
                    checkpoint_datasets = []
                    
                except Exception as e:
                    logger.error(f"Error saving checkpoint {checkpoint_idx+1}: {e}")
            
            # Force memory cleanup after saving
            free_memory()
            log_memory_usage()
            
            # Additional cleanup to ensure memory is released
            import sys
            for obj in locals().values():
                if obj not in [temp_dir, checkpoint_paths, dataset, batch_size, checkpoint_every]:
                    try:
                        del obj
                    except:
                        pass
            free_memory()
        
        # After all checkpoints are processed, load and combine them
        # Modified checkpoint combination approach - incremental loading
        logger.info(f"Loading and combining {len(checkpoint_paths)} checkpoints incrementally")
        combined_dataset = None
        
        for i, path in enumerate(checkpoint_paths):
            try:
                logger.info(f"Loading checkpoint {i+1}/{len(checkpoint_paths)}")
                # When loading checkpoints, use memory mapping
                if use_memory_mapping:
                    checkpoint_data = Dataset.load_from_disk(path, keep_in_memory=False)
                else:
                    checkpoint_data = Dataset.load_from_disk(path)
                logger.info(f"Loaded checkpoint {i+1} with {len(checkpoint_data)} samples")
                
                # For the first checkpoint, just assign it
                if combined_dataset is None:
                    combined_dataset = checkpoint_data
                # For subsequent checkpoints, concatenate incrementally
                else:
                    combined_dataset = concatenate_datasets([combined_dataset, checkpoint_data])
                    logger.info(f"Combined dataset now has {len(combined_dataset)} samples")
                
                # Important: Remove the reference to checkpoint_data to free memory
                del checkpoint_data
                free_memory_aggressive()
                log_memory_usage()
                
            except Exception as e:
                logger.error(f"Error loading checkpoint {i+1}: {e}")
        
        if combined_dataset is not None:
            logger.info(f"Successfully processed {len(combined_dataset)} samples out of original {len(dataset)}")
            return combined_dataset
        else:
            raise ValueError("All checkpoints failed - no data processed")
            
    except Exception as e:
        logger.error(f"Error in dataset processing: {e}")
        raise
    finally:
        # Clean up temp files
        import shutil
        try:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        except:
            logger.warning(f"Could not clean up temporary directory: {temp_dir}")

# compare with free_memory()
def free_memory_aggressive():
    """
    Aggressively free memory with multiple garbage collection passes and object deletion
    """
    # First, regular garbage collection
    import gc
    gc.collect()
    
    # Force garbage collection with multiple passes
    for _ in range(3):
        gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Additional CUDA memory management
        try:
            torch.cuda.synchronize()  # Wait for CUDA operations to complete
        except:
            pass
    
    # Force Python to release memory back to OS if possible
    if sys.platform.startswith('linux'):
        try:
            # On Linux, try to release memory back to OS
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
        except:
            pass

def optimize_dataset_memory(dataset):
    """
    Optimize dataset to use less memory by:
    1. Converting to numpy arrays where possible
    2. Using more efficient data types
    3. Using memory mapping for large arrays
    """
    logger.info("Optimizing dataset memory usage")
    
    # Define a function to process each sample
    def optimize_sample(sample):
        # Convert float64 features to float32 to save memory
        if 'input_features' in sample:
            try:
                if isinstance(sample['input_features'], np.ndarray) and sample['input_features'].dtype == np.float64:
                    sample['input_features'] = sample['input_features'].astype(np.float32)
            except Exception as e:
                logger.warning(f"Error optimizing input features: {e}")
        return sample
    
    # Apply optimization to dataset
    try:
        optimized_dataset = dataset.map(
            optimize_sample,
            desc="Optimizing memory usage",
            num_proc=1,
        )
        return optimized_dataset
    except Exception as e:
        logger.error(f"Error optimizing dataset: {e}")
        return dataset  # Return original dataset if optimization fails

def apply_memory_limit():
    """
    Try to apply memory limits to prevent OOM errors
    """
    import resource
    
    try:
        # Get system memory information
        import psutil
        mem = psutil.virtual_memory()
        total_mem = mem.total / (1024 * 1024 * 1024)  # Convert to GB
        
        # Set memory limit to 80% of total memory
        mem_limit = int(total_mem * 0.8 * 1024 * 1024 * 1024)  # Convert back to bytes
        
        # Try to set memory limit (this may not work on all systems)
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        
        logger.info(f"Memory limit set to {mem_limit/(1024*1024*1024):.1f} GB (80% of total {total_mem:.1f} GB)")
    except Exception as e:
        logger.warning(f"Could not set memory limit: {e}")

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

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Main execution code
if __name__ == "__main__":
    # Import additional libraries needed for memory management
    import sys
    import gc
    
    # Configuration - MODIFY THESE VALUES
    MUCS_DATA_DIR = "datasets/mucs"  # Update with actual path to MUCS dataset
    MODEL_NAME = "openai/whisper-medium"  # Using whisper-medium model
    # LANGUAGE = "hi"  # Hindi as primary language, code-switched with English
    OUTPUT_DIR = "./whisper-finetuned-mucs"
    BATCH_SIZE = 4  # Reduced batch size for RTX 4060 8GB VRAM
    LEARNING_RATE = 5e-5  # Slightly increased learning rate
    MAX_STEPS = 10000  # Increased steps for 90-hour dataset
    GRADIENT_ACCUMULATION = 2  # Accumulate gradients to simulate larger batch
    PROCESSING_BATCH_SIZE = 500  # REDUCED from 1000 to 500 - smaller batches
    CHECKPOINT_EVERY = 1  # Save checkpoint after every 2 batches
    
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Try to set memory limits
    apply_memory_limit()
    
    try:
        # Log memory at start
        logger.info("Starting fine-tuning pipeline")
        log_memory_usage()
        
        # Step 1: Load MUCS dataset
        logger.info(f"Loading MUCS dataset from {MUCS_DATA_DIR}")
        try:
            dataset_dict = load_mucs_dataset(MUCS_DATA_DIR)
            train_dataset = dataset_dict["train"]
            test_dataset = dataset_dict["test"]
            logger.info(f"Dataset loaded. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

        # NOW load model components after dataset is processed
        logger.info(f"Loading model components for {MODEL_NAME}")
        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
            
            # For Hindi-English code-switching, we'll use Hindi as the base language
            tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task="transcribe")
            processor = WhisperProcessor.from_pretrained(MODEL_NAME, task="transcribe")
            
            # Log information about the model and tokenizer
            logger.info(f"Task set to: transcribe")
            logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
            logger.info("Processor components loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            raise
        
        # Step 2: Load model components - do this AFTER dataset processing to save memory
        logger.info("Preprocessing training dataset")
        try:
            # Process training dataset with checkpointing to avoid memory growth
            train_dataset = process_dataset_with_checkpoint(
                train_dataset, 
                batch_size=PROCESSING_BATCH_SIZE,
                checkpoint_every=CHECKPOINT_EVERY
            )
            
            # Optimize memory usage
            train_dataset = optimize_dataset_memory(train_dataset)
            
            # Free memory after training dataset processing
            free_memory_aggressive()
            log_memory_usage()
            
            # Process test dataset - it's small enough to process in one go
            logger.info("Preprocessing test dataset")
            test_dataset = test_dataset.map(
                prepare_dataset_mucs,
                remove_columns=test_dataset.column_names,
                num_proc=1,
                desc="Processing test data",
            )
            
            # Filter out samples that had processing errors
            if "skip_sample" in test_dataset.features:
                before_count = len(test_dataset)
                test_dataset = test_dataset.filter(lambda x: not x.get("skip_sample", False))
                after_count = len(test_dataset)
                logger.info(f"Filtered out {before_count - after_count} problematic test samples")
            
            # Optimize test dataset memory
            test_dataset = optimize_dataset_memory(test_dataset)
            
            logger.info(f"Preprocessing completed. Final dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")
            
            # Free memory after preprocessing
            free_memory_aggressive()
            log_memory_usage()
        except Exception as e:
            logger.error(f"Failed during dataset preprocessing: {str(e)}")
            raise
            
        # Verify that datasets contain the expected features
        expected_features = ["input_features", "labels"]
        if not all(feat in train_dataset.features for feat in expected_features):
            missing = [f for f in expected_features if f not in train_dataset.features]
            logger.error(f"Training dataset is missing expected features: {missing}")
            raise ValueError(f"Training dataset is missing required features: {missing}")
            
        if not all(feat in test_dataset.features for feat in expected_features):
            missing = [f for f in expected_features if f not in test_dataset.features]
            logger.error(f"Test dataset is missing expected features: {missing}")
            raise ValueError(f"Test dataset is missing required features: {missing}")
            
        
        # Step 4: Create data collator
        try:
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=processor,
                decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            )
            logger.info("Data collator initialized")
        except Exception as e:
            logger.error(f"Failed to create data collator: {str(e)}")
            raise
        
        # Step 5: Setup evaluation metrics
        wer_metric = evaluate.load("wer")
        
        # Step 6: Initialize model with optimizations for Hindi-English code-switching
        logger.info(f"Initializing model {MODEL_NAME}")
        
        # Free up memory before loading the model
        free_memory_aggressive()
        
        try:
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
            
            # Model configuration for Hindi-English code-switching
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            
            # Set up for Hindi-English code-switching - force language ID token for Hindi
            # This helps the model focus on Hindi while still handling English portions
            forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
            model.config.forced_decoder_ids = forced_decoder_ids
            
            logger.info(f"Model initialized successfully with forced decoder IDs: {forced_decoder_ids}")
            
            # Log model size and parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model size: {total_params/1e6:.2f}M parameters")
            
            # Log memory after model loading
            log_memory_usage()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
        
        # Step 7: Configure training arguments
        try:
            training_args = Seq2SeqTrainingArguments(
                output_dir=OUTPUT_DIR,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                learning_rate=LEARNING_RATE,
                warmup_steps=1000,
                max_steps=MAX_STEPS,
                gradient_checkpointing=True,  # Memory optimization
                fp16=True,  # Use half precision for memory efficiency
                evaluation_strategy="steps",
                per_device_eval_batch_size=BATCH_SIZE,
                predict_with_generate=True,
                generation_max_length=225,
                save_steps=1000,
                eval_steps=500,
                logging_steps=100,
                report_to=["tensorboard"],
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                logging_dir="./logs",
                # Memory and performance optimizations
                dataloader_num_workers=1,  # Reduced to 1 for stability
                group_by_length=True,
                save_total_limit=2,  # Keep only 2 checkpoints
                # Memory optimizations
                optim="adamw_torch",
                ddp_find_unused_parameters=False,
                dataloader_pin_memory=False,  # Disable pinning memory
            )
            logger.info("Training arguments configured")
        except Exception as e:
            logger.error(f"Failed to configure training arguments: {str(e)}")
            raise
        
        # Step 8: Initialize trainer
        logger.info("Initializing Trainer")
        try:
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
            logger.error(f"Failed to initialize trainer: {str(e)}")
            raise
        
        # Log memory before training
        log_memory_usage()
        
        # Step 9: Start training
        logger.info("Starting training process")
        try:
            train_result = trainer.train()
            logger.info("Training completed successfully")
            logger.info(f"Training metrics: {train_result.metrics}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        
        # Step 10: Save model
        logger.info("Saving final model")
        try:
            trainer.save_model()
            processor.save_pretrained(training_args.output_dir)
            logger.info(f"Model saved to {training_args.output_dir}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
        
        # Step 11: Final evaluation
        logger.info("Starting final evaluation")
        try:
            metrics = trainer.evaluate(test_dataset)
            logger.info("Evaluation metrics:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")
                
            logger.info("Fine-tuning pipeline completed successfully")
        except Exception as e:
            logger.error(f"Final evaluation failed: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise