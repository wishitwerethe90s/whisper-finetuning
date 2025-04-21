import os
import torch
import numpy as np
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import datetime
import warnings
from secret_keys import huggingface_token

warnings.filterwarnings("ignore")

class SpeakerDiarization:
    def __init__(self, pyannote_auth_token=None, whisper_model="base"):
        """
        Initialize the speaker diarization and transcription pipeline.
        
        Args:
            pyannote_auth_token (str): HuggingFace token for pyannote.audio
            whisper_model (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Whisper model
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        # Initialize Pyannote diarization pipeline
        if not pyannote_auth_token:
            print("Warning: No Pyannote auth token provided. You'll need to set this up.")
            print("Get a token at https://huggingface.co/pyannote/speaker-diarization and accept the user agreement")
            self.diarization_pipeline = None
        else:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=pyannote_auth_token
            ).to(self.device)
    
    def _format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(datetime.timedelta(seconds=int(seconds)))
    
    def extract_segment(self, audio_file, start_time, end_time):
        """Extract a segment from the audio file"""
        audio = AudioSegment.from_file(audio_file)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment = audio[start_ms:end_ms]
        return segment
        
    def transcribe_segment(self, segment):
        """Transcribe an audio segment using Whisper"""
        # Save segment to a temporary file
        temp_path = "temp_segment.wav"
        segment.export(temp_path, format="wav")
        
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(temp_path, fp16=False)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return result["text"].strip()
    
    def process_audio(self, audio_file, min_segment_length=1.0):
        """
        Process an audio file to get diarized transcription.
        
        Args:
            audio_file (str): Path to the audio file
            min_segment_length (float): Minimum segment length in seconds
            
        Returns:
            list: List of dictionaries with speaker, start time, end time, and transcription
        """
        if self.diarization_pipeline is None:
            raise ValueError("Diarization pipeline not initialized. Please provide a valid auth token.")
        
        # Perform diarization
        print("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_file)
        
        # Process results
        print("Transcribing segments...")
        results = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            duration = end_time - start_time
            
            # Skip very short segments
            if duration < min_segment_length:
                continue
                
            # Extract and transcribe this segment
            segment = self.extract_segment(audio_file, start_time, end_time)
            transcription = self.transcribe_segment(segment)
            
            # Add to results
            results.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "start_formatted": self._format_timestamp(start_time),
                "end_formatted": self._format_timestamp(end_time),
                "text": transcription
            })
            
        return results
    
    def save_transcript(self, results, output_file):
        """Save the diarized transcript to a file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in results:
                f.write(f"[{segment['speaker']}] {segment['start_formatted']} - {segment['end_formatted']}: {segment['text']}\n\n")
        print(f"Transcript saved to {output_file}")
        
    def display_transcript(self, results):
        """Display the diarized transcript"""
        for segment in results:
            print(f"[{segment['speaker']}] {segment['start_formatted']} - {segment['end_formatted']}: {segment['text']}")
            print()

# Example usage
if __name__ == "__main__":
    # Get your token from: https://huggingface.co/pyannote/speaker-diarization
    PYANNOTE_AUTH_TOKEN = huggingface_token
    
    # Initialize the pipeline
    diarizer = SpeakerDiarization(
        pyannote_auth_token=PYANNOTE_AUTH_TOKEN,
        whisper_model="tiny"  # Use larger models like "medium" or "large" for better transcription
    )
    
    # Process an audio file
    audio_file = "samples/joe-rogan-[01.00-02.00].mp3"
    results = diarizer.process_audio(audio_file)
    
    # Display results
    diarizer.display_transcript(results)
    
    # Save transcript to file
    diarizer.save_transcript(results, "diarized_transcript.txt")