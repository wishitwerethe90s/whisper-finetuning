from speaker_diarization import SpeakerDiarization
from secret_keys import huggingface_token

# Initialize with your HuggingFace token
diarizer = SpeakerDiarization(
    pyannote_auth_token=huggingface_token,
    whisper_model="small"  # Choose model size based on your needs and hardware
)

# Process your audio file
audio_file = "/home/paarthgupta/hdfc/innovation/speech-to-text/whisper-diarization"
results = diarizer.process_audio(audio_file)

# Display the transcript with speaker labels
diarizer.display_transcript(results)

# Save to file
diarizer.save_transcript(results, "transcript.txt")