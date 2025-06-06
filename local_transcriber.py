import os
import datetime
from faster_whisper import WhisperModel # Corrected import
import logging
import json
# google.colab.drive will be imported within the main function for conditional mounting

# --- Configuration Variables ---
MODEL_SIZE = "large-v3"
INITIAL_PROMPT_TEXT = "這是佛教關於密教真言宗藥師佛" # Consider making this configurable if needed
INPUT_AUDIO_DIR = "/content/drive/MyDrive/input_audio"
OUTPUT_TRANSCRIPTIONS_ROOT_DIR = "/content/drive/MyDrive/output_transcriptions" # Root for new subdirs
STATE_FILE_PATH = os.path.join(OUTPUT_TRANSCRIPTIONS_ROOT_DIR, ".processed_audio_files.json")


# --- Helper Functions ---
def load_processed_files(state_file_path):
    try:
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure it's a list, convert to set for efficient lookup
                return set(data if isinstance(data, list) else [])
        logging.info(f"State file '{state_file_path}' not found. Starting fresh.")
    except json.JSONDecodeError:
        logging.warning(f"Error decoding state file '{state_file_path}'. Starting fresh.")
    except Exception as e:
        logging.error(f"Error loading state file '{state_file_path}': {e}. Starting fresh.", exc_info=True)
    return set()

def save_processed_files(state_file_path, processed_files_set):
    temp_state_file_path = state_file_path + ".tmp"
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        with open(temp_state_file_path, 'w', encoding='utf-8') as f:
            # Convert set to list for JSON serialization
            json.dump(list(processed_files_set), f, ensure_ascii=False, indent=4)
        # Atomic rename (on POSIX systems)
        os.replace(temp_state_file_path, state_file_path)
        logging.debug(f"Successfully saved state to '{state_file_path}'.") # Changed to debug
    except Exception as e:
        logging.error(f"Error saving state to '{state_file_path}': {e}", exc_info=True)
        if os.path.exists(temp_state_file_path):
            try:
                os.remove(temp_state_file_path)
            except OSError as oe:
                logging.error(f"Error removing temporary state file '{temp_state_file_path}': {oe}", exc_info=True)

def format_srt_time(seconds):
    """將秒數格式化為 SRT 格式的時間字串 (HH:MM:SS,ms)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def main():
    # --- Basic Logging Configuration ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Log to console
            # Optionally, add a FileHandler later if needed:
            # logging.FileHandler('local_transcriber.log')
        ]
    )
    logging.info("local_transcriber.py script started.")

    # --- Load State ---
    processed_files = load_processed_files(STATE_FILE_PATH)
    logging.info(f"Loaded {len(processed_files)} processed file name(s) from state: {STATE_FILE_PATH}")

    # --- Mount Google Drive ---
    logging.info("Attempting to mount Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        logging.info("Google Drive mounted successfully.")
    except ImportError:
        logging.warning("Skipping Drive mount, not in Colab or google.colab not available.")
        # Depending on whether Drive is strictly necessary, you might exit here
        # For local execution without Colab, INPUT_AUDIO_DIR would need to be a local path
    except Exception as e:
        logging.error(f"Error mounting Google Drive: {e}", exc_info=True)
        # Potentially exit if Drive is essential and mount fails
        return # Exit if drive mount fails and it's considered critical

    # --- Load Faster Whisper Model ---
    logging.info(f"Loading Faster Whisper model: {MODEL_SIZE}...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        logging.info("Faster Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Faster Whisper model: {e}", exc_info=True)
        logging.error("This script requires a CUDA-enabled GPU and appropriate libraries.")
        logging.error("Ensure PyTorch and CTranslate2 with CUDA support are correctly installed.")
        return

    # --- Check Input Directory ---
    if not os.path.exists(INPUT_AUDIO_DIR):
        logging.error(f"Input audio directory '{INPUT_AUDIO_DIR}' not found. Exiting.")
        return

    logging.info(f"Input audio directory: {INPUT_AUDIO_DIR}")
    logging.info(f"Output root directory: {OUTPUT_TRANSCRIPTIONS_ROOT_DIR}")
    logging.info(f"State file path: {STATE_FILE_PATH}")

    # --- Iterate Through Audio Files ---
    audio_files_to_process = [f for f in os.listdir(INPUT_AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.mp4'))]

    if not audio_files_to_process:
        logging.info(f"No audio files found in '{INPUT_AUDIO_DIR}'.")
        return

    logging.info(f"Found {len(audio_files_to_process)} audio file(s) to process.")

    unwanted_phrase = "字幕由 Amara.org 社群提供" # As per original script

    for audio_file_name in audio_files_to_process:
        # Use audio_file_name as the unique identifier for processed state
        if audio_file_name in processed_files:
            logging.info(f"Skipping '{audio_file_name}' as it was already processed.")
            continue

        base_name = os.path.splitext(audio_file_name)[0]
        audio_path = os.path.join(INPUT_AUDIO_DIR, audio_file_name)

        logging.info(f"--- Processing file: {audio_path} ---")

        # --- Transcription ---
        try:
            logging.info(f"Starting transcription for {audio_file_name}...")
            # VAD parameters from the original script (can be made configurable)
            vad_parameters = {
                "min_speech_duration_ms": 50,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 500,
            }
            segments_generator, info = model.transcribe(
                audio_path,
                beam_size=5,
                initial_prompt=INITIAL_PROMPT_TEXT,
                vad_filter=True,
                vad_parameters=vad_parameters
            )
            segments_list = list(segments_generator) # Consume generator
            logging.info(f"Transcription finished for {audio_file_name}. Language: {info.language} with probability {info.language_probability:.2f}")
            logging.info(f"Detected {len(segments_list)} segments for {audio_file_name}.")

        except Exception as e:
            logging.error(f"Error during transcription for {audio_file_name}: {e}", exc_info=True)
            continue # Skip to the next file

        # --- Generate "Normal Text" ---
        whisper_transcription_lines = []
        for segment in segments_list:
            cleaned_text = segment.text.strip().replace(unwanted_phrase, "").strip()
            if cleaned_text: # Only add if there's actual text after cleaning
                whisper_transcription_lines.append(cleaned_text)

        normal_text_content = "\n".join(whisper_transcription_lines)
        # print(f"\nNormal Text Preview (first 100 chars for {base_name}):\n'{normal_text_content[:100]}...'") # For debugging

        # --- Generate SRT Content ---
        srt_content = ""
        srt_sequence_number = 1
        for segment in segments_list:
            start_time_srt = format_srt_time(segment.start)
            end_time_srt = format_srt_time(segment.end)
            cleaned_segment_text = segment.text.strip().replace(unwanted_phrase, "").strip()

            if cleaned_segment_text: # Only include segments with actual text in SRT
                srt_content += f"{srt_sequence_number}\n"
                srt_content += f"{start_time_srt} --> {end_time_srt}\n"
                srt_content += f"{cleaned_segment_text}\n\n"
                srt_sequence_number += 1
        # print(f"\nSRT Content Preview (first 100 chars for {base_name}):\n'{srt_content[:100]}...'") # For debugging


        # --- Output to Files ---
        output_dir_for_file = os.path.join(OUTPUT_TRANSCRIPTIONS_ROOT_DIR, base_name)
        try:
            os.makedirs(output_dir_for_file, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating output directory {output_dir_for_file}: {e}", exc_info=True)
            continue # Skip to next file if directory creation fails

        normal_text_filename = f"{base_name}_normal.txt"
        normal_text_path = os.path.join(output_dir_for_file, normal_text_filename)

        srt_filename = f"{base_name}.srt"
        srt_path = os.path.join(output_dir_for_file, srt_filename)

        try:
            with open(normal_text_path, "w", encoding="utf-8") as f:
                f.write(normal_text_content)
            logging.info(f"Successfully wrote normal text to: {normal_text_path}")
        except IOError as e:
            logging.error(f"Error writing normal text to {normal_text_path}: {e}", exc_info=True)

        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            logging.info(f"Successfully wrote SRT to: {srt_path}")
        except IOError as e:
            logging.error(f"Error writing SRT to {srt_path}: {e}", exc_info=True)
            continue # If file writing fails, don't mark as processed

        # If all outputs for this file are successfully saved, mark as processed
        processed_files.add(audio_file_name)
        save_processed_files(STATE_FILE_PATH, processed_files)
        logging.info(f"Marked '{audio_file_name}' as processed and updated state file.")

    logging.info("All audio files processed.")
    logging.info("local_transcriber.py script finished.")

if __name__ == "__main__":
    main()
