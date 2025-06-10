import os
import gspread
from google.auth import default
import datetime
import math # For math.ceil

# --- Configuration ---
# OUTPUT_ROOT_DIR: Root directory where segmented text files will be saved
OUTPUT_ROOT_DIR = "./output_transcriptions"
SPREADSHEET_NAME = "T095P002" # Placeholder for user input, as per plan
# MAX_CHARS_PER_FILE and MAX_LINES_PER_FILE are no longer needed for time-based segmentation

# --- Helper Functions ---
def srt_time_to_seconds(time_str):
    """Converts SRT time string "HH:MM:SS,ms" to total seconds."""
    if not time_str or time_str.count(':') != 2 or ',' not in time_str:
        # print(f"Warning: Invalid SRT time string format: '{time_str}'. Returning 0.")
        # Or raise an error: raise ValueError(f"Invalid SRT time string format: '{time_str}'")
        return 0.0 # Default to 0 or handle error as appropriate
    parts = time_str.split(',')
    h_m_s = parts[0].split(':')
    hours = int(h_m_s[0])
    minutes = int(h_m_s[1])
    seconds = int(h_m_s[2])
    milliseconds = int(parts[1])
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds

def seconds_to_srt_time(total_seconds):
    """Converts total seconds to SRT time string "HH:MM:SS,ms"."""
    if total_seconds < 0:
        total_seconds = 0 # Or handle error
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    secs_float = total_seconds % 60
    seconds = int(secs_float)
    milliseconds = int((secs_float - seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def authenticate():
    """Handles Google Sheets authentication using Application Default Credentials."""
    print("Authenticating for Google Sheets...")
    try:
        creds, _ = default()
        gc = gspread.authorize(creds)
        print("Authentication successful.")
        return gc
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None

# create_output_subdir is no longer needed as the directory structure is simpler.
# The specific output directory will be SPREADSHEET_NAME inside OUTPUT_ROOT_DIR.

def main():
    # Global SPREADSHEET_NAME will be updated by this prompt
    global SPREADSHEET_NAME

    user_spreadsheet_name = input(f"Enter the name of the Google Spreadsheet to process (default: '{SPREADSHEET_NAME}'): ")
    if user_spreadsheet_name.strip():
        SPREADSHEET_NAME = user_spreadsheet_name.strip()
    else:
        # If user enters nothing, keep the default.
        # If default is also empty or you want to force input, add more checks.
        if not SPREADSHEET_NAME: # Handles if default was also empty
             print("No spreadsheet name provided. Exiting.")
             return
        print(f"No input given, using default spreadsheet name: '{SPREADSHEET_NAME}'")

    gc = authenticate()
    if not gc:
        print("Exiting due to authentication failure.")
        return

    print(f"Script to process spreadsheet: {SPREADSHEET_NAME}")

    # Create the root output directory if it doesn't exist
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)
        print(f"Created root output directory: {OUTPUT_ROOT_DIR}")
    else:
        print(f"Root output directory already exists: {OUTPUT_ROOT_DIR}")

    # --- Prepare Output Directory ---
    output_spreadsheet_dir = os.path.join(OUTPUT_ROOT_DIR, SPREADSHEET_NAME)
    try:
        os.makedirs(output_spreadsheet_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_spreadsheet_dir}")
    except OSError as e:
        print(f"Error creating directory {output_spreadsheet_dir}: {e}")
        return

    print(f"All output for this run will be saved in: {output_spreadsheet_dir}")

    # --- Open Spreadsheet and Access Worksheets ---
    try:
        print(f"Attempting to open spreadsheet: '{SPREADSHEET_NAME}'...")
        spreadsheet = gc.open(SPREADSHEET_NAME)
        print(f"Successfully opened spreadsheet: '{SPREADSHEET_NAME}' (ID: {spreadsheet.id})")

        text_ws_title = "文本校對"
        timeline_ws_title = "時間軸"

        print(f"Attempting to open worksheet: '{text_ws_title}'...")
        text_worksheet = spreadsheet.worksheet(text_ws_title)
        print(f"Successfully opened worksheet: '{text_worksheet.title}'")

        print(f"Attempting to open worksheet: '{timeline_ws_title}'...")
        timeline_worksheet = spreadsheet.worksheet(timeline_ws_title)
        print(f"Successfully opened worksheet: '{timeline_worksheet.title}'")

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet '{SPREADSHEET_NAME}' not found.")
        return
    except gspread.exceptions.WorksheetNotFound as e:
        print(f"Error: A required worksheet was not found in spreadsheet '{SPREADSHEET_NAME}'. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while opening spreadsheet/worksheets: {e}")
        return

    # --- Extract Data ---
    try:
        print("Reading data from '文本校對' worksheet...")
        text_rows = text_worksheet.get_all_values()
        if not text_rows or len(text_rows) <= 1: # Assuming header row
            print(f"Worksheet '{text_ws_title}' is empty or has no data beyond headers.")
            return
        # Corrected text from Column B (index 1), skipping header
        corrected_texts = [row[1].strip() if len(row) > 1 else "" for row in text_rows[1:]]
        print(f"Extracted {len(corrected_texts)} text entries from '{text_ws_title}'.")

        print("Reading data from '時間軸' worksheet...")
        timeline_rows = timeline_worksheet.get_all_values()
        if not timeline_rows or len(timeline_rows) <= 1: # Assuming header row
            print(f"Worksheet '{timeline_ws_title}' is empty or has no data beyond headers.")
            return
        # Start time (Col B, index 1), End time (Col C, index 2), skipping header
        time_data = []
        for row_idx, row in enumerate(timeline_rows[1:]):
            start_str = row[1] if len(row) > 1 else ""
            end_str = row[2] if len(row) > 2 else ""
            if not start_str or not end_str:
                print(f"Warning: Missing start or end time in '{timeline_ws_title}' at row {row_idx + 2}. Skipping this entry.")
                # Add a placeholder or decide how to handle incomplete time data
                # For now, we'll create an entry that might fail srt_time_to_seconds if empty
            time_data.append({'start_str': start_str, 'end_str': end_str})
        print(f"Extracted {len(time_data)} time entries from '{timeline_ws_title}'.")

        # Validation
        if len(corrected_texts) != len(time_data):
            print(f"Error: Mismatch in number of entries. Text entries: {len(corrected_texts)}, Time entries: {len(time_data)}.")
            print("A one-to-one correspondence is expected. Please check the worksheets.")
            return
        if not corrected_texts: # or not time_data, already covered by len check
             print("No data extracted. Exiting.")
             return

    except Exception as e:
        print(f"An error occurred while reading or processing worksheet data: {e}")
        return

    # --- Calculate Total Duration and Parts ---
    total_audio_duration_seconds = 0
    if time_data:
        last_end_time_str = time_data[-1]['end_str']
        total_audio_duration_seconds = srt_time_to_seconds(last_end_time_str)
        if total_audio_duration_seconds == 0 and last_end_time_str != "00:00:00,000": # check if it was an invalid string
             print(f"Warning: The last segment's end time '{last_end_time_str}' is invalid or zero. Total duration may be incorrect.")


    segment_duration_seconds = 30 * 60  # 30 minutes
    total_parts = math.ceil(total_audio_duration_seconds / segment_duration_seconds) if total_audio_duration_seconds > 0 else 1
    print(f"Total audio duration: {seconds_to_srt_time(total_audio_duration_seconds)} ({total_audio_duration_seconds:.2f}s). This will be split into {total_parts} part(s).")

    # --- Process and Segment Text ---
    part_number = 1
    current_part_texts = []
    # current_part_actual_start_seconds is the theoretical start of the 30-min block
    current_part_block_start_seconds = 0.0
    # current_part_first_item_actual_start_seconds is the actual start time of the first item in the current part
    current_part_first_item_actual_start_seconds = -1.0 # Use -1 to indicate not yet set for the current part


    for i in range(len(time_data)):
        item_text = corrected_texts[i]
        item_start_str = time_data[i]['start_str']
        item_end_str = time_data[i]['end_str'] # For determining actual end of part

        item_start_seconds = srt_time_to_seconds(item_start_str)
        item_end_seconds = srt_time_to_seconds(item_end_str) # For determining actual end of part

        current_part_target_end_seconds = part_number * segment_duration_seconds

        if current_part_first_item_actual_start_seconds == -1.0: # First item for this part
            current_part_first_item_actual_start_seconds = item_start_seconds
            current_part_block_start_seconds = (part_number - 1) * segment_duration_seconds


        # Check if item belongs to a new part
        if item_start_seconds >= current_part_target_end_seconds and current_part_texts:
            # Write current part to file
            txt_filename = os.path.join(output_spreadsheet_dir, f"{SPREADSHEET_NAME}M{part_number:02d}.txt")
            header_line1 = f"{SPREADSHEET_NAME} Part {part_number} of {total_parts}"

            # Actual end time for this part is the end time of the *last item fully contained* in this segment
            # or the target end if no item crossed it.
            # The last item processed *before* this new item (which starts the *next* part) defines the end of the current part.
            actual_part_end_time_seconds = srt_time_to_seconds(time_data[i-1]['end_str']) if i > 0 else item_end_seconds

            header_line2 = f"{seconds_to_srt_time(current_part_first_item_actual_start_seconds)} --> {seconds_to_srt_time(min(actual_part_end_time_seconds, current_part_target_end_seconds, total_audio_duration_seconds))}"

            try:
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(header_line1 + "\n")
                    f.write(header_line2 + "\n\n")
                    f.write("\n".join(current_part_texts))
                print(f"Generated: {txt_filename} ({len(current_part_texts)} text items)")
            except IOError as e:
                print(f"Error writing to file {txt_filename}: {e}")

            # Advance to next part
            part_number += 1
            current_part_texts = []
            current_part_first_item_actual_start_seconds = item_start_seconds # This item is the start of the new part
            current_part_block_start_seconds = (part_number - 1) * segment_duration_seconds


        current_part_texts.append(item_text)

    # After the loop, write any remaining part
    if current_part_texts:
        txt_filename = os.path.join(output_spreadsheet_dir, f"{SPREADSHEET_NAME}M{part_number:02d}.txt")
        header_line1 = f"{SPREADSHEET_NAME} Part {part_number} of {total_parts}"

        # Actual end time for the last part is simply the end time of the very last item, capped by total duration
        actual_part_end_time_seconds = srt_time_to_seconds(time_data[-1]['end_str']) if time_data else current_part_target_end_seconds

        header_line2 = f"{seconds_to_srt_time(current_part_first_item_actual_start_seconds if current_part_first_item_actual_start_seconds != -1.0 else 0.0)} --> {seconds_to_srt_time(min(actual_part_end_time_seconds, total_audio_duration_seconds))}"

        try:
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(header_line1 + "\n")
                f.write(header_line2 + "\n\n")
                f.write("\n".join(current_part_texts))
            print(f"Generated: {txt_filename} ({len(current_part_texts)} text items)")
        except IOError as e:
            print(f"Error writing final segment to file {txt_filename}: {e}")

    print(f"\nTime-based segmentation complete. Total parts generated: {part_number if current_part_texts or part_number > 1 else 0}")


if __name__ == '__main__':
    main()
