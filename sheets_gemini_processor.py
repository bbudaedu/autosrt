# 清除 pip 緩存，以確保下載最新且正確的包
!pip cache purge

# 安裝必要的函式庫
!apt update && apt install -y libcublas11 libcudnn8

# 優先強制安裝兼容的 NumPy 版本，確保沒有舊的或衝突的版本殘留
# 這是解決 RecursionError 的關鍵步驟
!pip install --force-reinstall numpy==1.26.4

print("\n--- 第一部分安裝完成 ---")
print("請務必重新啟動 Colab 執行階段 (Runtime -> Restart runtime)，然後運行第二個程式碼區塊。")


# 確保 onnxruntime 和 ctranslate2 在正確的 NumPy 環境下安裝
# 使用 --no-deps 避免 pip 重新安裝 NumPy 的最新版本
!pip install --force-reinstall --no-deps ctranslate2==4.3.1
!pip install --force-reinstall --no-deps onnxruntime-gpu==1.15.0 # <-- 明確指定 GPU 版本並使用 --no-deps

# 安裝 faster-whisper 和其他工具
# 移除了 jiwer，以解決與 NumPy 的依賴衝突
!pip install faster-whisper
!pip install gspread oauth2client
!pip install pypdf
!pip install requests

import os
from google.colab import drive, auth, userdata
# from faster_whisper import WhisperModel # Removed
import datetime # Keep for PDF date logic if any, or general utility
import gspread
from google.auth import default
import textwrap
import pypdf
import requests
import json
import time
import re # Added for SRT parsing
from IPython.display import HTML # <-- 修正：導入 HTML
import warnings # 導入 warnings 模組

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置區塊 ---
# model_size = "large-v3" # Removed - Whisper specific
# initial_prompt_text = "這是佛教關於密教真言宗藥師佛" # Removed - Whisper specific

# input_audio_dir = "/content/drive/MyDrive/input_audio" # Removed - No longer directly used for transcription input
TRANSCRIPTIONS_ROOT_INPUT_DIR = "/content/drive/MyDrive/output_transcriptions" # Renamed: Input for this script
pdf_handout_dir = "/content/drive/MyDrive/lecture_handouts" # Kept for Gemini context

# --- 輔助函式：格式化SRT時間 ---
# def format_srt_time(seconds): # Removed - Belongs to local_transcriber.py
#     """將秒數格式化為 SRT 格式的時間字串 (HH:MM:SS,ms)"""
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = seconds % 60
#     milliseconds = int((secs - int(secs)) * 1000)
#     return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

# --- 輔助函式：從 PDF 資料夾提取所有文本 ---
def extract_text_from_pdf_dir(pdf_dir):
    full_text = []
    if not os.path.exists(pdf_dir):
        print(f"錯誤: PDF 講義資料夾 '{pdf_dir}' 不存在。")
        return None

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"警告: 資料夾 '{pdf_dir}' 中沒有找到任何 PDF 檔案。")
        return None

    print(f"正在從資料夾 '{pdf_dir}' 中的 {len(pdf_files)} 個 PDF 檔案提取文本...")
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file_name)
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    full_text.append(reader.pages[page_num].extract_text())
            print(f"  - 成功提取 '{pdf_file_name}'。")
        except Exception as e:
            print(f"  - 從 '{pdf_file_name}' 提取文本時發生錯誤: {e}")
            # 即使一個 PDF 失敗，也嘗試繼續處理其他 PDF

    combined_text = "\n".join(full_text)
    if combined_text:
        print(f"所有 PDF 提取完成，共 {len(combined_text)} 字元文本。")
    else:
        print("警告: 未能從任何 PDF 檔案提取到文本。")
    return combined_text

# --- Helper Function: Parse SRT Content ---
def parse_srt_content(srt_content_str):
    segments = []
    # Regex to capture ID, start time, end time, and text
    # Handles multiline text for a single segment
    pattern = re.compile(
        r"(\d+)\s*\n"                       # ID
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"  # Start and End times
        r"((?:.+\n?)+)",                  # Text (non-greedy, captures multiple lines)
        re.MULTILINE
    )
    for match in pattern.finditer(srt_content_str):
        segments.append({
            'id': match.group(1),
            'start': match.group(2),
            'end': match.group(3),
            'text': match.group(4).strip()
        })
    return segments

# --- 輔助函式：調用 Gemini API 進行校對 ---
def get_gemini_correction(transcribed_text_lines, pdf_context):
    api_key = userdata.get('GEMINI_API_KEY')

    if not api_key:
        print("錯誤: GEMINI_API_KEY 未設定。請在 Colab Secrets 中設定您的 Gemini API 金鑰。")
        return None

    headers = {
        'Content-Type': 'application/json',
    }

    transcribed_text_single_string = "\n".join(transcribed_text_lines)

    full_prompt = f"""
    你是一個佛學大師，精通經律論三藏十二部經典。
    以下文本是whisper產生的字幕文本，關於觀無量壽經、善導大師觀經四帖疏、傳通記的內容。
    有很多聽打錯誤，幫我依據我提供的上課講義校對文本，嚴格依照以下規則，直接修正錯誤：

    上課講義內容（作為校對參考，請仔細閱讀）：
    ---
    {pdf_context}
    ---

    以下是需要校對的字幕文本 (共 {len(transcribed_text_lines)} 行):
    ---
    {transcribed_text_single_string}
    ---

    校對規則：
    1. 這是講座字幕的文本。請逐行處理提供的「字幕文本」。
    2. **嚴格依照原本的斷句輸出，保持每一行的獨立性，不要合併或拆分行。輸出結果必須與輸入的行數完全相同 (共 {len(transcribed_text_lines)} 行)。**
    3. 如果某一行不需要修改，請直接輸出原始該行內容。
    4. 根據「上課講義內容」修正「字幕文本」中的任何聽打錯誤或不準確之處。
    5. 不要加標點符號。
    6. 輸出繁體中文。
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "topK": 64,
            "maxOutputTokens": 8192,
            "responseMimeType": "text/plain",
        }
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"

    max_retries = 7
    base_delay = 60  # seconds
    response = None # Initialize response to None

    print("正在調用 Gemini API 進行文本校對，這可能需要一些時間...")
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)

            # If successful, break out of the loop
            break
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Gemini API rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Gemini API rate limit hit (429). Max retries reached. Failing. Error: {e}")
                    return None
            else:
                # For other RequestExceptions (network errors, etc.)
                print(f"調用 Gemini API 時發生錯誤: {e}")
                return None
        # If we successfully broke from the loop, response should be valid.
        # The case where response is None after the loop (e.g. max_retries for 429) is handled.

    # Proceed with processing if the request was eventually successful
    if response is None or not response.ok: # Check if response is still None or not successful
        print(f"Gemini API 調用最終失敗。") # Should be covered by returns inside loop, but as a safeguard.
        return None

    # 增加 API 調用後的延遲，以避免觸發速率限制 (This was an existing sleep)
    time.sleep(15)

    try:
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            corrected_text_from_api = result["candidates"][0]["content"]["parts"][0]["text"]
            corrected_lines = corrected_text_from_api.strip().split('\n')

            if len(corrected_lines) == len(transcribed_text_lines):
                print(f"Gemini 校對完成，行數與原始文本一致 ({len(corrected_lines)} 行)。")
                return corrected_text_from_api
            else:
                print(f"警告: Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。")
                final_corrected_lines = []
                for i in range(len(transcribed_text_lines)):
                    if i < len(corrected_lines):
                        final_corrected_lines.append(corrected_lines[i])
                    else:
                        final_corrected_lines.append(transcribed_text_lines[i])

                final_corrected_lines = final_corrected_lines[:len(transcribed_text_lines)]
                print(f"已嘗試調整行數以匹配原始文本。建議檢查校對結果。")
                return "\n".join(final_corrected_lines)
        else:
            print(f"Gemini API 響應結構異常或內容缺失: {result}")
            return None
    except json.JSONDecodeError as e:
        # response.text might not be defined if response is None, though logic above should prevent this.
        response_text = response.text if response else "No response text available"
        print(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}")
        return None
    except Exception as e: # Catch any other unexpected errors during processing
        print(f"處理 Gemini API 響應時發生未知錯誤: {e}")
        return None

# --- Google Sheets 認證與設定 ---
print("正在進行 Google 帳戶認證...")
try:
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    print("Google 帳戶認證成功。")
except Exception as e:
    print(f"Google 帳戶認證失敗: {e}")
    print("請檢查您的 Colab 環境是否已正確登入 Google 帳戶。")
    exit()

# --- 掛載 Google Drive ---
print("\n正在掛載 Google Drive...")
try:
    drive.mount('/content/drive')
    print("Google Drive 掛載成功。")
except Exception as e:
    print(f"Google Drive 掛載失敗: {e}")
    print("請檢查您的 Google Drive 權限或稍後重試。")
    exit()

# --- 檢查並提取 PDF 講義內容 ---
pdf_context_text = extract_text_from_pdf_dir(pdf_handout_dir)
if pdf_context_text is None or not pdf_context_text.strip():
    print(f"警告: 未能從資料夾 '{pdf_handout_dir}' 中的任何 PDF 檔案提取到有效文本。Gemini 校對將無法使用講義內容。")
    pdf_context_text = ""

# --- Main Processing Logic ---
# The main loop that iterated through audio files for transcription is removed.
# This script will now need a new main loop to iterate through subdirectories
# in TRANSCRIPTIONS_ROOT_INPUT_DIR (outputs from local_transcriber.py).
# This new loop will be implemented in a subsequent step.

# For now, the script structure is set up for its new role.
# The following is a placeholder for where the new main loop and its logic will go.
# It demonstrates that the Google Sheets and Gemini parts are preserved.

def process_transcriptions_and_apply_gemini():
    # This function will be fleshed out in the next subtask.
    # It will:
    # 1. List subdirectories in TRANSCRIPTIONS_ROOT_INPUT_DIR.
    # 2. For each subdirectory (representing an audio file processed by local_transcriber.py):
    #    a. Read the _normal.txt file.
    #    b. Read the .srt file.
    #    c. Create or open a Google Spreadsheet for this audio file.
    #    d. Upload normal text to "文本校對" (Column A).
    #    e. Upload SRT data to "時間軸".
    #    f. If pdf_context_text is available, call get_gemini_correction with lines from "文本校對" (Column A).
    #    g. Update "文本校對" (Column B) with Gemini's output.
    #    h. Include appropriate error handling and print/logging statements.

    # Ensure gc is available (it's initialized globally after auth)
    global gc, pdf_context_text
    if gc is None:
        print("Error: gspread client (gc) not initialized. Authentication might have failed.")
        return

    print(f"Scanning for transcription folders in {TRANSCRIPTIONS_ROOT_INPUT_DIR}...")
    if not os.path.exists(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        print(f"Error: Transcriptions input directory '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' not found.")
        print("Please ensure local_transcriber.py has run and generated output.")
        return

    processed_item_count = 0
    for item_name in os.listdir(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        item_path = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, item_name)

        if os.path.isdir(item_path):
            base_name = item_name # directory name is the base_name
            print(f"\n--- Processing transcriptions for: {base_name} ---")

            normal_text_path = os.path.join(item_path, f"{base_name}_normal.txt")
            srt_path = os.path.join(item_path, f"{base_name}.srt")

            normal_text_content = None
            if os.path.exists(normal_text_path):
                try:
                    with open(normal_text_path, 'r', encoding='utf-8') as f:
                        normal_text_content = f.read()
                    print(f"Successfully read normal text for {base_name} ({len(normal_text_content.splitlines())} lines).")
                except Exception as e:
                    print(f"Error reading normal text file {normal_text_path}: {e}")
                    continue
            else:
                print(f"Warning: Normal text file not found: {normal_text_path}. Skipping {base_name}.")
                continue

            srt_content_str = None # Renamed to avoid conflict with srt_content dict later
            if os.path.exists(srt_path):
                try:
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        srt_content_str = f.read()
                    print(f"Successfully read SRT file for {base_name} ({len(srt_content_str.splitlines())} lines).")
                except Exception as e:
                    print(f"Error reading SRT file {srt_path}: {e}")
                    print(f"Skipping {base_name} due to error reading SRT file.")
                    continue
            else:
                print(f"Warning: SRT file not found: {srt_path}. Skipping {base_name} as SRT is needed for '時間軸'.")
                continue

            # --- Google Spreadsheet Operations ---
            spreadsheet = None
            spreadsheet_name = base_name # Use base_name (original audio filename without ext) as Spreadsheet title
            try:
                print(f"Attempting to open or create Google Spreadsheet: '{spreadsheet_name}'")
                try:
                    spreadsheet = gc.open(spreadsheet_name)
                    print(f"Opened existing spreadsheet. URL: {spreadsheet.url}")
                except gspread.exceptions.SpreadsheetNotFound:
                    spreadsheet = gc.create(spreadsheet_name)
                    print(f"Created new spreadsheet. URL: {spreadsheet.url}")
                    # Optional: Share the newly created spreadsheet if needed
                    # spreadsheet.share('your-email@example.com', perm_type='user', role='writer')

                # --- "文本校對" Sheet Handling ---
                normal_worksheet_title = "文本校對"
                try:
                    normal_worksheet = spreadsheet.worksheet(normal_worksheet_title)
                    print(f"Found existing worksheet: '{normal_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    normal_worksheet = spreadsheet.add_worksheet(title=normal_worksheet_title, rows="100", cols="20") # Adjust rows/cols
                    print(f"Created new worksheet: '{normal_worksheet_title}'")

                normal_worksheet.clear()
                header_normal = ["Whisper"]
                lines_to_upload_normal = [[line] for line in normal_text_content.splitlines()]
                data_for_normal_sheet = [header_normal] + lines_to_upload_normal
                normal_worksheet.update(range_name='A1', values=data_for_normal_sheet)
                print(f"Uploaded data to '{normal_worksheet_title}' sheet.")

                # --- "時間軸" Sheet Handling ---
                subtitle_worksheet_title = "時間軸"
                try:
                    subtitle_worksheet = spreadsheet.worksheet(subtitle_worksheet_title)
                    print(f"Found existing worksheet: '{subtitle_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    subtitle_worksheet = spreadsheet.add_worksheet(title=subtitle_worksheet_title, rows="100", cols="20") # Adjust
                    print(f"Created new worksheet: '{subtitle_worksheet_title}'")

                subtitle_worksheet.clear()
                parsed_srt_segments = parse_srt_content(srt_content_str)
                header_subtitle = ['序號', '開始時間', '結束時間', '文字']
                rows_to_upload_subtitle = [[seg['id'], seg['start'], seg['end'], seg['text']] for seg in parsed_srt_segments]
                data_for_subtitle_sheet = [header_subtitle] + rows_to_upload_subtitle
                subtitle_worksheet.update(range_name='A1', values=data_for_subtitle_sheet)
                print(f"Uploaded data to '{subtitle_worksheet_title}' sheet.")

                processed_item_count += 1
                print(f"Successfully processed and uploaded data for {base_name} to Google Sheets.")

                # --- Gemini API Processing ---
                whisper_lines_for_gemini = normal_text_content.splitlines()
                if whisper_lines_for_gemini and (pdf_context_text or not pdf_context_text): # Process even if no PDF context, Gemini might still improve
                    print(f"Calling Gemini API for {base_name}...")
                    if not pdf_context_text:
                        print("Note: PDF context is empty. Gemini will process without handout reference.")

                    corrected_text_str = get_gemini_correction(whisper_lines_for_gemini, pdf_context_text if pdf_context_text else "") # Pass empty string if None

                    if corrected_text_str:
                        gemini_lines = corrected_text_str.strip().split('\n')
                        data_for_gemini_column = [["Gemini"]] + [[line] for line in gemini_lines]
                        try:
                            normal_worksheet.update(range_name='B1', values=data_for_gemini_column)
                            print(f"Successfully uploaded Gemini corrections to Column B for {base_name}.")
                        except Exception as e:
                            print(f"Error updating Column B with Gemini output for {base_name}: {e}")
                    else:
                        print(f"Gemini processing returned no output or failed for {base_name}. Column B will be empty.")
                else:
                    print(f"Skipping Gemini API call for {base_name} due to no input text lines from Whisper.")

                display(HTML(f"<p>Finished processing for {base_name}. View Spreadsheet: <a href='{spreadsheet.url}' target='_blank'>{spreadsheet.url}</a></p>"))

            except Exception as e:
                print(f"Error during Google Sheets operations or Gemini processing for {base_name}: {e}")
                # Consider logging the error more formally if logging module is integrated
                continue # Skip to next base_name

    if processed_item_count == 0:
        print(f"No valid transcription subdirectories found or processed in {TRANSCRIPTIONS_ROOT_INPUT_DIR}.")
    else:
        print(f"\nFinished processing {processed_item_count} item(s).")

# --- Script Execution ---
if __name__ == '__main__':
    # Authentication and Drive mount happen globally at script start in Colab.
    # PDF context is also loaded globally.

    # The old main loop is gone. We call the new top-level function.
    process_transcriptions_and_apply_gemini()

    print("\nSheet and Gemini processing script finished (or reached end of placeholder logic).")

# Note: The `time.sleep(60)` from the end of the old main loop is removed as that loop is gone.
# Any necessary delays for API rate limits are handled within get_gemini_correction.
