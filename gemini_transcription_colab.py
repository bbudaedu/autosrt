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
from faster_whisper import WhisperModel
import datetime
import gspread
from google.auth import default
import textwrap
import pypdf
import requests
import json
import time
from IPython.display import HTML # <-- 修正：導入 HTML
import warnings # 導入 warnings 模組

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置區塊 ---
model_size = "large-v3"
initial_prompt_text = "這是佛教關於密教真言宗藥師佛"

input_audio_dir = "/content/drive/MyDrive/input_audio"
output_transcriptions_dir = "/content/drive/MyDrive/output_transcriptions"
pdf_handout_dir = "/content/drive/MyDrive/lecture_handouts"

# --- 輔助函式：格式化SRT時間 ---
def format_srt_time(seconds):
    """將秒數格式化為 SRT 格式的時間字串 (HH:MM:SS,ms)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

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

    max_retries = 5
    base_delay = 10  # seconds
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

# --- 檔案處理區塊 (從 Google Drive 讀取檔案) ---
if not os.path.exists(input_audio_dir):
    print(f"錯誤: Google Drive 中的輸入資料夾 '{input_audio_dir}' 不存在。請在您的 Google Drive 中建立此資料夾並放入音檔。")
    exit()

os.makedirs(output_transcriptions_dir, exist_ok=True)
print(f"輸出資料夾已設定為: {output_transcriptions_dir}")

audio_files_to_process = [f for f in os.listdir(input_audio_dir) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.mp4'))]

if not audio_files_to_process:
    print(f"在 '{input_audio_dir}' 資料夾中沒有找到任何支援的音檔。請將音檔放入此資料夾。")
    exit()

print(f"\n在 '{input_audio_dir}' 中找到 {len(audio_files_to_process)} 個檔案準備處理。")

# --- 模型載入 ---
print(f"正在載入 Faster Whisper 模型: {model_size}...")
model = WhisperModel(model_size, device="cuda", compute_type="float16")
print("模型載入完成。")

# 批次處理每個音檔
for audio_file_name in audio_files_to_process:
    audio_path = os.path.join(input_audio_dir, audio_file_name)
    file_name_without_ext = os.path.splitext(audio_file_name)[0]

    spreadsheet_name = file_name_without_ext

    print(f"\n--- 正在處理檔案: {audio_file_name} ---")

    spreadsheet = None
    try:
        spreadsheet = gc.open(spreadsheet_name)
        print(f"已開啟現有 Google 試算表: {spreadsheet_name}")
    except gspread.exceptions.SpreadsheetNotFound:
        spreadsheet = gc.create(spreadsheet_name)
        print(f"已建立新的 Google 試算表: {spreadsheet_name}")
        print(f"試算表連結: {spreadsheet.url}")
    except Exception as e:
        print(f"開啟或建立 Google 試算表 '{spreadsheet_name}' 時發生錯誤: {e}")
        continue

    try:
        normal_worksheet = spreadsheet.worksheet("文本校對")
        print("已找到分頁: 文本校對")
    except gspread.exceptions.WorksheetNotFound:
        normal_worksheet = spreadsheet.add_worksheet(title="文本校對", rows="100", cols="20")
        print("已建立分頁: 文本校對")

    try:
        subtitle_worksheet = spreadsheet.worksheet("時間軸")
        print("已找到分頁: 時間軸")
    except gspread.exceptions.WorksheetNotFound:
        subtitle_worksheet = spreadsheet.add_worksheet(title="時間軸", rows="100", cols="20")
        print("已建立分頁: 時間軸")

    print("正在清除現有分頁內容...")
    normal_worksheet.clear()
    subtitle_worksheet.clear()
    print("分頁內容已清除。")

    try:
        print("正在進行語音轉錄，這可能需要一些時間...")
        vad_parameters = {
            "min_speech_duration_ms": 50,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 500,
        }
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=5,
            initial_prompt=initial_prompt_text,
            vad_filter=True,
            vad_parameters=vad_parameters
        )
        segments_list = list(segments_generator)

        print(f"轉錄完成。語言: {info.language}, 語言機率: {info.language_probability:.2f}")
        print(f"轉錄生成了 {len(segments_list)} 個語音片段。")

        unwanted_phrase = "字幕由 Amara.org 社群提供"
        whisper_transcription_lines = []
        for segment in segments_list:
            cleaned_text = segment.text.strip().replace(unwanted_phrase, "").strip()
            if cleaned_text:
                whisper_transcription_lines.append(cleaned_text)

        whisper_transcription_full = "\n".join(whisper_transcription_lines)

        # Prepare data for normal_worksheet (A column for Whisper, B column for Gemini)
        # Ensure whisper_lines_for_sheet is always populated for A column
        whisper_lines_for_sheet = [[line] for line in whisper_transcription_lines]
        gemini_lines_for_sheet = [] # Will be filled if Gemini succeeds

        corrected_text = None
        if whisper_transcription_lines and pdf_context_text:
            corrected_text = get_gemini_correction(whisper_transcription_lines, pdf_context_text)
            if corrected_text:
                gemini_lines_for_sheet = [[line] for line in corrected_text.strip().split('\n')]
                print(f"Gemini 校對完成。")
            else:
                print("Gemini 校對失敗或無返回內容。B 欄將留空。")
        elif whisper_transcription_lines:
            print("未提供 PDF 講義內容，Gemini 校對將無法參考講義。B 欄將留空。")
        else:
            print("無 Whisper 轉錄內容，跳過 Gemini 校對。")

        # Combine Whisper and Gemini data for updating normal_worksheet
        max_lines = max(len(whisper_lines_for_sheet), len(gemini_lines_for_sheet))
        combined_data_for_normal_sheet = []
        for i in range(max_lines):
            whisper_line = whisper_lines_for_sheet[i][0] if i < len(whisper_lines_for_sheet) else ""
            gemini_line = gemini_lines_for_sheet[i][0] if i < len(gemini_lines_for_sheet) else ""
            combined_data_for_normal_sheet.append([whisper_line, gemini_line])

        # Clear and update the '文本校對' worksheet
        normal_worksheet.clear()
        if combined_data_for_normal_sheet:
            normal_worksheet.update(range_name='A1', values=combined_data_for_normal_sheet) # <-- 修正：使用命名參數
            print("「文本校對」分頁已更新 (A欄為Whisper轉錄，B欄為Gemini校對)。")
        else:
            normal_worksheet.update(range_name='A1', values=[['無轉錄內容']])
            print("「文本校對」分頁更新完成 (無轉錄內容)。")

        subtitle_transcription_local = ""
        subtitle_data_for_sheet = [['序號', '開始時間', '結束時間', '文字']]

        for i, segment in enumerate(segments_list, 1):
            start_time_srt = format_srt_time(segment.start)
            end_time_srt = format_srt_time(segment.end)

            cleaned_segment_text = segment.text.strip().replace(unwanted_phrase, "").strip()

            if cleaned_segment_text:
                subtitle_data_for_sheet.append([
                    i,
                    start_time_srt,
                    end_time_srt,
                    cleaned_segment_text
                ])

                subtitle_transcription_local += f"{i}\n"
                subtitle_transcription_local += f"{start_time_srt} --> {end_time_srt}\n"
                subtitle_transcription_local += f"{cleaned_segment_text}\n\n"

        subtitle_output_path = os.path.join(output_transcriptions_dir, f"{file_name_without_ext}_subtitle.srt")
        with open(subtitle_output_path, "w", encoding="utf-8") as f:
            f.write(subtitle_transcription_local)
        print(f"字幕檔 (SRT) 已儲存至 Google Drive: {subtitle_output_path}")

        if subtitle_data_for_sheet and len(subtitle_data_for_sheet) > 1:
            subtitle_worksheet.update(range_name='A1', values=subtitle_data_for_sheet)
            print(f"「時間軸」分頁已更新。")
        else:
            print("沒有字幕資料可寫入「時間軸」分頁 (可能轉錄結果為空或被過濾)。")

        print(f"\n'{audio_file_name}' 的 Google 試算表連結: {spreadsheet.url}")
        display(HTML(f"<p>點擊這裡開啟 '{audio_file_name}' 的 Google 試算表：<a href='{spreadsheet.url}' target='_blank'>{spreadsheet.url}</a></p>"))

    except Exception as e:
        print(f"處理檔案 '{audio_file_name}' 時發生錯誤: {e}")

    time.sleep(5)

print("\n所有檔案處理完畢。")
