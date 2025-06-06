import os
from google.colab import drive, auth, userdata
# from faster_whisper import WhisperModel # 已移除
import datetime # 保留，用於 PDF 日期邏輯 (如果有的話) 或通用工具
import gspread
from google.auth import default
import logging # 為日誌記錄添加
import textwrap
import pypdf
import requests
import json
import time
import re # 為 SRT 解析添加
import glob # For PDF cleanup
from IPython.display import HTML # <-- 修正：導入 HTML
import warnings # 導入 warnings 模組

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置區塊 ---
TRANSCRIPTIONS_ROOT_INPUT_DIR = "/content/drive/MyDrive/output_transcriptions" # 重命名：此腳本的輸入目錄
pdf_handout_dir = "/content/drive/MyDrive/lecture_handouts" # 保留，用於 Gemini 上下文
GEMINI_STATE_FILE_PATH = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, ".gemini_processed_state.json") # Gemini 處理狀態檔案路徑

# --- 默認 Gemini API 提示詞常量 ---
DEFAULT_GEMINI_MAIN_INSTRUCTION = (
    "你是一個佛學大師，精通經律論三藏十二部經典。\n"
    "以下文本是whisper產生的字幕文本，關於觀無量壽經、善導大師觀經四帖疏、傳通記的內容。\n"
    "有很多聽打錯誤，幫我依據我提供的上課講義校對文本，嚴格依照以下規則，直接修正錯誤："
)

DEFAULT_GEMINI_CORRECTION_RULES = (
    "校對規則：\n"
    "    1. 這是講座字幕的文本。請逐行處理提供的「字幕文本」。\n"
    "    2. **嚴格依照原本的斷句輸出，保持每一行的獨立性，不要合併或拆分行。輸出結果必須與輸入的行數完全相同 (共 {len(transcribed_text_lines)} 行)。**\n"
    "    3. 如果某一行不需要修改，請直接輸出原始該行內容。\n"
    "    4. 根據「上課講義內容」修正「字幕文本」中的任何聽打錯誤或不準確之處。\n"
    "    5. 不要加標點符號。\n"
    "    6. 輸出繁體中文。"
)

# --- 輔助函式：從 PDF 資料夾提取所有文本 ---
def extract_text_from_pdf_dir(pdf_dir):
    # 此函數使用 print 輸出其進度，除非特別指定更改內部 print，否則將保持原樣
    full_text = []
    if not os.path.exists(pdf_dir):
        logging.error(f"PDF 講義資料夾 '{pdf_dir}' 不存在。")
        return None

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"資料夾 '{pdf_dir}' 中沒有找到任何 PDF 檔案。")
        return None

    logging.info(f"正在從資料夾 '{pdf_dir}' 中的 {len(pdf_files)} 個 PDF 檔案提取文本...")
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file_name)
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    full_text.append(reader.pages[page_num].extract_text())
            logging.info(f"  - 成功提取 '{pdf_file_name}'。")
        except Exception as e:
            logging.error(f"  - 從 '{pdf_file_name}' 提取文本時發生錯誤: {e}", exc_info=True)
            # 即使一個 PDF 失敗，也嘗試繼續處理其他 PDF

    combined_text = "\n".join(full_text)
    if combined_text:
        logging.info(f"所有 PDF 提取完成，共 {len(combined_text)} 字元文本。")
    else:
        logging.warning("未能從任何 PDF 檔案提取到文本。")
    return combined_text

# --- Gemini 處理狀態持久化函數 ---
def load_gemini_processed_state(state_file_path):
    try:
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data if isinstance(data, list) else [])
        logging.info(f"Gemini 狀態檔案 '{state_file_path}' 未找到。將處理所有項目。")
    except json.JSONDecodeError:
        logging.warning(f"解碼 Gemini 狀態檔案 '{state_file_path}' 時發生錯誤。將重新處理所有項目。")
    except Exception as e:
        logging.error(f"載入 Gemini 狀態檔案 '{state_file_path}' 時發生錯誤: {e}。將重新處理所有項目。", exc_info=True)
    return set()

def save_gemini_processed_state(state_file_path, processed_items_set):
    temp_state_file_path = state_file_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        with open(temp_state_file_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_items_set), f, ensure_ascii=False, indent=4)
        os.replace(temp_state_file_path, state_file_path)
        logging.debug(f"Gemini 處理狀態已成功儲存至 '{state_file_path}'。")
    except Exception as e:
        logging.error(f"儲存 Gemini 處理狀態至 '{state_file_path}' 時發生錯誤: {e}", exc_info=True)
        if os.path.exists(temp_state_file_path):
            try:
                os.remove(temp_state_file_path)
            except OSError as oe:
                logging.error(f"移除臨時 Gemini 狀態檔案 '{temp_state_file_path}' 時發生錯誤: {oe}", exc_info=True)

# --- 輔助函數：解析 SRT 內容 ---
def parse_srt_content(srt_content_str):
    segments = []
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
        r"((?:.+\n?)+)",
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
# 修改此函數以接受 main_instruction 和 correction_rules
def get_gemini_correction(transcribed_text_lines, pdf_context, main_instruction, correction_rules):
    api_key = userdata.get('GEMINI_API_KEY')

    if not api_key:
        print("錯誤: GEMINI_API_KEY 未設定。請在 Colab Secrets 中設定您的 Gemini API 金鑰。")
        logging.critical("GEMINI_API_KEY 未設定。")
        return None

    headers = {
        'Content-Type': 'application/json',
    }

    transcribed_text_single_string = "\n".join(transcribed_text_lines)

    # 動態格式化校對規則中的行數信息
    formatted_correction_rules = correction_rules.format(len_transcribed_text_lines=len(transcribed_text_lines))

    # 使用傳入的指令和規則構建 full_prompt
    full_prompt = f"""{main_instruction}

上課講義內容（作為校對參考，請仔細閱讀）：
---
{pdf_context}
---

以下是需要校對的字幕文本 (共 {len(transcribed_text_lines)} 行):
---
{transcribed_text_single_string}
---

{formatted_correction_rules}
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
    base_delay = 60  # 秒
    response = None
    print("正在調用 Gemini API 進行文本校對，這可能需要一些時間...")
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
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
                    logging.error(f"Gemini API 達到最大重試次數 (429)。錯誤: {e}", exc_info=True)
                    return None
            else:
                print(f"調用 Gemini API 時發生錯誤: {e}")
                logging.error(f"調用 Gemini API 時發生錯誤: {e}", exc_info=True)
                return None

    if response is None or not response.ok:
        print(f"Gemini API 調用最終失敗。")
        logging.error("Gemini API 調用最終失敗。")
        return None

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
                logging.warning(f"Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。")
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
            logging.error(f"Gemini API 響應結構異常或內容缺失: {result}")
            return None
    except json.JSONDecodeError as e:
        response_text = response.text if response else "No response text available"
        print(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}")
        logging.error(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}", exc_info=True)
        return None
    except Exception as e:
        print(f"處理 Gemini API 響應時發生未知錯誤: {e}")
        logging.error(f"處理 Gemini API 響應時發生未知錯誤: {e}", exc_info=True)
        return None

gc = None
pdf_context_text = ""
current_main_instruction = DEFAULT_GEMINI_MAIN_INSTRUCTION # Initialize with default
current_correction_rules = DEFAULT_GEMINI_CORRECTION_RULES # Initialize with default

def initial_setup():
    global gc, pdf_context_text, current_main_instruction, current_correction_rules

    logger = logging.getLogger('SheetsGeminiProcessorLogger') # Use the same logger name as in main execution block

    logging.info("正在進行 Google Drive 和 Sheets 身份驗證...")
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        logging.info("Google Drive 和 Sheets 身份驗證成功。")
    except Exception as e:
        logging.error(f"Google Drive 或 Sheets 身份驗證失敗: {e}", exc_info=True)
        gc = None # Ensure gc is None if auth fails

    if gc: # Only proceed if auth was successful
        logging.info("正在掛載 Google Drive...")
        try:
            drive.mount('/content/drive', force_remount=True)
            logging.info("Google Drive 掛載成功。")
        except Exception as e:
            logging.error(f"Google Drive 掛載失敗: {e}", exc_info=True)

    # --- Gemini API 提示詞設定 ---
    logger.info("\n--- Gemini API 提示詞設定 ---")
    logger.info("當前默認的主要指令 (main instruction):")
    for line in DEFAULT_GEMINI_MAIN_INSTRUCTION.split('\n'):
        logger.info(f"  {line}")

    print("\n" + "="*30) # Use print for direct user interaction in Colab for input
    user_main_instruction_input = input(f"請輸入新的主要指令 (可直接粘貼多行文本)，或直接按 Enter 使用默認指令:\n")
    print("="*30)
    current_main_instruction = user_main_instruction_input.strip() if user_main_instruction_input.strip() else DEFAULT_GEMINI_MAIN_INSTRUCTION

    logger.info(f"Gemini API 將使用以下主要指令:")
    for line in current_main_instruction.split('\n'):
        logger.info(f"  {line}")

    logger.info("\n當前默認的校對規則 (correction rules):")
    # Displaying the template rule with the placeholder for the user
    temp_formatted_rules_for_display = DEFAULT_GEMINI_CORRECTION_RULES.replace("{len(transcribed_text_lines)}", "<行數>")
    for line in temp_formatted_rules_for_display.split('\n'):
        logger.info(f"  {line}")

    print("\n" + "="*30)
    print("請輸入新的校對規則。您可以直接粘貼多行文本。")
    print("如果您的規則中需要引用轉錄文本的總行數，請使用占位符 '{len_transcribed_text_lines}' (不含引號)。")
    print("如果保留空白並直接按 Enter，將使用上述默認規則。")
    print("="*30)
    user_correction_rules_input = input("粘貼您的校對規則於此:\n")
    current_correction_rules = user_correction_rules_input.strip() if user_correction_rules_input.strip() else DEFAULT_GEMINI_CORRECTION_RULES

    logger.info(f"Gemini API 將使用以下校對規則 (占位符将在运行时替换):")
    temp_formatted_rules_for_display_user = current_correction_rules.replace("{len_transcribed_text_lines}", "<行數>")
    for line in temp_formatted_rules_for_display_user.split('\n'):
        logger.info(f"  {line}")

    # --- PDF 講義文件夾清理 ---
    logging.info(f"準備清理 PDF 講義文件夾: '{pdf_handout_dir}'...")
    if os.path.exists(pdf_handout_dir):
        pdf_files_to_delete = []
        pdf_files_to_delete.extend(glob.glob(os.path.join(pdf_handout_dir, "*.pdf")))
        pdf_files_to_delete.extend(glob.glob(os.path.join(pdf_handout_dir, "*.PDF")))

        if not pdf_files_to_delete:
            logging.info("PDF 講義文件夾中沒有找到 PDF 文件，無需清理。")
        else:
            deleted_count = 0
            for pdf_file in pdf_files_to_delete:
                try:
                    os.remove(pdf_file)
                    logging.debug(f"已刪除舊 PDF 文件: {pdf_file}")
                    deleted_count += 1
                except Exception as e_remove:
                    logging.error(f"刪除 PDF 文件 '{pdf_file}' 時發生錯誤: {e_remove}", exc_info=True)
            logging.info(f"PDF 講義文件夾清理完畢。共刪除了 {deleted_count} 個 PDF 文件。")
    else:
        logging.warning(f"PDF 講義文件夾 '{pdf_handout_dir}' 不存在，跳過清理步驟。")

    # --- PDF Upload UI ---
    logging.info(f"準備 PDF 上傳界面，目標文件夾: '{pdf_handout_dir}'")
    try:
        from google.colab import files
        if not os.path.exists(pdf_handout_dir):
            logging.info(f"PDF 講義文件夾 '{pdf_handout_dir}' 不存在，正在創建...")
            os.makedirs(pdf_handout_dir, exist_ok=True)
            logging.info(f"PDF 講義文件夾 '{pdf_handout_dir}' 創建成功。")

        print(f"請選擇要上傳到 '{pdf_handout_dir}' 的 PDF 講義文件：")
        uploaded_files = files.upload()

        if not uploaded_files:
            logging.info("沒有選擇任何 PDF 文件進行上傳。")
        else:
            for file_name, file_content in uploaded_files.items():
                destination_path = os.path.join(pdf_handout_dir, file_name)
                try:
                    with open(destination_path, 'wb') as f:
                        f.write(file_content)
                    logging.info(f"文件 '{file_name}' 已成功上傳至 '{destination_path}'。")
                except Exception as e_upload:
                    logging.error(f"儲存上傳的 PDF 文件 '{file_name}' 至 '{destination_path}' 時發生錯誤: {e_upload}", exc_info=True)
            logging.info(f"共 {len(uploaded_files)} 個 PDF 文件上傳操作完成。")

    except ImportError:
        logging.warning("`google.colab.files` 模塊無法導入。PDF 上傳功能僅在 Google Colab 環境中可用。")
    except Exception as e_colab_files:
        logging.error(f"使用 `google.colab.files.upload()` 進行 PDF 上傳時發生錯誤: {e_colab_files}", exc_info=True)

    logging.info(f"正在從 '{pdf_handout_dir}' 提取 PDF 講義內容...")
    pdf_context_text_local = extract_text_from_pdf_dir(pdf_handout_dir)
    if pdf_context_text_local is None or not pdf_context_text_local.strip():
        logging.warning(f"未能從資料夾 '{pdf_handout_dir}' 提取到有效文本，或資料夾不存在/為空。Gemini 校對將不使用講義參考。")
        pdf_context_text = ""
    else:
        pdf_context_text = pdf_context_text_local
        logging.info("PDF 講義內容提取完成。")

    # Return all necessary global-like variables that were set up
    return gc, pdf_context_text, current_main_instruction, current_correction_rules


def process_transcriptions_and_apply_gemini(current_main_instruction, current_correction_rules): # Added parameters
    global gc, pdf_context_text
    if gc is None:
        logging.error("gspread client (gc) 未初始化。身份驗證可能失敗。")
        return

    gemini_processed_items = load_gemini_processed_state(GEMINI_STATE_FILE_PATH)
    logging.info(f"已載入 {len(gemini_processed_items)} 個已完成 Gemini 校對的項目記錄。")

    logging.info(f"開始掃描輸入目錄: {TRANSCRIPTIONS_ROOT_INPUT_DIR}")
    if not os.path.exists(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        logging.error(f"轉錄輸入目錄 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 未找到。請確保 local_transcriber.py 已運行並生成輸出。")
        return

    processed_item_count = 0
    for item_name in os.listdir(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        item_path = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, item_name)

        if os.path.isdir(item_path):
            base_name = item_name
            logging.info(f"--- 開始處理項目: {base_name} ---")

            normal_text_path = os.path.join(item_path, f"{base_name}_normal.txt")
            srt_path = os.path.join(item_path, f"{base_name}.srt")

            normal_text_content = None
            if os.path.exists(normal_text_path):
                try:
                    with open(normal_text_path, 'r', encoding='utf-8') as f:
                        normal_text_content = f.read()
                    logging.info(f"成功讀取一般文本檔案: {normal_text_path} ({len(normal_text_content.splitlines())} 行)。")
                except Exception as e:
                    logging.error(f"讀取一般文本檔案失敗: {normal_text_path} - {e}", exc_info=True)
                    continue
            else:
                logging.warning(f"一般文本檔案未找到: {normal_text_path}，跳過 {base_name}。")
                continue

            srt_content_str = None
            if os.path.exists(srt_path):
                try:
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        srt_content_str = f.read()
                    logging.info(f"成功讀取 SRT 字幕檔案: {srt_path} ({len(srt_content_str.splitlines())} 行)。")
                except Exception as e:
                    logging.error(f"讀取 SRT 字幕檔案失敗: {srt_path} - {e}", exc_info=True)
                    logging.warning(f"由於讀取 SRT 字幕檔案失敗，跳過 {base_name}。")
                    continue
            else:
                logging.warning(f"SRT 字幕檔案未找到: {srt_path}，跳過 {base_name} (因需要 SRT 檔案以建立 '時間軸' 工作表)。")
                continue

            spreadsheet = None
            spreadsheet_name = base_name
            try:
                logging.info(f"正在嘗試開啟或創建 Google 試算表: '{spreadsheet_name}'")
                try:
                    spreadsheet = gc.open(spreadsheet_name)
                    logging.info(f"已開啟現有試算表 '{spreadsheet_name}'。URL: {spreadsheet.url}")
                except gspread.exceptions.SpreadsheetNotFound:
                    spreadsheet = gc.create(spreadsheet_name)
                    logging.info(f"已創建新的試算表 '{spreadsheet_name}'。URL: {spreadsheet.url}")

                normal_worksheet_title = "文本校對"
                normal_worksheet = None
                try:
                    normal_worksheet = spreadsheet.worksheet(normal_worksheet_title)
                    logging.info(f"找到現有工作表: '{normal_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    normal_worksheet = spreadsheet.add_worksheet(title=normal_worksheet_title, rows="100", cols="20")
                    logging.info(f"已創建新的工作表: '{normal_worksheet_title}'")

                logging.info(f"正在清除工作表 '{normal_worksheet_title}' 的現有內容...")
                normal_worksheet.clear()
                header_normal = ["Whisper"]
                lines_to_upload_normal = [[line] for line in normal_text_content.splitlines()]
                data_for_normal_sheet = [header_normal] + lines_to_upload_normal
                normal_worksheet.update(range_name='A1', values=data_for_normal_sheet)
                logging.info(f"數據已成功上傳至工作表 '{normal_worksheet_title}'。")

                subtitle_worksheet_title = "時間軸"
                subtitle_worksheet = None
                try:
                    subtitle_worksheet = spreadsheet.worksheet(subtitle_worksheet_title)
                    logging.info(f"找到現有工作表: '{subtitle_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    subtitle_worksheet = spreadsheet.add_worksheet(title=subtitle_worksheet_title, rows="100", cols="20")
                    logging.info(f"已創建新的工作表: '{subtitle_worksheet_title}'")

                logging.info(f"正在清除工作表 '{subtitle_worksheet_title}' 的現有內容...")
                subtitle_worksheet.clear()
                parsed_srt_segments = parse_srt_content(srt_content_str)
                header_subtitle = ['序號', '開始時間', '結束時間', '文字']
                rows_to_upload_subtitle = [[seg['id'], seg['start'], seg['end'], seg['text']] for seg in parsed_srt_segments]
                data_for_subtitle_sheet = [header_subtitle] + rows_to_upload_subtitle
                subtitle_worksheet.update(range_name='A1', values=data_for_subtitle_sheet)
                logging.info(f"數據已成功上傳至工作表 '{subtitle_worksheet_title}'。")

                if base_name in gemini_processed_items:
                    logging.info(f"'{base_name}' 的 Gemini 校對先前已完成，跳過對 API 的調用。")
                elif not normal_text_content.splitlines():
                    logging.info(f"'{base_name}' 的 Whisper 文本為空或無實質內容，跳過 Gemini API 校對。")
                else:
                    logging.info(f"準備對 '{base_name}' 的文本進行 Gemini API 校對...")
                    whisper_lines_for_gemini = normal_text_content.splitlines()

                    if not pdf_context_text:
                        logging.info("注意: PDF 講義內容為空，Gemini 校對將不使用講義參考。")

                    # Use the passed-in prompt components
                    corrected_text_str = get_gemini_correction(
                        whisper_lines_for_gemini,
                        pdf_context_text if pdf_context_text else "",
                        current_main_instruction, # Use configured main instruction
                        current_correction_rules  # Use configured correction rules
                    )

                    if corrected_text_str:
                        gemini_lines = corrected_text_str.strip().split('\n')
                        data_for_gemini_column = [["Gemini"]] + [[line] for line in gemini_lines]
                        try:
                            normal_worksheet.update(range_name='B1', values=data_for_gemini_column)
                            logging.info(f"Gemini API 校對完成 ({len(gemini_lines)} 行)。已成功上傳 Gemini 校對結果至 B欄 ({base_name})。")

                            gemini_processed_items.add(base_name)
                            save_gemini_processed_state(GEMINI_STATE_FILE_PATH, gemini_processed_items)
                            logging.info(f"已將 '{base_name}' 標記為 Gemini 校對完成並更新狀態檔案。")
                        except Exception as e_update:
                            logging.error(f"更新 B欄 Gemini 校對結果時發生錯誤 ({base_name}): {e_update}", exc_info=True)
                    else:
                        logging.warning(f"Gemini API 校對失敗或無返回內容 ({base_name})，B欄將保持空白。將不會標記為 Gemini 校對完成。")

                processed_item_count += 1
                logging.info(f"項目 {base_name} 的表格處理完成。試算表連結: {spreadsheet.url}")
                display(HTML(f"<p>項目 {base_name} 處理完成。試算表連結: <a href='{spreadsheet.url}' target='_blank'>{spreadsheet.url}</a></p>"))

            except Exception as e_sheet_ops:
                logging.error(f"處理試算表 '{spreadsheet_name}' 時發生錯誤: {e_sheet_ops}", exc_info=True)
                continue

    if processed_item_count == 0:
        logging.info(f"在 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 目錄中未找到任何有效的轉錄項目進行處理。")
    else:
        logging.info(f"總共處理了 {processed_item_count} 個項目。")

# --- 腳本執行 ---
if __name__ == '__main__':
    # 日誌記錄器在此處配置，因此 initial_setup 和 process_transcriptions_and_apply_gemini 中的 logger.info 等將正常工作
    logger = logging.getLogger('SheetsGeminiProcessorLogger') # Specific logger name
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # logger.propagate = False # Optional: prevent propagation to root logger

    logger.info("sheets_gemini_processor.py 腳本已啟動。")

    # initial_setup 現在返回配置的提示詞
    # We need to handle the case where gc might be None if auth fails in initial_setup
    setup_results = initial_setup()
    if setup_results[0] is None: # Check if gc is None
        logger.critical("由於 gspread 客戶端 (gc) 初始化失敗，腳本無法繼續。請檢查認證和授權。")
    else:
        # Unpack all return values from initial_setup
        gc_client, pdf_text, main_instr, correct_rules = setup_results
        # Update global gc and pdf_context_text as they are used by process_transcriptions_and_apply_gemini implicitly
        # Better would be to pass them all as parameters to process_transcriptions_and_apply_gemini
        # For now, let's ensure globals are updated if initial_setup was successful.
        # gc = gc_client # gc is already global and set within initial_setup
        # pdf_context_text = pdf_text # pdf_context_text is already global and set

        # Pass the configured prompts to the main processing function
        process_transcriptions_and_apply_gemini(main_instr, correct_rules)

    logger.info("sheets_gemini_processor.py 腳本已完成。")
