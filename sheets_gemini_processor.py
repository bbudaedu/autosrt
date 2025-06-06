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
import glob # 用於 PDF 清理
from IPython.display import HTML # <-- 修正：導入 HTML
import warnings # 導入 warnings 模組

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置區塊 ---
TRANSCRIPTIONS_ROOT_INPUT_DIR = "/content/drive/MyDrive/output_transcriptions" # 重命名：此腳本的輸入目錄
pdf_handout_dir = "/content/drive/MyDrive/lecture_handouts" # 保留，用於 Gemini 上下文
GEMINI_STATE_FILE_PATH = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, ".gemini_processed_state.json") # Gemini 處理狀態檔案路徑
INTER_SPREADSHEET_DELAY_SECONDS = 15 # 秒，處理不同表格間的延遲
GEMINI_API_BATCH_MAX_LINES = 100  # 每批次發送給 Gemini API 的最大行數

# --- 默認 Gemini API 提示詞常量 ---
DEFAULT_GEMINI_MAIN_INSTRUCTION = (
    "你是一個佛學大師，精通經律論三藏十二部經典。\n"
    "以下文本是whisper產生的字幕文本，關於觀無量壽經、善導大師觀經四帖疏、傳通記的內容。\n"
    "有很多聽打錯誤，幫我依據我提供的上課講義校對文本，嚴格依照以下規則，直接修正錯誤："
)

DEFAULT_GEMINI_CORRECTION_RULES = (
    "校對規則：\n"
    "    1. 這是講座字幕的文本。請逐行處理提供的「字幕文本」。\n"
    "    2. **嚴格依照原本的斷句輸出，保持每一行的獨立性，不要合併或拆分行。輸出結果必須與輸入的行數完全相同 (共 {batch_line_count} 行)。**\n"
    "    3. 如果某一行不需要修改，請直接輸出原始該行內容。\n"
    "    4. 根據「上課講義內容」修正「字幕文本」中的任何聽打錯誤或不準確之處。\n"
    "    5. 不要加標點符號。\n"
    "    6. 輸出繁體中文。"
)

# --- Google Sheets API 輔助函數 ---
def execute_gspread_write(logger, gspread_operation_func, *args, max_retries=5, base_delay_seconds=5, **kwargs):
    """
    執行 gspread 寫入操作，並為 API 錯誤 (特別是 429) 提供指數退避重試機制。
    logger: 用於日誌記錄的 logging 實例。
    gspread_operation_func: 要調用的 gspread 方法 (例如 worksheet.clear, worksheet.update)。
    *args, **kwargs: 傳遞給 gspread_operation_func 的參數。
    """
    for attempt in range(max_retries):
        try:
            return gspread_operation_func(*args, **kwargs) # 執行操作
        except gspread.exceptions.APIError as e:
            # 檢查錯誤響應和狀態碼是否存在
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429: # HTTP 429: Too Many Requests (請求過多)
                if attempt < max_retries - 1:
                    delay = base_delay_seconds * (2 ** attempt)
                    logger.warning(f"Google Sheets API 速率限制 (429)。將在 {delay} 秒後重試... (嘗試 {attempt + 1}/{max_retries}) 操作: {gspread_operation_func.__name__}")
                    time.sleep(delay)
                else:
                    logger.error(f"Google Sheets API 達到最大重試次數 (429) 操作: {gspread_operation_func.__name__}。錯誤: {e}", exc_info=True)
                    raise # 如果達到最大重試次數，則重新引發異常
            else: # 對於其他 APIError，立即重新引發
                logger.error(f"Google Sheets API 操作 {gspread_operation_func.__name__} 時發生非預期的 API 錯誤 (非429): {e}", exc_info=True)
                raise
        except Exception as e: # 捕獲調用期間其他潛在的非 gspread 錯誤
            logger.error(f"執行 gspread 操作 {gspread_operation_func.__name__} 時發生未知錯誤: {e}", exc_info=True)
            raise
    return None # 理論上，如果最終失敗總是引發異常，則不應到達此處


# --- 輔助函式：從 PDF 資料夾提取所有文本 ---
def extract_text_from_pdf_dir(pdf_dir):
    # 此函數的日誌記錄已在先前步驟中本地化為中文。
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

    combined_text = "\n".join(full_text)
    if combined_text:
        logging.info(f"所有 PDF 提取完成，共 {len(combined_text)} 字元文本。")
    else:
        logging.warning("未能從任何 PDF 檔案提取到文本。")
    return combined_text

# --- Gemini 處理狀態持久化函數 ---
def load_gemini_processed_state(state_file_path):
    # 從狀態檔案載入已完成 Gemini 校對的項目集合
    try:
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data if isinstance(data, list) else []) # 確保是列表，轉換為集合以便高效查找
        logging.info(f"Gemini 狀態檔案 '{state_file_path}' 未找到。將處理所有項目。")
    except json.JSONDecodeError:
        logging.warning(f"解碼 Gemini 狀態檔案 '{state_file_path}' 時發生錯誤。將重新處理所有項目。")
    except Exception as e:
        logging.error(f"載入 Gemini 狀態檔案 '{state_file_path}' 時發生錯誤: {e}。將重新處理所有項目。", exc_info=True)
    return set()

def save_gemini_processed_state(state_file_path, processed_items_set):
    # 將已完成 Gemini 校對的項目集合儲存到狀態檔案
    temp_state_file_path = state_file_path + ".tmp" # 使用臨時檔案以確保原子性寫入
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True) # 確保目錄存在
        with open(temp_state_file_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_items_set), f, ensure_ascii=False, indent=4) # 將集合轉換為列表以便 JSON 序列化
        os.replace(temp_state_file_path, state_file_path) # 原子性重命名 (在 POSIX 系統上)
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
    # 正則表達式，用於捕獲 ID、開始時間、結束時間和文本
    # 處理單個片段的多行文本
    pattern = re.compile(
        r"(\d+)\s*\n"                       # ID
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"  # 開始和結束時間
        r"((?:.+\n?)+)",                  # 文本 (非貪婪模式，捕獲多行)
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

# --- 輔助函式：調用 Gemini API 進行校對 (含分批處理邏輯) ---
def get_gemini_correction(logger, transcribed_text_lines, pdf_context, main_instruction, correction_rules):
    api_key = userdata.get('GEMINI_API_KEY')
    if not api_key:
        print("錯誤: GEMINI_API_KEY 未設定。請在 Colab Secrets 中設定您的 Gemini API 金鑰。")
        logger.critical("GEMINI_API_KEY 未設定。")
        return None

    headers = {'Content-Type': 'application/json'}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
    max_retries_gemini = 7
    base_delay_gemini = 60

    all_corrected_lines_from_batches = []
    total_lines = len(transcribed_text_lines)

    if total_lines == 0:
        logger.info("沒有需要校對的文本行 (輸入為空)。")
        return ""

    num_batches = (total_lines + GEMINI_API_BATCH_MAX_LINES - 1) // GEMINI_API_BATCH_MAX_LINES
    logger.info(f"文本總行數: {total_lines}。按每批次最多 {GEMINI_API_BATCH_MAX_LINES} 行，將分割成 {num_batches} 個批次進行 Gemini API 校對。")

    for batch_idx in range(num_batches):
        start_index = batch_idx * GEMINI_API_BATCH_MAX_LINES
        end_index = min((batch_idx + 1) * GEMINI_API_BATCH_MAX_LINES, total_lines)
        current_batch_lines = transcribed_text_lines[start_index:end_index]

        if not current_batch_lines:
            logger.warning(f"第 {batch_idx+1}/{num_batches} 批次為空，跳過。")
            continue

        logger.info(f"正在處理第 {batch_idx+1}/{num_batches} 批次的文本 (行 {start_index+1} 到 {end_index}) 進行 Gemini API 校對...")

        batch_transcribed_text_single_string = "\n".join(current_batch_lines)
        try:
            # 使用新的佔位符名稱 'batch_line_count'
            batch_specific_correction_rules = correction_rules.format(batch_line_count=len(current_batch_lines))
        except KeyError as ke:
            logger.error(f"格式化校對規則 (批次 {batch_idx+1}/{num_batches}) 時發生錯誤：佔位符 'batch_line_count' 可能不正確或缺失。規則模板: '{correction_rules}' 錯誤: {ke}", exc_info=True)
            batch_specific_correction_rules = correction_rules
            logger.warning("由於校對規則格式化錯誤，將嘗試使用未格式化的校對規則。")

        full_prompt_for_batch = (
            f"{main_instruction}\n\n"
            f"上課講義內容（作為校對參考，請仔細閱讀）：\n---\n{pdf_context}\n---\n\n"
            f"以下是需要校對的字幕文本 (共 {len(current_batch_lines)} 行):\n---\n{batch_transcribed_text_single_string}\n---\n\n"
            f"{batch_specific_correction_rules}"
        )

        payload_for_batch = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt_for_batch}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.95, "topK": 64, "maxOutputTokens": 8192, "responseMimeType": "text/plain"}
        }

        response_for_batch = None
        batch_call_successful = False
        for attempt in range(max_retries_gemini):
            try:
                logger.debug(f"Gemini API (批次 {batch_idx+1}/{num_batches}) - 嘗試 {attempt + 1}/{max_retries_gemini}...")
                response_for_batch = requests.post(api_url, headers=headers, data=json.dumps(payload_for_batch), timeout=300)
                response_for_batch.raise_for_status()
                batch_call_successful = True
                logger.info(f"Gemini API (批次 {batch_idx+1}/{num_batches}) - 嘗試 {attempt + 1} 成功。")
                break
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    if attempt < max_retries_gemini - 1:
                        delay = base_delay_gemini * (2 ** attempt)
                        logger.warning(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 速率限制 (429)。將在 {delay} 秒後重試... (嘗試 {attempt + 1}/{max_retries_gemini})")
                        print(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 速率限制 (429)。將在 {delay} 秒後重試... (嘗試 {attempt + 1}/{max_retries_gemini})")
                        time.sleep(delay)
                    else:
                        detailed_error_message = (
                            f"Gemini API (批次 {batch_idx+1}/{num_batches}) 調用因達到最大重試次數 ({max_retries_gemini} 次) 而最終失敗 (HTTP 429)。錯誤詳情: {e}\n"
                            f"這可能表示您的 Gemini API 金鑰遇到了較嚴格的配額限制（例如每日請求總量、每分鐘請求數或特定於 'gemini-1.5-pro-latest' 模型的限制）。\n"
                            f"**強烈建議您前往 Google Cloud Console (或您的 API 金鑰管理平台) 檢查 Gemini API 的用量和配額詳情。**"
                        )
                        logger.error(detailed_error_message)
                        print(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 已達最大重試次數 (429)。失敗。")
                        return None
                else:
                    logger.error(f"調用 Gemini API (批次 {batch_idx+1}/{num_batches}) 時發生請求錯誤 (非429): {e}", exc_info=True)
                    print(f"調用 Gemini API (批次 {batch_idx+1}/{num_batches}) 時發生錯誤: {e}")
                    return None

        if not batch_call_successful:
            logger.error(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 調用最終失敗 (所有重試均未成功)。")
            return None

        logger.debug(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 成功，等待15秒...")
        time.sleep(15)

        try:
            result_for_batch = response_for_batch.json()
            if (result_for_batch.get("candidates") and
                result_for_batch["candidates"][0].get("content") and
                result_for_batch["candidates"][0]["content"].get("parts") and
                result_for_batch["candidates"][0]["content"]["parts"][0].get("text")):

                corrected_text_from_api_batch = result_for_batch["candidates"][0]["content"]["parts"][0]["text"]
                corrected_lines_for_this_batch = corrected_text_from_api_batch.strip().split('\n')

                if len(corrected_lines_for_this_batch) == len(current_batch_lines):
                    logger.info(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 校對完成，行數與原始批次文本一致 ({len(corrected_lines_for_this_batch)} 行)。")
                else:
                    logger.warning(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 返回的行數 ({len(corrected_lines_for_this_batch)}) 與原始批次文本行數 ({len(current_batch_lines)}) 不一致。將嘗試調整。")
                    adjusted_batch_lines = []
                    for k_idx in range(len(current_batch_lines)):
                        if k_idx < len(corrected_lines_for_this_batch):
                            adjusted_batch_lines.append(corrected_lines_for_this_batch[k_idx])
                        else:
                            adjusted_batch_lines.append(current_batch_lines[k_idx])
                    corrected_lines_for_this_batch = adjusted_batch_lines[:len(current_batch_lines)]
                    logger.info(f"已嘗試調整批次 {batch_idx+1}/{num_batches} 的行數為 {len(corrected_lines_for_this_batch)} 行。")

                all_corrected_lines_from_batches.extend(corrected_lines_for_this_batch)
            else:
                logger.error(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 響應結構異常或內容缺失: {result_for_batch}")
                print(f"Gemini API (批次 {batch_idx+1}/{num_batches}) 響應結構異常或內容缺失。")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"解析 Gemini API (批次 {batch_idx+1}/{num_batches}) 響應時發生 JSON 解碼錯誤: {e}. 響應文本: {response_for_batch.text if response_for_batch else 'N/A'}", exc_info=True)
            print(f"解析 Gemini API (批次 {batch_idx+1}/{num_batches}) 響應時發生錯誤。")
            return None
        except Exception as e:
            logger.error(f"處理 Gemini API (批次 {batch_idx+1}/{num_batches}) 響應時發生未知錯誤: {e}", exc_info=True)
            print(f"處理 Gemini API (批次 {batch_idx+1}/{num_batches}) 響應時發生未知錯誤。")
            return None

    logger.info("所有批次的 Gemini API 校對請求均已成功完成。")
    return "\n".join(all_corrected_lines_from_batches)

gc = None
pdf_context_text = ""
current_main_instruction = DEFAULT_GEMINI_MAIN_INSTRUCTION
current_correction_rules = DEFAULT_GEMINI_CORRECTION_RULES

def initial_setup(logger_instance):
    global gc, pdf_context_text, current_main_instruction, current_correction_rules

    logger_instance.info("正在進行 Google Drive 和 Sheets 身份驗證...")
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        logger_instance.info("Google Drive 和 Sheets 身份驗證成功。")
    except Exception as e:
        logger_instance.error(f"Google Drive 或 Sheets 身份驗證失敗: {e}", exc_info=True)
        gc = None

    if gc:
        logger_instance.info("正在掛載 Google Drive...")
        try:
            drive.mount('/content/drive', force_remount=True)
            logger_instance.info("Google Drive 掛載成功。")
        except Exception as e:
            logger_instance.error(f"Google Drive 掛載失敗: {e}", exc_info=True)

    logger_instance.info("\n--- Gemini API 提示詞設定 ---")
    logger_instance.info("當前默認的主要指令 (main instruction):")
    for line in DEFAULT_GEMINI_MAIN_INSTRUCTION.split('\n'):
        logger_instance.info(f"  {line}")

    print("\n" + "="*30)
    user_main_instruction_input = input(f"請輸入新的主要指令 (可直接粘貼多行文本)，或直接按 Enter 使用默認指令:\n")
    print("="*30)
    current_main_instruction = user_main_instruction_input.strip() if user_main_instruction_input.strip() else DEFAULT_GEMINI_MAIN_INSTRUCTION

    logger_instance.info(f"Gemini API 將使用以下主要指令:")
    for line in current_main_instruction.split('\n'):
        logger_instance.info(f"  {line}")

    logger_instance.info("\n當前默認的校對規則 (correction rules):")
    temp_formatted_rules_for_display = DEFAULT_GEMINI_CORRECTION_RULES.replace("{batch_line_count}", "<批次行數>")
    for line in temp_formatted_rules_for_display.split('\n'):
        logger_instance.info(f"  {line}")

    print("\n" + "="*30)
    print("請輸入新的校對規則。您可以直接粘貼多行文本。")
    print("如果您的規則中需要引用轉錄文本的總行數，請使用占位符 '{batch_line_count}' (不含引號)。")
    print("如果保留空白並直接按 Enter，將使用上述默認規則。")
    print("="*30)
    user_correction_rules_input = input("粘貼您的校對規則於此:\n")
    current_correction_rules = user_correction_rules_input.strip() if user_correction_rules_input.strip() else DEFAULT_GEMINI_CORRECTION_RULES

    logger_instance.info(f"Gemini API 將使用以下校對規則 (占位符将在运行时替换):")
    temp_formatted_rules_for_display_user = current_correction_rules.replace("{batch_line_count}", "<批次行數>")
    for line in temp_formatted_rules_for_display_user.split('\n'):
        logger_instance.info(f"  {line}")

    logger_instance.info(f"準備清理 PDF 講義文件夾: '{pdf_handout_dir}'...")
    if os.path.exists(pdf_handout_dir):
        pdf_files_to_delete = []
        pdf_files_to_delete.extend(glob.glob(os.path.join(pdf_handout_dir, "*.pdf")))
        pdf_files_to_delete.extend(glob.glob(os.path.join(pdf_handout_dir, "*.PDF")))

        if not pdf_files_to_delete:
            logger_instance.info("PDF 講義文件夾中沒有找到 PDF 文件，無需清理。")
        else:
            deleted_count = 0
            for pdf_file in pdf_files_to_delete:
                try:
                    os.remove(pdf_file)
                    logger_instance.debug(f"已刪除舊 PDF 文件: {pdf_file}")
                    deleted_count += 1
                except Exception as e_remove:
                    logger_instance.error(f"刪除 PDF 文件 '{pdf_file}' 時發生錯誤: {e_remove}", exc_info=True)
            logger_instance.info(f"PDF 講義文件夾清理完畢。共刪除了 {deleted_count} 個 PDF 文件。")
    else:
        logger_instance.warning(f"PDF 講義文件夾 '{pdf_handout_dir}' 不存在，跳過清理步驟。")

    logger_instance.info(f"準備 PDF 上傳界面，目標文件夾: '{pdf_handout_dir}'")
    try:
        from google.colab import files
        if not os.path.exists(pdf_handout_dir):
            logger_instance.info(f"PDF 講義文件夾 '{pdf_handout_dir}' 不存在，正在創建...")
            os.makedirs(pdf_handout_dir, exist_ok=True)
            logger_instance.info(f"PDF 講義文件夾 '{pdf_handout_dir}' 創建成功。")

        print(f"請選擇要上傳到 '{pdf_handout_dir}' 的 PDF 講義文件：")
        uploaded_files = files.upload()

        if not uploaded_files:
            logger_instance.info("沒有選擇任何 PDF 文件進行上傳。")
        else:
            for file_name, file_content in uploaded_files.items():
                destination_path = os.path.join(pdf_handout_dir, file_name)
                try:
                    with open(destination_path, 'wb') as f:
                        f.write(file_content)
                    logger_instance.info(f"文件 '{file_name}' 已成功上傳至 '{destination_path}'。")
                except Exception as e_upload:
                    logger_instance.error(f"儲存上傳的 PDF 文件 '{file_name}' 至 '{destination_path}' 時發生錯誤: {e_upload}", exc_info=True)
            logger_instance.info(f"共 {len(uploaded_files)} 個 PDF 文件上傳操作完成。")

    except ImportError:
        logger_instance.warning("`google.colab.files` 模塊無法導入。PDF 上傳功能僅在 Google Colab 環境中可用。")
    except Exception as e_colab_files:
        logger_instance.error(f"使用 `google.colab.files.upload()` 進行 PDF 上傳時發生錯誤: {e_colab_files}", exc_info=True)

    logger_instance.info(f"正在從 '{pdf_handout_dir}' 提取 PDF 講義內容...")
    pdf_context_text_local = extract_text_from_pdf_dir(pdf_handout_dir)
    if pdf_context_text_local is None or not pdf_context_text_local.strip():
        logger_instance.warning(f"未能從資料夾 '{pdf_handout_dir}' 提取到有效文本，或資料夾不存在/為空。Gemini 校對將不使用講義參考。")
        pdf_context_text = ""
    else:
        pdf_context_text = pdf_context_text_local
        logger_instance.info("PDF 講義內容提取完成。")

    return gc, pdf_context_text, current_main_instruction, current_correction_rules


def process_transcriptions_and_apply_gemini(logger, current_main_instruction_param, current_correction_rules_param):
    global gc, pdf_context_text
    if gc is None:
        logger.error("gspread client (gc) 未初始化。身份驗證可能失敗。")
        return

    gemini_processed_items = load_gemini_processed_state(GEMINI_STATE_FILE_PATH)
    logger.info(f"已載入 {len(gemini_processed_items)} 個已完成 Gemini 校對的項目記錄。")

    logger.info(f"開始掃描輸入目錄: {TRANSCRIPTIONS_ROOT_INPUT_DIR}")
    if not os.path.exists(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        logger.error(f"轉錄輸入目錄 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 未找到。請確保 local_transcriber.py 已運行並生成輸出。")
        return

    processed_item_count = 0
    for item_name in os.listdir(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        item_path = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, item_name)

        if os.path.isdir(item_path):
            base_name = item_name
            logger.info(f"--- 開始處理項目: {base_name} ---")

            normal_text_path = os.path.join(item_path, f"{base_name}_normal.txt")
            srt_path = os.path.join(item_path, f"{base_name}.srt")

            normal_text_content = None
            if os.path.exists(normal_text_path):
                try:
                    with open(normal_text_path, 'r', encoding='utf-8') as f:
                        normal_text_content = f.read()
                    logger.info(f"成功讀取一般文本檔案: {normal_text_path} ({len(normal_text_content.splitlines())} 行)。")
                except Exception as e:
                    logger.error(f"讀取一般文本檔案失敗: {normal_text_path} - {e}", exc_info=True)
                    continue
            else:
                logger.warning(f"一般文本檔案未找到: {normal_text_path}，跳過 {base_name}。")
                continue

            srt_content_str = None
            if os.path.exists(srt_path):
                try:
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        srt_content_str = f.read()
                    logger.info(f"成功讀取 SRT 字幕檔案: {srt_path} ({len(srt_content_str.splitlines())} 行)。")
                except Exception as e:
                    logger.error(f"讀取 SRT 字幕檔案失敗: {srt_path} - {e}", exc_info=True)
                    logger.warning(f"由於讀取 SRT 字幕檔案失敗，跳過 {base_name}。")
                    continue
            else:
                logger.warning(f"SRT 字幕檔案未找到: {srt_path}，跳過 {base_name} (因需要 SRT 檔案以建立 '時間軸' 工作表)。")
                continue

            spreadsheet = None
            spreadsheet_name = base_name
            try:
                logger.info(f"正在嘗試開啟或創建 Google 試算表: '{spreadsheet_name}'")
                try:
                    spreadsheet = gc.open(spreadsheet_name)
                    logger.info(f"已開啟現有試算表 '{spreadsheet_name}'。URL: {spreadsheet.url}")
                except gspread.exceptions.SpreadsheetNotFound:
                    spreadsheet = gc.create(spreadsheet_name)
                    logger.info(f"已創建新的試算表 '{spreadsheet_name}'。URL: {spreadsheet.url}")

                normal_worksheet_title = "文本校對"
                normal_worksheet = None
                try:
                    normal_worksheet = spreadsheet.worksheet(normal_worksheet_title)
                    logger.info(f"找到現有工作表: '{normal_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    normal_worksheet = execute_gspread_write(logger, spreadsheet.add_worksheet, title=normal_worksheet_title, rows="100", cols="20")
                    if normal_worksheet is None: raise Exception(f"創建工作表 '{normal_worksheet_title}' 失敗。")
                    logger.info(f"已創建新的工作表: '{normal_worksheet_title}'")

                logger.info(f"正在清除工作表 '{normal_worksheet_title}' 的現有內容...")
                execute_gspread_write(logger, normal_worksheet.clear)
                header_normal = ["Whisper"]
                lines_to_upload_normal = [[line] for line in normal_text_content.splitlines()]
                data_for_normal_sheet = [header_normal] + lines_to_upload_normal
                execute_gspread_write(logger, normal_worksheet.update, range_name='A1', values=data_for_normal_sheet)
                logger.info(f"數據已成功上傳至工作表 '{normal_worksheet_title}'。")

                subtitle_worksheet_title = "時間軸"
                subtitle_worksheet = None
                try:
                    subtitle_worksheet = spreadsheet.worksheet(subtitle_worksheet_title)
                    logger.info(f"找到現有工作表: '{subtitle_worksheet_title}'")
                except gspread.exceptions.WorksheetNotFound:
                    subtitle_worksheet = execute_gspread_write(logger, spreadsheet.add_worksheet, title=subtitle_worksheet_title, rows="100", cols="20")
                    if subtitle_worksheet is None: raise Exception(f"創建工作表 '{subtitle_worksheet_title}' 失敗。")
                    logger.info(f"已創建新的工作表: '{subtitle_worksheet_title}'")

                logger.info(f"正在清除工作表 '{subtitle_worksheet_title}' 的現有內容...")
                execute_gspread_write(logger, subtitle_worksheet.clear)
                parsed_srt_segments = parse_srt_content(srt_content_str)
                header_subtitle = ['序號', '開始時間', '結束時間', '文字']
                rows_to_upload_subtitle = [[seg['id'], seg['start'], seg['end'], seg['text']] for seg in parsed_srt_segments]
                data_for_subtitle_sheet = [header_subtitle] + rows_to_upload_subtitle
                execute_gspread_write(logger, subtitle_worksheet.update, range_name='A1', values=data_for_subtitle_sheet)
                logger.info(f"數據已成功上傳至工作表 '{subtitle_worksheet_title}'。")

                if base_name in gemini_processed_items:
                    logger.info(f"'{base_name}' 的 Gemini 校對先前已完成，跳過對 API 的調用。")
                elif not normal_text_content.splitlines():
                    logger.info(f"'{base_name}' 的 Whisper 文本為空或無實質內容，跳過 Gemini API 校對。")
                else:
                    logger.info(f"準備對 '{base_name}' 的文本進行 Gemini API 校對...")
                    whisper_lines_for_gemini = normal_text_content.splitlines()

                    if not pdf_context_text:
                        logger.info("注意: PDF 講義內容為空，Gemini 校對將不使用講義參考。")

                    corrected_text_str = get_gemini_correction(
                        logger,
                        whisper_lines_for_gemini,
                        pdf_context_text if pdf_context_text else "",
                        current_main_instruction_param,
                        current_correction_rules_param
                    )

                    if corrected_text_str:
                        gemini_lines = corrected_text_str.strip().split('\n')
                        data_for_gemini_column = [["Gemini"]] + [[line] for line in gemini_lines]
                        try:
                            execute_gspread_write(logger, normal_worksheet.update, range_name='B1', values=data_for_gemini_column)
                            logger.info(f"Gemini API 校對完成 ({len(gemini_lines)} 行)。已成功上傳 Gemini 校對結果至 B欄 ({base_name})。")

                            gemini_processed_items.add(base_name)
                            save_gemini_processed_state(GEMINI_STATE_FILE_PATH, gemini_processed_items)
                            logger.info(f"已將 '{base_name}' 標記為 Gemini 校對完成並更新狀態檔案。")
                        except Exception as e_update:
                            logger.error(f"更新 B欄 Gemini 校對結果時發生錯誤 ({base_name}): {e_update}", exc_info=True)
                    else:
                        logger.warning(f"Gemini API 校對失敗或無返回內容 ({base_name})，B欄將保持空白。將不會標記為 Gemini 校對完成。")

                processed_item_count += 1
                logger.info(f"項目 {base_name} 的表格處理完成。試算表連結: {spreadsheet.url}")
                display(HTML(f"<p>項目 {base_name} 處理完成。試算表連結: <a href='{spreadsheet.url}' target='_blank'>{spreadsheet.url}</a></p>"))

                logger.info(f"已完成對 '{base_name}' 的所有處理。等待 {INTER_SPREADSHEET_DELAY_SECONDS} 秒後處理下一個項目...")
                time.sleep(INTER_SPREADSHEET_DELAY_SECONDS)


            except Exception as e_sheet_ops:
                logger.error(f"處理試算表 '{spreadsheet_name}' 時發生錯誤: {e_sheet_ops}", exc_info=True)
                continue

    if processed_item_count == 0:
        logger.info(f"在 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 目錄中未找到任何有效的轉錄項目進行處理。")
    else:
        logger.info(f"總共處理了 {processed_item_count} 個項目。")

# --- 腳本執行 ---
if __name__ == '__main__':
    logger = logging.getLogger('SheetsGeminiProcessorLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("sheets_gemini_processor.py 腳本已啟動。")

    setup_results = initial_setup(logger)

    if setup_results[0] is None:
        logger.critical("由於 gspread 客戶端 (gc) 初始化失敗，腳本無法繼續。請檢查認證和授權。")
    else:
        _, _, main_instr, correct_rules = setup_results
        process_transcriptions_and_apply_gemini(logger, main_instr, correct_rules)

    logger.info("sheets_gemini_processor.py 腳本已完成。")

[end of sheets_gemini_processor.py]
