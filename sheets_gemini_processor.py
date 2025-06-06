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

# --- 輔助函式：調用 Gemini API 進行校對 ---
def get_gemini_correction(logger, transcribed_text_lines, pdf_context, main_instruction, correction_rules): # logger 實例傳入
    api_key = userdata.get('GEMINI_API_KEY')
    if not api_key:
        print("錯誤: GEMINI_API_KEY 未設定。請在 Colab Secrets 中設定您的 Gemini API 金鑰。") # 保留此 print 作為對用戶的關鍵反饋
        logger.critical("GEMINI_API_KEY 未設定。")
        return None
    headers = {'Content-Type': 'application/json',}
    transcribed_text_single_string = "\n".join(transcribed_text_lines)
    # 動態格式化校對規則中的行數信息
    formatted_correction_rules = correction_rules.format(len_transcribed_text_lines=len(transcribed_text_lines))
    # 使用傳入的指令和規則構建 full_prompt
    full_prompt = f"{main_instruction}\n\n上課講義內容（作為校對參考，請仔細閱讀）：\n---\n{pdf_context}\n---\n\n以下是需要校對的字幕文本 (共 {len(transcribed_text_lines)} 行):\n---\n{transcribed_text_single_string}\n---\n\n{formatted_correction_rules}"
    payload = {"contents": [{"role": "user", "parts": [{"text": full_prompt}]}], "generationConfig": {"temperature": 0.2, "topP": 0.95, "topK": 64, "maxOutputTokens": 8192, "responseMimeType": "text/plain",}}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
    max_retries = 7
    base_delay = 60  # 秒
    response = None
    logger.info("正在調用 Gemini API 進行文本校對，這可能需要一些時間...")
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    # 保留 print 作為重試期間的直接交互反饋, 同時也用 logger 記錄
                    print(f"Gemini API rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"Gemini API 速率限制 (429)。將在 {delay} 秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    # 針對 429 錯誤達到最大重試次數的增強錯誤消息
                    detailed_error_message = (
                        f"Gemini API 調用因達到最大重試次數 ({max_retries} 次) 而最終失敗 (HTTP 429 Too Many Requests)。錯誤詳情: {e}\n"
                        f"這可能表示您的 Gemini API 金鑰遇到了較嚴格的配額限制（例如每日請求總量、每分鐘請求數或特定於 'gemini-1.5-pro-latest' 模型的限制）。\n"
                        f"**強烈建議您前往 Google Cloud Console (或您的 API 金鑰管理平台) 檢查 Gemini API 的用量和配額詳情。**\n"
                        f"如果持續遇到此問題且確認非暫時性網絡波動，您可能需要考慮：\n"
                        f"  1. 聯繫 Google Cloud 支持或查閱 Gemini API 文檔以了解確切的配額限制。\n"
                        f"  2. 升級您的 API 配額（如果可能）。\n"
                        f"  3. 調整您的任務批次大小或處理頻率以適應當前配額。\n"
                        f"當前文件將不會進行 Gemini 校對。"
                    )
                    logger.error(detailed_error_message)
                    print(f"Gemini API rate limit hit (429). Max retries reached. Failing. Error: {e}") # 保留此 print 作為對用戶的直接反饋
                    return None
            else:
                print(f"調用 Gemini API 時發生錯誤: {e}")
                logger.error(f"調用 Gemini API 時發生錯誤 (非429): {e}", exc_info=True)
                return None
    if response is None or not response.ok:
        print(f"Gemini API 調用最終失敗。")
        logger.error("Gemini API 調用最終失敗 (循環結束後 response 仍無效)。")
        return None
    time.sleep(15) # 保留此延遲
    try:
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            corrected_text_from_api = result["candidates"][0]["content"]["parts"][0]["text"]
            corrected_lines = corrected_text_from_api.strip().split('\n')
            if len(corrected_lines) == len(transcribed_text_lines):
                print(f"Gemini 校對完成，行數與原始文本一致 ({len(corrected_lines)} 行)。")
                logger.info(f"Gemini 校對完成，行數與原始文本一致 ({len(corrected_lines)} 行)。")
                return corrected_text_from_api
            else:
                print(f"警告: Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。")
                logger.warning(f"Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。")
                final_corrected_lines = []
                for i in range(len(transcribed_text_lines)):
                    if i < len(corrected_lines):
                        final_corrected_lines.append(corrected_lines[i])
                    else:
                        final_corrected_lines.append(transcribed_text_lines[i])
                final_corrected_lines = final_corrected_lines[:len(transcribed_text_lines)]
                print(f"已嘗試調整行數以匹配原始文本。建議檢查校對結果。")
                logger.info("已嘗試調整 Gemini API 返回文本的行數以匹配原始文本。")
                return "\n".join(final_corrected_lines)
        else:
            print(f"Gemini API 響應結構異常或內容缺失: {result}")
            logger.error(f"Gemini API 響應結構異常或內容缺失: {result}")
            return None
    except json.JSONDecodeError as e:
        response_text = response.text if response else "No response text available"
        print(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}")
        logger.error(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}", exc_info=True)
        return None
    except Exception as e:
        print(f"處理 Gemini API 響應時發生未知錯誤: {e}")
        logger.error(f"處理 Gemini API 響應時發生未知錯誤: {e}", exc_info=True)
        return None

gc = None # 全局 gspread 客戶端實例
pdf_context_text = "" # 全局 PDF 文本內容
# 使用默認值初始化全局提示詞變量
current_main_instruction = DEFAULT_GEMINI_MAIN_INSTRUCTION
current_correction_rules = DEFAULT_GEMINI_CORRECTION_RULES

def initial_setup(logger_instance): # 接受 logger 實例作為參數
    global gc, pdf_context_text, current_main_instruction, current_correction_rules

    logger_instance.info("正在進行 Google Drive 和 Sheets 身份驗證...")
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        logger_instance.info("Google Drive 和 Sheets 身份驗證成功。")
    except Exception as e:
        logger_instance.error(f"Google Drive 或 Sheets 身份驗證失敗: {e}", exc_info=True)
        gc = None # 如果認證失敗，確保 gc 為 None

    if gc: # 僅在認證成功時繼續
        logger_instance.info("正在掛載 Google Drive...")
        try:
            drive.mount('/content/drive', force_remount=True) # force_remount 對於一般使用可能過於頻繁
            logger_instance.info("Google Drive 掛載成功。")
        except Exception as e:
            logger_instance.error(f"Google Drive 掛載失敗: {e}", exc_info=True)

    # --- Gemini API 提示詞設定 ---
    logger_instance.info("\n--- Gemini API 提示詞設定 ---")
    logger_instance.info("當前默認的主要指令 (main instruction):")
    for line in DEFAULT_GEMINI_MAIN_INSTRUCTION.split('\n'):
        logger_instance.info(f"  {line}")

    print("\n" + "="*30) # 使用 print 以便在 Colab 中進行直接用戶交互以獲取輸入
    user_main_instruction_input = input(f"請輸入新的主要指令 (可直接粘貼多行文本)，或直接按 Enter 使用默認指令:\n")
    print("="*30)
    current_main_instruction = user_main_instruction_input.strip() if user_main_instruction_input.strip() else DEFAULT_GEMINI_MAIN_INSTRUCTION

    logger_instance.info(f"Gemini API 將使用以下主要指令:")
    for line in current_main_instruction.split('\n'):
        logger_instance.info(f"  {line}")

    logger_instance.info("\n當前默認的校對規則 (correction rules):")
    # 向用戶顯示帶有佔位符的模板規則
    temp_formatted_rules_for_display = DEFAULT_GEMINI_CORRECTION_RULES.replace("{len(transcribed_text_lines)}", "<行數>")
    for line in temp_formatted_rules_for_display.split('\n'):
        logger_instance.info(f"  {line}")

    print("\n" + "="*30)
    print("請輸入新的校對規則。您可以直接粘貼多行文本。")
    print("如果您的規則中需要引用轉錄文本的總行數，請使用占位符 '{len_transcribed_text_lines}' (不含引號)。")
    print("如果保留空白並直接按 Enter，將使用上述默認規則。")
    print("="*30)
    user_correction_rules_input = input("粘貼您的校對規則於此:\n")
    current_correction_rules = user_correction_rules_input.strip() if user_correction_rules_input.strip() else DEFAULT_GEMINI_CORRECTION_RULES

    logger_instance.info(f"Gemini API 將使用以下校對規則 (占位符将在运行时替换):")
    temp_formatted_rules_for_display_user = current_correction_rules.replace("{len_transcribed_text_lines}", "<行數>")
    for line in temp_formatted_rules_for_display_user.split('\n'):
        logger_instance.info(f"  {line}")

    # --- PDF 講義文件夾清理 ---
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

    # --- PDF 上傳界面 ---
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

    logger_instance.info(f"正在從 '{pdf_handout_dir}' 提取 PDF 講義內容...") # 在清理和潛在上傳之後執行
    pdf_context_text_local = extract_text_from_pdf_dir(pdf_handout_dir)
    if pdf_context_text_local is None or not pdf_context_text_local.strip():
        logger_instance.warning(f"未能從資料夾 '{pdf_handout_dir}' 提取到有效文本，或資料夾不存在/為空。Gemini 校對將不使用講義參考。")
        pdf_context_text = ""
    else:
        pdf_context_text = pdf_context_text_local
        logger_instance.info("PDF 講義內容提取完成。")

    # 返回所有必要的全局性變量，這些變量在此處設置
    return gc, pdf_context_text, current_main_instruction, current_correction_rules


def process_transcriptions_and_apply_gemini(logger, current_main_instruction_param, current_correction_rules_param): # 接受 logger 和重命名的參數
    global gc, pdf_context_text # 確保訪問在 initial_setup 中可能已更新的全局變量
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

            # 注意：表格準備 (上傳 Whisper 和 SRT 數據) 仍會發生，
            # 因為 Gemini 狀態僅追蹤 Gemini 步驟本身的完成情況。
            # 如果需要完全跳過 (例如，如果表格也已確認完成)，此邏輯需要放在表格操作之外/之前，
            # 並可能需要一個更全面的狀態文件。

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
                    normal_worksheet = execute_gspread_write(logger, spreadsheet.add_worksheet, title=normal_worksheet_title, rows="100", cols="20") # 調整行/列數
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
                    subtitle_worksheet = execute_gspread_write(logger, spreadsheet.add_worksheet, title=subtitle_worksheet_title, rows="100", cols="20") # 調整
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


                # --- Gemini API 處理 (有條件地基於狀態) ---
                if base_name in gemini_processed_items:
                    logger.info(f"'{base_name}' 的 Gemini 校對先前已完成，跳過對 API 的調用。")
                elif not normal_text_content.splitlines(): # 檢查 normal_text_content 本身是否為空或僅包含空白
                    logger.info(f"'{base_name}' 的 Whisper 文本為空或無實質內容，跳過 Gemini API 校對。")
                else:
                    logger.info(f"準備對 '{base_name}' 的文本進行 Gemini API 校對...")
                    whisper_lines_for_gemini = normal_text_content.splitlines()

                    if not pdf_context_text: # 檢查全局 pdf_context_text 是否為空
                        logger.info("注意: PDF 講義內容為空，Gemini 校對將不使用講義參考。")

                    corrected_text_str = get_gemini_correction(
                        logger, # 傳入 logger
                        whisper_lines_for_gemini,
                        pdf_context_text if pdf_context_text else "",
                        current_main_instruction_param, # 使用配置的主要指令
                        current_correction_rules_param  # 使用配置的校對規則
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

                processed_item_count += 1 # 此計數統計已準備好表格的項目。
                                          # Gemini 處理是其中的一個步驟。
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
    # 日誌記錄器在此處配置，因此 initial_setup 和 process_transcriptions_and_apply_gemini 中的 logger.info 等將正常工作
    logger = logging.getLogger('SheetsGeminiProcessorLogger') # 特定日誌記錄器名稱
    logger.setLevel(logging.INFO)
    if not logger.handlers: # 避免在 Colab 中重複運行儲存格時重複添加 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # logger.propagate = False # 可選：阻止日誌消息傳播到 root logger

    logger.info("sheets_gemini_processor.py 腳本已啟動。")

    # initial_setup 現在返回配置的提示詞，並接受 logger 實例
    # 需要處理 gc 可能為 None 的情況 (如果認證失敗)
    setup_results = initial_setup(logger)

    if setup_results[0] is None: # 檢查 gc 是否為 None
        logger.critical("由於 gspread 客戶端 (gc) 初始化失敗，腳本無法繼續。請檢查認證和授權。")
    else:
        # 從 initial_setup 解包所有返回值
        # gc_client, pdf_text, main_instr, correct_rules = setup_results # gc 和 pdf_text 已是全局變量並在 initial_setup 中設置
        _, _, main_instr, correct_rules = setup_results # 僅需要提示詞部分，gc 和 pdf_context_text 是全局的

        # 將配置的提示詞和 logger 傳遞給主處理函數
        process_transcriptions_and_apply_gemini(logger, main_instr, correct_rules)

    logger.info("sheets_gemini_processor.py 腳本已完成。")
