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
from IPython.display import HTML # <-- 修正：導入 HTML
import warnings # 導入 warnings 模組

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置區塊 ---
# model_size = "large-v3" # 已移除 - Whisper 特定配置
# initial_prompt_text = "這是佛教關於密教真言宗藥師佛" # 已移除 - Whisper 特定配置

# input_audio_dir = "/content/drive/MyDrive/input_audio" # 已移除 - 不再直接用於轉錄輸入
TRANSCRIPTIONS_ROOT_INPUT_DIR = "/content/drive/MyDrive/output_transcriptions" # 重命名：此腳本的輸入目錄
pdf_handout_dir = "/content/drive/MyDrive/lecture_handouts" # 保留，用於 Gemini 上下文
GEMINI_STATE_FILE_PATH = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, ".gemini_processed_state.json") # Gemini 處理狀態檔案路徑

# --- 輔助函式：格式化SRT時間 ---
# def format_srt_time(seconds): # 已移除 - 屬於 local_transcriber.py
#     """將秒數格式化為 SRT 格式的時間字串 (HH:MM:SS,ms)"""
#     # ... (程式碼已移除)

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
        logging.warning("未能從任何 PDF 檔案提取到文本。") # 從 print 改為 logging.warning
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
def get_gemini_correction(transcribed_text_lines, pdf_context):
    api_key = userdata.get('GEMINI_API_KEY')

    if not api_key:
        # 此 print 對於用戶設置至關重要，可以保留為 print 或使用 logging.critical
        print("錯誤: GEMINI_API_KEY 未設定。請在 Colab Secrets 中設定您的 Gemini API 金鑰。")
        logging.critical("GEMINI_API_KEY 未設定。")
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
    base_delay = 60  # 秒
    response = None # 初始化 response 為 None
    # get_gemini_correction 函數內部的 print 語句被保留，因為它們是其內部重試邏輯顯示的一部分
    # 如果此函數要成為庫函數，那些 print 也應轉換為日誌記錄
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
                    print(f"Gemini API rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})") # 保留此 print 給用戶看
                    time.sleep(delay)
                    continue
                else:
                    print(f"Gemini API rate limit hit (429). Max retries reached. Failing. Error: {e}") # 保留此 print
                    logging.error(f"Gemini API 達到最大重試次數 (429)。錯誤: {e}", exc_info=True)
                    return None
            else:
                print(f"調用 Gemini API 時發生錯誤: {e}") # 保留此 print
                logging.error(f"調用 Gemini API 時發生錯誤: {e}", exc_info=True)
                return None

    if response is None or not response.ok:
        print(f"Gemini API 調用最終失敗。") # 保留此 print
        logging.error("Gemini API 調用最終失敗。")
        return None

    time.sleep(15) # 保留此延遲

    try:
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            corrected_text_from_api = result["candidates"][0]["content"]["parts"][0]["text"]
            corrected_lines = corrected_text_from_api.strip().split('\n')

            if len(corrected_lines) == len(transcribed_text_lines):
                print(f"Gemini 校對完成，行數與原始文本一致 ({len(corrected_lines)} 行)。") # 保留
                return corrected_text_from_api
            else:
                # 這種情況比較棘手，原始的 print 對用戶查看差異很有價值
                print(f"警告: Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。") # 保留
                logging.warning(f"Gemini API 返回的行數 ({len(corrected_lines)}) 與原始文本行數 ({len(transcribed_text_lines)}) 不一致。")
                final_corrected_lines = []
                for i in range(len(transcribed_text_lines)):
                    if i < len(corrected_lines):
                        final_corrected_lines.append(corrected_lines[i])
                    else:
                        final_corrected_lines.append(transcribed_text_lines[i])
                final_corrected_lines = final_corrected_lines[:len(transcribed_text_lines)]
                print(f"已嘗試調整行數以匹配原始文本。建議檢查校對結果。") # 保留
                return "\n".join(final_corrected_lines)
        else:
            print(f"Gemini API 響應結構異常或內容缺失: {result}") # 保留
            logging.error(f"Gemini API 響應結構異常或內容缺失: {result}")
            return None
    except json.JSONDecodeError as e:
        response_text = response.text if response else "No response text available"
        print(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}") # 保留
        logging.error(f"解析 Gemini API 響應時發生錯誤: {e}. 響應文本: {response_text}", exc_info=True)
        return None
    except Exception as e:
        print(f"處理 Gemini API 響應時發生未知錯誤: {e}") # 保留
        logging.error(f"處理 Gemini API 響應時發生未知錯誤: {e}", exc_info=True)
        return None

# --- Google Sheets 認證與設定 ---
# Colab 用戶體驗中，這些用於初始設定的 print 可保留
# logging.info("正在進行 Google Drive 和 Sheets 身份驗證...") (將在 initial_setup 中添加)
gc = None # 全局初始化 gc
pdf_context_text = "" # 全局初始化

def initial_setup():
    global gc, pdf_context_text
    logging.info("正在進行 Google Drive 和 Sheets 身份驗證...")
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        logging.info("Google Drive 和 Sheets 身份驗證成功。")
    except Exception as e:
        logging.error(f"Google Drive 或 Sheets 身份驗證失敗: {e}", exc_info=True)
        # exit() # 原始的 exit，記錄錯誤已足夠，主函數將檢查 gc

    logging.info("正在掛載 Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True) # force_remount 對於一般使用可能過於頻繁
        logging.info("Google Drive 掛載成功。")
    except Exception as e:
        logging.error(f"Google Drive 掛載失敗: {e}", exc_info=True)
        # exit() # 原始的 exit

    logging.info(f"正在從 '{pdf_handout_dir}' 提取 PDF 講義內容...")
    pdf_context_text_local = extract_text_from_pdf_dir(pdf_handout_dir)
    if pdf_context_text_local is None or not pdf_context_text_local.strip():
        logging.warning(f"未能從資料夾 '{pdf_handout_dir}' 提取到有效文本，或資料夾不存在/為空。Gemini 校對將不使用講義參考。")
        pdf_context_text = ""
    else:
        pdf_context_text = pdf_context_text_local
        logging.info("PDF 講義內容提取完成。")


# --- 主要處理邏輯 ---
# 此腳本現在需要一個新的主循環來遍歷 TRANSCRIPTIONS_ROOT_INPUT_DIR 中的子目錄
# (來自 local_transcriber.py 的輸出)。
# 此新循環將在後續步驟中實現。

# 目前，腳本結構已為其新角色設置好。
# 以下是新主循環及其邏輯的佔位符。
# 它演示了 Google Sheets 和 Gemini 部分的保留。

def process_transcriptions_and_apply_gemini():
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

            # 注意：表格準備 (上傳 Whisper 和 SRT 數據) 仍會發生，
            # 因為 Gemini 狀態僅追蹤 Gemini 步驟本身的完成情況。
            # 如果需要完全跳過，此邏輯也需要放在表格操作之外/之前。

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
                    normal_worksheet = spreadsheet.add_worksheet(title=normal_worksheet_title, rows="100", cols="20") # 調整行/列數
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
                    subtitle_worksheet = spreadsheet.add_worksheet(title=subtitle_worksheet_title, rows="100", cols="20") # 調整
                    logging.info(f"已創建新的工作表: '{subtitle_worksheet_title}'")

                logging.info(f"正在清除工作表 '{subtitle_worksheet_title}' 的現有內容...")
                subtitle_worksheet.clear()
                parsed_srt_segments = parse_srt_content(srt_content_str)
                header_subtitle = ['序號', '開始時間', '結束時間', '文字']
                rows_to_upload_subtitle = [[seg['id'], seg['start'], seg['end'], seg['text']] for seg in parsed_srt_segments]
                data_for_subtitle_sheet = [header_subtitle] + rows_to_upload_subtitle
                subtitle_worksheet.update(range_name='A1', values=data_for_subtitle_sheet)
                logging.info(f"數據已成功上傳至工作表 '{subtitle_worksheet_title}'。")


                # --- Gemini API 處理 (有條件地基於狀態) ---
                if base_name in gemini_processed_items:
                    logging.info(f"'{base_name}' 的 Gemini 校對先前已完成，跳過對 API 的調用。")
                elif not normal_text_content.splitlines(): # 檢查 normal_text_content 本身是否為空或僅包含空白
                    logging.info(f"'{base_name}' 的 Whisper 文本為空或無實質內容，跳過 Gemini API 校對。")
                else:
                    logging.info(f"準備對 '{base_name}' 的文本進行 Gemini API 校對...")
                    whisper_lines_for_gemini = normal_text_content.splitlines()

                    if not pdf_context_text: # 檢查全局 pdf_context_text 是否為空
                        logging.info("注意: PDF 講義內容為空，Gemini 校對將不使用講義參考。")

                    corrected_text_str = get_gemini_correction(whisper_lines_for_gemini, pdf_context_text if pdf_context_text else "")

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

                processed_item_count += 1 # 此計數統計已準備好表格的項目。
                                          # Gemini 處理是其中的一個步驟。
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("sheets_gemini_processor.py 腳本已啟動。")
    initial_setup() # 處理身份驗證、掛載 Drive、PDF 上下文
    process_transcriptions_and_apply_gemini()
    logging.info("sheets_gemini_processor.py 腳本已完成。")

# 注意: 舊主循環末尾的 time.sleep(60) 已被移除，因為該循環已不存在。
# Gemini API 調用中已處理任何必要的 API 速率限制延遲。
