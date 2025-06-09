# ==============================================================================
# 導入必要的函式庫
# ==============================================================================
import os
import json
import time
import re
import glob
import warnings
import datetime
import logging

# Google Colab 和 API 相關函式庫
from google.colab import drive, auth, userdata
from google.auth import default
from IPython.display import HTML
import gspread
import google.generativeai as genai
import pypdf

# 抑制 gspread 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
# 全域配置區塊 (Global Configuration)
# ==============================================================================
# --- 目錄與檔案路徑 ---
TRANSCRIPTIONS_ROOT_INPUT_DIR = "/content/drive/MyDrive/output_transcriptions"
PDF_HANDOUT_DIR = "/content/drive/MyDrive/lecture_handouts"
GEMINI_STATE_FILE_PATH = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, ".gemini_processed_state.json")

# --- API 延遲與批次設定 ---
INTER_SPREADSHEET_DELAY_SECONDS = 15  # 處理不同Google試算表之間的延遲
GEMINI_API_BATCH_MAX_LINES = 100      # 每批次發送給 Gemini API 的最大行數

# --- Gemini API 模型與延遲 ---
# 注意: gemini-1.5-pro 免費額度為 2 RPM (每分鐘請求數)，因此批次間延遲需 > 30秒
GEMINI_MODEL_NAME = "models/gemini-2.5-pro-experimental-0325-latest"
GEMINI_PRO_SAFE_DELAY_SECONDS = 15 # 為 2 RPM (30秒/次) 限制增加安全邊際

# ==============================================================================
# 默認 Gemini API 提示詞常量 (Default Prompts)
# ==============================================================================

# --- 當有 PDF 講義時的提示詞 ---
PDF_BASED_MAIN_INSTRUCTION = (
    "# ROLE\n"
    "你是一位精通三藏十二部經的佛學專家，專長是依據文獻進行文本校對。\n\n"
    "# CONTEXT\n"
    "- **校對目標**: 以下是關於《觀無量壽經》、《觀經四帖疏》與《傳通記》講座的 Whisper 自動語音辨識字幕。\n"
    "- **參考資料**: 我將提供一份上課講義作為唯一的校對依據。\n"
    "- **任務**: 你的任務是依據「參考資料」逐行校對「字幕文本」，修正所有聽打錯誤。"
)
PDF_BASED_CORRECTION_RULES = (
    "# RULES\n"
    "1.  **[最重要] 格式一致性**: 輸出必須與輸入的行數完全相同 (共 {batch_line_count} 行)。絕不合併或拆分任何行。\n"
    "2.  **逐行處理**: 逐行校對。若某行無誤，直接複製該行原文。\n"
    "3.  **校對依據**: 僅根據下方提供的「參考資料」修正錯字、專有名詞和法義錯誤。\n"
    "4.  **輸出格式**:\n"
    "    - 全程使用繁體中文。\n"
    "    - 移除所有標點符號。"
)

# --- 當沒有 PDF 講義時的提示詞 ---
KNOWLEDGE_BASED_MAIN_INSTRUCTION = (
    "# ROLE\n"
    "你是一位精通三藏十二部經的佛學專家，專長是進行文本校對。\n\n"
    "# CONTEXT\n"
    "- **校對目標**: 以下是關於《觀無量壽經》、《觀經四帖疏》與《傳通記》講座的 Whisper 自動語音辨識字幕。\n"
    "- **任務**: 由於沒有提供參考講義，請依據你自身的佛學專業知識和通用語言規則，逐行校對「字幕文本」，修正所有聽打錯誤。"
)
KNOWLEDGE_BASED_CORRECTION_RULES = (
    "# RULES\n"
    "1.  **[最重要] 格式一致性**: 輸出必須與輸入的行數完全相同 (共 {batch_line_count} 行)。絕不合併或拆分任何行。\n"
    "2.  **逐行處理**: 逐行校對。若某行無誤，直接複製該行原文。\n"
    "3.  **校對依據**: 修正任何明顯的聽打錯誤、錯別字或不通順之處。\n"
    "4.  **輸出格式**:\n"
    "    - 全程使用繁體中文。\n"
    "    - 移除所有標點符號。"
)

# ==============================================================================
# 輔助函式 (Helper Functions)
# ==============================================================================

def execute_gspread_write(logger, gspread_operation_func, *args, max_retries=5, base_delay_seconds=5, **kwargs):
    """執行 gspread 寫入操作，並為 API 錯誤 (特別是 429) 提供指數退避重試機制。"""
    for attempt in range(max_retries):
        try:
            return gspread_operation_func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            if hasattr(e, 'response') and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay_seconds * (2 ** attempt)
                    logger.warning(f"Google Sheets API 速率限制 (429)。將在 {delay} 秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Google Sheets API 達到最大重試次數 (429)。錯誤: {e}", exc_info=True)
                    raise
            else:
                logger.error(f"Google Sheets API 發生非預期的 API 錯誤: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"執行 gspread 操作時發生未知錯誤: {e}", exc_info=True)
            raise
    return None

def extract_text_from_pdf_dir(logger, pdf_dir):
    """從指定資料夾中的所有 PDF 檔案提取文本。"""
    full_text = []
    if not os.path.exists(pdf_dir):
        logger.warning(f"PDF 講義資料夾 '{pdf_dir}' 不存在。")
        return ""

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.info(f"資料夾 '{pdf_dir}' 中沒有找到任何 PDF 檔案。")
        return ""

    logger.info(f"正在從 {len(pdf_files)} 個 PDF 檔案提取文本...")
    for pdf_file_name in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file_name)
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    full_text.append(page.extract_text())
            logger.info(f"  - 成功提取 '{pdf_file_name}'。")
        except Exception as e:
            logger.error(f"  - 從 '{pdf_file_name}' 提取文本時發生錯誤: {e}", exc_info=True)

    combined_text = "\n".join(filter(None, full_text))
    if combined_text:
        logger.info(f"所有 PDF 提取完成，總文本長度: {len(combined_text)} 字元。")
    else:
        logger.warning("未能從任何 PDF 檔案中提取到有效文本。")
    return combined_text

def load_gemini_processed_state(logger, state_file_path):
    """從狀態檔案載入已處理的項目集合。"""
    if not os.path.exists(state_file_path):
        logger.info(f"Gemini 狀態檔案未找到，將處理所有項目。")
        return set()
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data if isinstance(data, list) else [])
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"解析 Gemini 狀態檔案 '{state_file_path}' 時出錯 ({e})。將重新處理所有項目。")
        return set()

def save_gemini_processed_state(logger, state_file_path, processed_items_set):
    """將已處理的項目集合儲存到狀態檔案。"""
    temp_state_file_path = state_file_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        with open(temp_state_file_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_items_set), f, ensure_ascii=False, indent=4)
        os.replace(temp_state_file_path, state_file_path)
        logger.debug(f"Gemini 處理狀態已成功儲存。")
    except Exception as e:
        logger.error(f"儲存 Gemini 處理狀態時發生錯誤: {e}", exc_info=True)

def parse_srt_content(srt_content_str):
    """解析 SRT 字幕內容為結構化數據。"""
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
            'text': match.group(4).strip().replace('\n', ' ') # 將多行字幕合併為一行
        })
    return segments

# ==============================================================================
# Gemini API 核心函式
# ==============================================================================

def get_gemini_correction(logger, transcribed_text_lines, pdf_context, main_instruction, correction_rules):
    """使用 Gemini SDK 呼叫 API 進行文本校對，包含分批、重試和安全延遲。"""
    try:
        api_key = userdata.get('GEMINI_API_KEY')
        if not api_key:
            logger.critical("錯誤: GEMINI_API_KEY 未在 Colab Secrets 中設定。")
            return None
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"配置 Gemini SDK 時出錯: {e}", exc_info=True)
        return None

    logger.info(f"Gemini API 將使用模型: {GEMINI_MODEL_NAME}")
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config={
            "temperature": 0.2, "top_p": 0.95, "top_k": 64,
            "max_output_tokens": 8192, "response_mime_type": "text/plain"
        }
    )

    all_corrected_lines_from_batches = []
    total_lines = len(transcribed_text_lines)
    if total_lines == 0:
        logger.info("沒有需要校對的文本行。")
        return ""

    num_batches = (total_lines + GEMINI_API_BATCH_MAX_LINES - 1) // GEMINI_API_BATCH_MAX_LINES
    logger.info(f"文本總行數: {total_lines}。將分割成 {num_batches} 個批次進行校對。")

    max_retries_gemini = 5
    base_delay_gemini = 60

    for batch_idx in range(num_batches):
        start_index = batch_idx * GEMINI_API_BATCH_MAX_LINES
        end_index = min((batch_idx + 1) * GEMINI_API_BATCH_MAX_LINES, total_lines)
        current_batch_lines = transcribed_text_lines[start_index:end_index]

        if not current_batch_lines:
            continue

        logger.info(f"正在處理第 {batch_idx+1}/{num_batches} 批次的文本 (行 {start_index+1} 到 {end_index})...")
        batch_transcribed_text_single_string = "\n".join(current_batch_lines)

        # 格式化帶有批次行數的規則
        batch_specific_correction_rules = correction_rules.format(batch_line_count=len(current_batch_lines))

        # 根據是否有 PDF 內容來構建 Prompt
        if pdf_context.strip():
            full_prompt_for_batch = (
                f"{main_instruction}\n\n"
                f"{batch_specific_correction_rules}\n\n"
                f"--- 以下是你需要處理的實際任務 ---\n\n"
                f"<參考資料>\n{pdf_context}\n</參考資料>\n\n"
                f"<字幕文本>\n{batch_transcribed_text_single_string}\n</字幕文本>\n\n"
                f"<校對後輸出>"
            )
        else:
            full_prompt_for_batch = (
                f"{main_instruction}\n\n"
                f"{batch_specific_correction_rules}\n\n"
                f"--- 以下是你需要處理的實際任務 ---\n\n"
                f"<字幕文本>\n{batch_transcribed_text_single_string}\n</字幕文本>\n\n"
                f"<校對後輸出>"
            )

        for attempt in range(max_retries_gemini):
            try:
                response = model.generate_content(full_prompt_for_batch)
                corrected_text = response.text

                # 行數校驗與對齊
                api_lines = corrected_text.strip().split('\n')
                if len(api_lines) != len(current_batch_lines):
                    logger.warning(f"批次 {batch_idx+1}: API 返回行數 ({len(api_lines)}) 與預期 ({len(current_batch_lines)}) 不符，正在校準。")
                    adjusted_lines = []
                    for i in range(len(current_batch_lines)):
                        if i < len(api_lines):
                            adjusted_lines.append(api_lines[i])
                        else:
                            # 如果 API 返回行數不足，用原文填充以保證行數不變
                            adjusted_lines.append(current_batch_lines[i])
                    all_corrected_lines_from_batches.extend(adjusted_lines)
                else:
                    all_corrected_lines_from_batches.extend(api_lines)

                logger.info(f"批次 {batch_idx+1}/{num_batches} 處理成功。")
                break  # 成功，跳出重試循環

            except Exception as e:
                error_message = str(e).lower()
                if "429" in error_message or "resourceexhausted" in str(type(e)).lower() or "rate limit" in error_message:
                    if attempt < max_retries_gemini - 1:
                        delay = base_delay_gemini * (2 ** attempt)
                        logger.warning(f"Gemini API 速率限制。將在 {delay} 秒後重試 (嘗試 {attempt + 2}/{max_retries_gemini})...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Gemini API 已達最大重試次數，處理失敗。錯誤: {e}", exc_info=True)
                        return None
                else:
                    logger.error(f"調用 Gemini API 時發生嚴重錯誤: {e}", exc_info=True)
                    return None
        else: # for...else 循環，如果 for 正常結束 (沒有被 break)，則執行 else
            logger.error(f"批次 {batch_idx+1} 所有重試均失敗。")
            return None

        # 在批次之間增加安全延遲，避免 RPM 限制
        if batch_idx < num_batches - 1:
            logger.info(f"等待 {GEMINI_PRO_SAFE_DELAY_SECONDS} 秒以符合 API 速率限制...")
            time.sleep(GEMINI_PRO_SAFE_DELAY_SECONDS)

    logger.info("所有批次均已成功處理。")
    return "\n".join(all_corrected_lines_from_batches)

# ==============================================================================
# 主要執行流程 (Main Execution Flow)
# ==============================================================================

def initial_setup(logger):
    """執行初始設定，包括認證、掛載Drive、以及上傳PDF。"""
    global gc, pdf_context_text

    logger.info("正在進行 Google 認證...")
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        logger.info("Google 認證成功。")
    except Exception as e:
        logger.critical(f"Google 認證失敗: {e}", exc_info=True)
        return None, "", "", ""

    logger.info("正在掛載 Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True)
        logger.info("Google Drive 掛載成功。")
    except Exception as e:
        logger.error(f"Google Drive 掛載失敗: {e}", exc_info=True)

    # 清理並上傳 PDF
    if os.path.exists(PDF_HANDOUT_DIR):
        files_to_delete = glob.glob(os.path.join(PDF_HANDOUT_DIR, "*.pdf"))
        if files_to_delete:
            logger.info(f"正在清理舊的 PDF 講義...")
            for f in files_to_delete:
                os.remove(f)
            logger.info(f"已刪除 {len(files_to_delete)} 個舊 PDF 檔案。")
    else:
        os.makedirs(PDF_HANDOUT_DIR)
        logger.info(f"已創建 PDF 講義目錄: {PDF_HANDOUT_DIR}")

    print("\n請選擇要上傳到 Google Drive 的 PDF 講義文件（如果不需要，可點擊取消）：")
    try:
        from google.colab import files
        uploaded = files.upload()
        for filename, content in uploaded.items():
            dest_path = os.path.join(PDF_HANDOUT_DIR, filename)
            with open(dest_path, 'wb') as f:
                f.write(content)
            logger.info(f"已上傳 '{filename}' 至 '{dest_path}'")
    except Exception as e:
        logger.warning(f"PDF 上傳過程中斷或出錯: {e}")

    # 提取 PDF 文本
    pdf_context_text = extract_text_from_pdf_dir(logger, PDF_HANDOUT_DIR)

    # 根據是否有 PDF 內容，選擇對應的提示詞
    if pdf_context_text.strip():
        logger.info("檢測到 PDF 內容，將使用基於講義的校對提示詞。")
        main_instr = PDF_BASED_MAIN_INSTRUCTION
        correct_rules = PDF_BASED_CORRECTION_RULES
    else:
        logger.info("未檢測到 PDF 內容，將使用基於模型自身知識的校對提示詞。")
        main_instr = KNOWLEDGE_BASED_MAIN_INSTRUCTION
        correct_rules = KNOWLEDGE_BASED_CORRECTION_RULES

    return gc, pdf_context_text, main_instr, correct_rules

def process_transcriptions_and_apply_gemini(logger, main_instruction, correction_rules):
    """主處理循環：遍歷項目，讀取文本，上傳至Sheets，並調用Gemini進行校對。"""
    global gc, pdf_context_text
    if not gc:
        logger.error("gspread client 未初始化，終止處理。")
        return

    gemini_processed_items = load_gemini_processed_state(logger, GEMINI_STATE_FILE_PATH)
    logger.info(f"已載入 {len(gemini_processed_items)} 個已處理項目記錄。")

    if not os.path.exists(TRANSCRIPTIONS_ROOT_INPUT_DIR):
        logger.error(f"轉錄輸入目錄 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 未找到。")
        return

    item_names = sorted([d for d in os.listdir(TRANSCRIPTIONS_ROOT_INPUT_DIR) if os.path.isdir(os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, d))])
    if not item_names:
        logger.warning(f"在 '{TRANSCRIPTIONS_ROOT_INPUT_DIR}' 中未找到任何項目子目錄。")
        return

    logger.info(f"發現 {len(item_names)} 個項目，準備開始處理...")
    processed_count = 0
    for item_name in item_names:
        item_path = os.path.join(TRANSCRIPTIONS_ROOT_INPUT_DIR, item_name)

        logger.info(f"--- ({processed_count+1}/{len(item_names)}) 開始處理項目: {item_name} ---")

        # 讀取 Whisper 轉錄的 TXT 檔案
        normal_text_path = os.path.join(item_path, f"{item_name}_normal.txt")
        if not os.path.exists(normal_text_path):
            logger.warning(f"一般文本檔案未找到: {normal_text_path}，跳過此項目。")
            continue
        with open(normal_text_path, 'r', encoding='utf-8') as f:
            normal_text_content = f.read()

        # 準備上傳到 Google Sheets
        try:
            spreadsheet_name = item_name
            try:
                spreadsheet = gc.open(spreadsheet_name)
                logger.info(f"已開啟現有試算表: '{spreadsheet_name}'")
            except gspread.exceptions.SpreadsheetNotFound:
                spreadsheet = gc.create(spreadsheet_name)
                logger.info(f"已創建新的試算表: '{spreadsheet_name}'")

            # 處理 "文本校對" 工作表
            worksheet_title = "文本校對"
            try:
                worksheet = spreadsheet.worksheet(worksheet_title)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = execute_gspread_write(logger, spreadsheet.add_worksheet, title=worksheet_title, rows="100", cols="20")

            execute_gspread_write(logger, worksheet.clear)
            whisper_lines = normal_text_content.splitlines()
            data_to_upload = [["Whisper"]] + [[line] for line in whisper_lines]
            execute_gspread_write(logger, worksheet.update, range_name='A1', values=data_to_upload)
            logger.info(f"已將 Whisper 文本上傳至工作表 '{worksheet_title}'。")

            # 檢查是否需要調用 Gemini
            if item_name in gemini_processed_items:
                logger.info(f"項目 '{item_name}' 先前已完成 Gemini 校對，跳過 API 調用。")
            elif not whisper_lines:
                logger.info(f"項目 '{item_name}' 的文本為空，跳過 Gemini API 校對。")
            else:
                logger.info(f"準備對 '{item_name}' 的文本進行 Gemini API 校對...")
                corrected_text_str = get_gemini_correction(
                    logger,
                    whisper_lines,
                    pdf_context_text,
                    main_instruction,
                    correction_rules
                )

                if corrected_text_str:
                    gemini_lines = corrected_text_str.split('\n')
                    data_for_gemini_column = [["Gemini"]] + [[line] for line in gemini_lines]
                    execute_gspread_write(logger, worksheet.update, range_name='B1', values=data_for_gemini_column)
                    logger.info(f"Gemini 校對結果已成功上傳至 B 欄。")

                    gemini_processed_items.add(item_name)
                    save_gemini_processed_state(logger, GEMINI_STATE_FILE_PATH, gemini_processed_items)
                else:
                    logger.warning(f"Gemini API 校對失敗或無返回內容，B欄將保持空白。")

            processed_count += 1
            logger.info(f"項目 {item_name} 處理完成。試算表連結: {spreadsheet.url}")
            display(HTML(f"<p>項目 {item_name} 處理完成。試算表連結: <a href='{spreadsheet.url}' target='_blank'>{spreadsheet.url}</a></p>"))

            if processed_count < len(item_names):
                 logger.info(f"等待 {INTER_SPREADSHEET_DELAY_SECONDS} 秒後處理下一個項目...")
                 time.sleep(INTER_SPREADSHEET_DELAY_SECONDS)

        except Exception as e:
            logger.error(f"處理項目 '{item_name}' 時發生嚴重錯誤: {e}", exc_info=True)
            continue

    logger.info(f"所有項目處理流程結束。總共處理了 {processed_count} 個項目。")


# ==============================================================================
# 腳本執行入口 (Script Entry Point)
# ==============================================================================
if __name__ == '__main__':
    # 配置日誌記錄器 (Logger)
    logger = logging.getLogger('SheetsGeminiProcessorLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False

    logger.info("="*50)
    logger.info("佛學字幕自動校對腳本已啟動")
    logger.info("="*50)

    # 執行設定
    gc_client, _, instruction, rules = initial_setup(logger)

    # 執行主流程
    if gc_client:
        process_transcriptions_and_apply_gemini(logger, instruction, rules)
    else:
        logger.critical("由於 Google Client 初始化失敗，腳本無法繼續。")

    logger.info("="*50)
    logger.info("腳本執行完畢。")
    logger.info("="*50)
