# 音頻轉錄與 Gemini API 自動校對項目

## 1. 項目概述

本項目旨在提供一個自動化的流程，用於：
1.  轉錄音頻文件（支持多種常見格式）。
2.  將轉錄後的文本及時間軸上傳至 Google Sheets。
3.  利用 Gemini API 對轉錄文本進行自動校對。
4.  （可選）將校對後的文本按固定時長（如30分鐘）切分成多個文本文件，以便進一步的人工審閱。

項目主要由兩個核心 Python 腳本 (`local_transcriber.py` 和 `sheets_gemini_processor.py`) 以及一個輔助腳本 (`text_segmenter_colab.py`) 組成，設計在本地 Python 環境中運行，並可選擇利用 Google Sheets 和 Gemini API 進行雲端處理。

## 2. 腳本功能詳解

### 2.1. `local_transcriber.py` - 本地音頻轉錄腳本
*   **功能：**
    *   使用 `faster-whisper` 模型對指定的音頻文件夾中的音頻進行批量轉錄。
    *   交互式初始提示詞：腳本運行開始時，會提示用戶輸入自定義的 `INITIAL_PROMPT_TEXT`（初始轉錄提示詞），並提供一個默認值。所選提示詞將用於指導 Whisper 模型進行轉錄。
    *   為每個音頻文件生成兩種輸出：
        1.  純文本文件 (`[文件名]_normal.txt`)：包含完整的、經過初步清理的轉錄文字。
        2.  SRT 字幕文件 (`[文件名].srt`)：包含帶有時間碼的字幕片段。
    *   支持狀態持久化：能夠記錄已成功處理的音頻文件，在中斷後重新運行時會自動跳過這些文件。
    *   增強的 Drive 掛載穩定性：內部已包含針對 Google Colab 環境下 Google Drive 掛載交互的優化措施（如嘗試預先卸載和操作後延遲），以提高穩定性。
    *   包含中文日誌記錄。
*   **輸入：**
    *   運行時用戶輸入的初始提示詞。
    *   位於 Google Drive 中 `INPUT_AUDIO_DIR` 指定的文件夾內的音頻文件（支持 `.mp3`, `.wav`, `.flac`, `.m4a`, `.mp4`）。
*   **輸出：**
    *   在 Google Drive 中 `OUTPUT_TRANSCRIPTIONS_ROOT_DIR` 指定的路徑下，為每個音頻文件創建一個與音頻文件同名的子文件夾（例如 `[音頻文件名基礎名]/`）。
    *   在該子文件夾內保存對應的 `_normal.txt` 和 `.srt` 文件。

### 2.2. `sheets_gemini_processor.py` - Google Sheets 與 Gemini API 處理腳本
*   **功能：**
    *   讀取 `local_transcriber.py` 腳本生成的 `_normal.txt` 和 `.srt` 文件。
    *   為每個原始音頻文件（現在由一個文件夾代表）創建或更新一個同名的 Google Spreadsheet。
    *   在 Spreadsheet 中創建/更新兩個工作表：
        1.  **"文本校對"**：
            *   A欄：從 `_normal.txt` 讀取的 Whisper 轉錄文本（A1為標題 "Whisper"）。
            *   B欄：調用 Gemini API 進行校對後的文本（B1為標題 "Gemini"）。
        2.  **"時間軸"**：
            *   從 `.srt` 文件解析並上傳字幕的序號、開始時間、結束時間和文字。
    *   PDF講義管理：在提取講義內容前，腳本會自動清理 `pdf_handout_dir` 文件夾中所有舊的 PDF 文件。
    *   交互式PDF上傳：清理舊PDF後，腳本會提供一個文件上傳界面，允許用戶上傳新的 PDF 文件至 `pdf_handout_dir`，作為 Gemini 校對的參考資料。
    *   Gemini API 提示詞自定義：腳本運行初期會提示用戶輸入用於指導 Gemini API 的“主要指令”和“校對規則”，並提供可編輯的默認值。這允許用戶根據不同任務需求靈活調整對 Gemini 的指令。
    *   Gemini API 交互優化：調用 Gemini API 的部分已更新為使用官方 `google-generativeai` Python SDK，並默認使用 `gemini-1.5-pro-latest` 模型。同時，內部增強了對長文本的分批處理及每批次返回行數的校驗與自動調整機制，以確保輸出文本結構的完整性。
    *   支持 Gemini 校對的狀態持久化：記錄已成功完成 Gemini 校對的電子表格，在中斷後重新運行時會跳過這些電子表格的 Gemini API 調用步驟。
    *   增強的 Drive 掛載穩定性：與 `local_transcriber.py` 類似，此腳本的 `initial_setup` 函數也包含了優化 Drive 掛載穩定性的步驟。
    *   包含中文日誌記錄。
*   **輸入：**
    *   `local_transcriber.py` 腳本的輸出文件夾和文件。
    *   運行時通過交互式界面上傳的 PDF 講義文件，這些文件將被存儲在 Google Drive 中 `pdf_handout_dir` 指定的文件夾內，用於 Gemini 校對參考。
    *   Gemini API 提示詞：腳本啟動時，會提示用戶確認或修改用於 Gemini API 的主要指令和校對規則。默認值已在腳本中提供。
    *   Google Colab Secrets 中的 `GEMINI_API_KEY`。
*   **輸出：**
    *   在 Google Drive 中創建或更新 Google Spreadsheets。
    *   Spreadsheet 中的 "文本校對" 工作表的 B 欄會被 Gemini API 的校對結果填充。

### 2.3. `text_segmenter_colab.py` - （輔助）文本切分腳本
*   **功能：**
    *   讀取由 `sheets_gemini_processor.py` 生成並經 Gemini 校對的 Google Spreadsheet 中的 "文本校對" 工作表 (B欄) 以及 "時間軸" 工作表。
    *   根據 "時間軸" 的時間信息，將 "文本校對" 工作表B欄的文本內容按每30分鐘切分成多個 `.txt` 文件。
*   **輸入：**
    *   一個已由 `sheets_gemini_processor.py` 處理完畢的 Google Spreadsheet 的名稱。
*   **輸出：**
    *   在 Google Drive 中 `OUTPUT_ROOT_DIR/[Spreadsheet名稱]/` 文件夾下（與 `sheets_gemini_processor.py` 的輸出文件夾結構類似，但這裡指的是 Spreadsheet 名稱），保存切分後的 `.txt` 文件。
    *   每個 `.txt` 文件包含：
        *   第一行：文件名及分片編號 (例如 `T095P002 Part 1 of 6`)。
        *   第二行：該分片對應的時間軸範圍。
        *   第三行起：該時間段的文本內容。
*   **注意：** 此腳本的開發曾因數據不匹配問題提示用戶檢查數據。用戶需確保輸入的 Spreadsheet 中 "文本校對" 和 "時間軸" 的條目數一致方可成功運行。

## 3. 環境設置與安裝

### 3.1. 運行環境
本項目設計在本地 Python 環境 (建議 Python 3.8 或更高版本) 中運行。部分功能依賴 NVIDIA GPU 及相關庫。

### 3.2. 依賴安裝
1.  **創建並激活虛擬環境 (推薦):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

2.  **安裝 Python 依賴包:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **系統級別 GPU 庫 (針對 `faster-whisper` GPU 版本):**
    若要使用 GPU 進行音頻轉錄 (推薦，通過 `onnxruntime-gpu`)，您需要安裝 NVIDIA CUDA Toolkit 和 cuDNN。
    - 請參考 NVIDIA 官方文檔進行安裝，確保版本與 `onnxruntime-gpu` 兼容。
    - `requirements.txt` 中指定的 `onnxruntime-gpu` 版本會對 CUDA/cuDNN 版本有特定要求。

### 3.3. 本地目錄結構建議
腳本默認使用相對路徑。建議在項目根目錄下創建以下文件夾：
*   `./input_audio/`: 存放需要轉錄的音頻文件。 (對應 `local_transcriber.py` 中的 `INPUT_AUDIO_DIR`)
*   `./output_transcriptions/`: 作為 `local_transcriber.py` 和 `sheets_gemini_processor.py` 輸出的根目錄。腳本會在此文件夾下自動創建子文件夾。 (對應 `OUTPUT_TRANSCRIPTIONS_ROOT_DIR` 或 `TRANSCRIPTIONS_ROOT_INPUT_DIR`)
*   `./lecture_handouts/`: 存放 PDF 講義，供 `sheets_gemini_processor.py` 中的 Gemini API 校對時參考。 (對應 `pdf_handout_dir`)

這些路徑可以在各腳本的開頭部分進行修改。

### 3.4. API 密鑰與認證設置
**Gemini API 密鑰:**
您需要在環境變量中設置 `GEMINI_API_KEY`。
例如，在 Linux/macOS: `export GEMINI_API_KEY="YOUR_API_KEY"`
在 Windows (PowerShell): `$env:GEMINI_API_KEY="YOUR_API_KEY"`

**Google Cloud (Sheets API) 認證:**
腳本使用應用程序默認憑據 (Application Default Credentials, ADC) 來訪問 Google Sheets API。
1.  創建一個 Google Cloud Platform (GCP) 項目。
2.  在您的 GCP 項目中啟用 Google Sheets API。
3.  創建一個服務帳戶 (Service Account) 並下載其 JSON 密鑰文件。
4.  設置環境變量 `GOOGLE_APPLICATION_CREDENTIALS` 指向您下載的 JSON 密鑰文件的路徑。
    例如，在 Linux/macOS: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"`
    在 Windows (PowerShell): `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"`

### 3.6. Gemini API 提示詞默認內容
腳本為 Gemini API 校對提供了以下可自定義的默認提示詞結構：

**默認主要指令 (Main Instruction):**
```text
你是一個佛學大師，精通經律論三藏十二部經典。
以下文本是whisper產生的字幕文本，關於觀無量壽經、善導大師觀經四帖疏、傳通記的內容。
有很多聽打錯誤，幫我依據我提供的上課講義校對文本，嚴格依照以下規則，直接修正錯誤：
```

**默認校對規則 (Correction Rules):**
```text
校對規則：
    1. 這是講座字幕的文本。請逐行處理提供的「字幕文本」。
    2. **嚴格依照原本的斷句輸出，保持每一行的獨立性，不要合併或拆分行。輸出結果必須與輸入的行數完全相同 (共 {batch_line_count} 行)。**
    3. 如果某一行不需要修改，請直接輸出原始該行內容。
    4. 根據「上課講義內容」修正「字幕文本」中的任何聽打錯誤或不準確之處。
    5. 不要加標點符號。
    6. 輸出繁體中文。
```
*注意：校對規則中的 `{batch_line_count}` 是一個佔位符。當腳本將文本分批提交給 Gemini API 時，它會在每個批次的 API 調用前，動態地將此佔位符替換為該**當前批次所包含的文本行數**。如果您自定義此規則並希望引用行數，請使用此佔位符。*

### 3.7. Gemini API 文本分批處理機制
為了更穩定地處理較長的轉錄文本，`sheets_gemini_processor.py` 內部實現了對提交給 Gemini API 的文本進行分批處理的機制。腳本會將一個文件的完整轉錄內容按照預設的行數上限（當前為 Pro 模型無上下文測試配置，內部設置為 `GEMINI_API_BATCH_MAX_LINES = 100` 行）分割成若干批次。每個批次會單獨發送給 Gemini API 進行校對，並應用相同的重試邏輯。為遵循 `gemini-1.5-pro-latest` 的 RPM (每分鐘請求數) 限制，在每個批次成功處理後，腳本會內置一個延遲（當前配置為 30 秒）。所有批次成功處理後，結果會被合併。

此機制有助於降低單個 API 請求因文本過長而失敗的風險，並能更有效地利用 API 的處理能力。

### 3.8. 重要測試配置說明
目前版本的 `sheets_gemini_processor.py` 為了針對 `gemini-1.5-pro-latest` 模型進行嚴格的 API 速率限制和無上下文負載測試，在調用 Gemini API 時**默認配置為不使用 PDF 講義上下文** (即 `pdf_context` 參數會被傳遞為空字符串)。這意味著 Gemini 的校對將僅基於轉錄文本本身和您提供的通用指令及校對規則。

如果您希望恢復使用 PDF 上下文進行校對，您需要手動修改 `sheets_gemini_processor.py` 腳本中 `process_transcriptions_and_apply_gemini` 函數內對 `get_gemini_correction` 的調用，將實際從 `initial_setup` 獲取的 `pdf_context_text` 變量傳遞給 `pdf_context` 參數。例如，將：
`corrected_text_str = get_gemini_correction(logger, whisper_lines_for_gemini, "", current_main_instruction_param, current_correction_rules_param)`
修改回類似：
`corrected_text_str = get_gemini_correction(logger, whisper_lines_for_gemini, pdf_context_text, current_main_instruction_param, current_correction_rules_param)`
(請注意，`pdf_context_text` 變量需要在該作用域內可用，它通常是從 `initial_setup` 函數獲取的全局變量或被傳遞的參數。)

## 4. 執行順序

推薦的執行流程如下：

0.  **首次設置:**
    *   確保已安裝所有依賴 (參見 `### 3.2`)。
    *   設置必要的環境變量 (`GEMINI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`) (參見 `### 3.4`)。
    *   根據需要創建本地目錄 (`./input_audio`, `./lecture_handouts`) 並放置相應文件。
1.  **運行 `local_transcriber.py`**：
    *   腳本啟動時，系統會提示您輸入或確認用於 Whisper 轉錄的初始提示詞。
    *   之後，腳本會處理音頻文件，生成 `_normal.txt` 和 `.srt` 文件到配置的輸出目錄中 (默認 `./output_transcriptions/[audio_basename]/`)。
2.  **運行 `sheets_gemini_processor.py`**：
    *   確保 PDF 參考資料已放置在配置的 `pdf_handout_dir` (默認 `./lecture_handouts/`)。腳本會自動清理此目錄下舊的 PDF (如果適用，根據腳本邏輯)。
    *   系統會提示您確認或修改用於指導 Gemini API 的“主要指令”和“校對規則”。
    *   然後，腳本會讀取 `local_transcriber.py` 的輸出，創建/更新 Google Sheets，並調用 Gemini API 進行校對。
3.  **（可選）運行 `text_segmenter_colab.py`**：如果您需要將校對後的文本按30分鐘切分，則在 `sheets_gemini_processor.py` 完成對應的電子表格處理後，運行此腳本。

## 5. 狀態持久化

*   `local_transcriber.py`：會在 `OUTPUT_TRANSCRIPTIONS_ROOT_DIR` (默認 `./output_transcriptions/`) 文件夾下創建一個 `.processed_audio_files.json` 文件，記錄已成功轉錄的音頻文件名。重新運行時會跳過這些文件。
*   `sheets_gemini_processor.py`：會在 `TRANSCRIPTIONS_ROOT_INPUT_DIR` (默認 `./output_transcriptions/`) 文件夾下創建一個 `.gemini_processed_state.json` 文件，記錄已成功完成 Gemini 校對的電子表格（以 `base_name` 標識）。重新運行時，對於已記錄的項目，會跳過 Gemini API 的調用和結果寫入步驟。

## 6. 日誌與註釋語言

*   本項目的 Python 腳本中的**日誌信息**和**代碼註釋**主要使用**中文**編寫。
*   日誌系統：所有腳本均使用 Python 的 `logging` 模塊記錄詳細的操作日誌（中文）。日誌配置已優化，以確保在控制台中能清晰、無重複地輸出。
