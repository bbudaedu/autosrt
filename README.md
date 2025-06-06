# 音頻轉錄與 Gemini API 自動校對項目

## 1. 項目概述

本項目旨在提供一個自動化的流程，用於：
1.  轉錄音頻文件（支持多種常見格式）。
2.  將轉錄後的文本及時間軸上傳至 Google Sheets。
3.  利用 Gemini API 對轉錄文本進行自動校對。
4.  （可選）將校對後的文本按固定時長（如30分鐘）切分成多個文本文件，以便進一步的人工審閱。

項目主要由兩個核心 Python 腳本 (`local_transcriber.py` 和 `sheets_gemini_processor.py`) 以及一個輔助腳本 (`text_segmenter_colab.py`) 組成，設計在 Google Colab 環境中運行。

## 2. 腳本功能詳解

### 2.1. `local_transcriber.py` - 本地音頻轉錄腳本
*   **功能：**
    *   使用 `faster-whisper` 模型對指定的音頻文件夾中的音頻進行批量轉錄。
    *   為每個音頻文件生成兩種輸出：
        1.  純文本文件 (`[文件名]_normal.txt`)：包含完整的、經過初步清理的轉錄文字。
        2.  SRT 字幕文件 (`[文件名].srt`)：包含帶有時間碼的字幕片段。
    *   支持狀態持久化：能夠記錄已成功處理的音頻文件，在中斷後重新運行時會自動跳過這些文件。
    *   包含中文日誌記錄。
*   **輸入：**
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
    *   利用 Gemini API (`gemini-1.5-pro-latest` 模型) 對 "文本校對" 工作表A欄中的文本進行校對，參考 `pdf_handout_dir` 文件夾中的 PDF 講義內容（如果提供）。
    *   支持 Gemini 校對的狀態持久化：記錄已成功完成 Gemini 校對的電子表格，在中斷後重新運行時會跳過這些電子表格的 Gemini API 調用步驟。
    *   包含中文日誌記錄。
*   **輸入：**
    *   `local_transcriber.py` 腳本的輸出文件夾和文件。
    *   位於 Google Drive 中 `pdf_handout_dir` 指定文件夾內的 PDF 講義（可選，用於 Gemini 校對參考）。
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
*   本项目設計在 **Google Colab** 環境中運行。

### 3.2. 依賴安裝
*   在運行任何 Python 腳本之前，請務必在 Colab 筆記本的**最頂部創建一個代碼儲存格**，並執行以下完整的依賴安裝命令。
*   這些命令會設置運行 `faster-whisper` (GPU版) 以及與 Google API 交互所需的環境。

```sh
# Colab 安裝設定儲存格

# 0. 清除 pip 緩存 (可選，但建議保留)
print("INFO: 清除 pip 緩存...")
!pip cache purge

# 1. 安裝系統級別的 GPU 函式庫 (針對 faster-whisper GPU 版本)
print("\nINFO: 更新 apt 並安裝 CUDA/cuDNN 相關函式庫...")
!apt-get update && apt-get install -y libcublas11 libcudnn8

# 2. 強制安裝特定版本的 NumPy (faster-whisper 及其依賴所需)
print("\nINFO: 強制重新安裝 NumPy v1.26.4...")
!pip install --force-reinstall numpy==1.26.4

# 3. 安裝 faster-whisper 的核心依賴 (ctranslate2 和 onnxruntime-gpu)
print("\nINFO: 安裝 ctranslate2 v4.3.1 (無依賴)...")
!pip install --force-reinstall --no-deps ctranslate2==4.3.1
print("\nINFO: 安裝 onnxruntime-gpu v1.15.0 (無依賴)...")
!pip install --force-reinstall --no-deps onnxruntime-gpu==1.15.0

# 4. 安裝 faster-whisper
print("\nINFO: 安裝 faster-whisper...")
!pip install faster-whisper

# 5. 安裝 Google API、PDF 和 HTTP 請求相關函式庫
print("\nINFO: 安裝 gspread, pypdf, requests...")
!pip install gspread
!pip install pypdf
!pip install requests

print("\n--- 所有依賴包安裝指令已執行 ---")
```

### 3.3. Colab 執行階段重啟 (重要！)
*   在**首次**執行上述安裝儲存格後，或者當 `numpy` 或 GPU 相關庫被安裝/更新後，**必須手動重新啟動 Colab 執行階段**。
*   方法：在 Colab 菜單中選擇 **「執行階段」(Runtime) -> 「重新啟動執行階段」(Restart runtime)**。
*   **必須在安裝儲存格執行完畢後，再執行此重新啟動操作**，然後才能繼續運行 Python 腳本。

### 3.4. Google Drive 目錄結構
請在您的 Google Drive 中 `/MyDrive/` 路徑下，確保以下文件夾結構（如果腳本中定義的路徑是默認值）：
*   `/content/drive/MyDrive/input_audio/`：存放需要轉錄的音頻文件。 (對應 `local_transcriber.py` 的 `INPUT_AUDIO_DIR`)
*   `/content/drive/MyDrive/output_transcriptions/`：作為 `local_transcriber.py` 和 `sheets_gemini_processor.py` 輸出的根目錄。腳本會在此文件夾下自動創建子文件夾。 (對應 `OUTPUT_TRANSCRIPTIONS_ROOT_DIR` 或 `TRANSCRIPTIONS_ROOT_INPUT_DIR`)
*   `/content/drive/MyDrive/lecture_handouts/`：存放 PDF 講義，供 `sheets_gemini_processor.py` 中的 Gemini API 校對時參考。 (對應 `pdf_handout_dir`)

### 3.5. API 密鑰設置
*   對於 `sheets_gemini_processor.py` 中的 Gemini API 功能，您需要在 Google Colab 的 **Secrets (密鑰)** 功能中添加一個名為 `GEMINI_API_KEY` 的密鑰，其值為您的 Gemini API 金鑰。

## 4. 執行順序

推薦的執行流程如下：

1.  **環境設定儲存格**：執行包含所有依賴安裝命令的 Colab 儲存格。如果提示，請重新啟動執行階段。
2.  **運行 `local_transcriber.py`**：處理音頻文件，生成 `_normal.txt` 和 `.srt` 文件到 `output_transcriptions/[audio_basename]/`。
3.  **運行 `sheets_gemini_processor.py`**：讀取上述輸出，創建/更新 Google Sheets，並調用 Gemini API 進行校對。
4.  **（可選）運行 `text_segmenter_colab.py`**：如果您需要將校對後的文本按30分鐘切分，則在 `sheets_gemini_processor.py` 完成對應的電子表格處理後，運行此腳本。

## 5. 狀態持久化

*   `local_transcriber.py`：會在 `OUTPUT_TRANSCRIPTIONS_ROOT_DIR` 文件夾下創建一個 `.processed_audio_files.json` 文件，記錄已成功轉錄的音頻文件名。重新運行時會跳過這些文件。
*   `sheets_gemini_processor.py`：會在 `TRANSCRIPTIONS_ROOT_INPUT_DIR` 文件夾下創建一個 `.gemini_processed_state.json` 文件，記錄已成功完成 Gemini 校對的電子表格（以 `base_name` 標識）。重新運行時，對於已記錄的項目，會跳過 Gemini API 的調用和結果寫入步驟。

## 6. 日誌與註釋語言

*   本項目的 Python 腳本中的**日誌信息**和**代碼註釋**主要使用**中文**編寫。
