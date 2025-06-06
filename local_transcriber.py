import os
import datetime
from faster_whisper import WhisperModel # 已修正導入
import logging
import json
# google.colab.drive 將在主函數中有條件地導入，用於掛載

# --- 配置變數 ---
MODEL_SIZE = "large-v3"
DEFAULT_INITIAL_PROMPT = "這是佛教關於密教真言宗藥師佛" # 初始提示詞的默認值
INPUT_AUDIO_DIR = "/content/drive/MyDrive/input_audio"
OUTPUT_TRANSCRIPTIONS_ROOT_DIR = "/content/drive/MyDrive/output_transcriptions" # 新子目錄的根目錄
STATE_FILE_PATH = os.path.join(OUTPUT_TRANSCRIPTIONS_ROOT_DIR, ".processed_audio_files.json") # 狀態檔案路徑


# --- 輔助函數 ---
def load_processed_files(state_file_path, logger): # logger instance passed for logging within function
    # 從狀態檔案載入已處理的檔案名稱集合
    try:
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 確保是列表，轉換為集合以便高效查找
                return set(data if isinstance(data, list) else [])
        logger.info(f"狀態檔案 '{state_file_path}' 未找到。將從頭開始處理。")
    except json.JSONDecodeError:
        logger.warning(f"解碼狀態檔案 '{state_file_path}' 時發生錯誤。將從頭開始處理。")
    except Exception as e:
        logger.error(f"載入狀態檔案 '{state_file_path}' 時發生錯誤: {e}。將從頭開始處理。", exc_info=True)
    return set()

def save_processed_files(state_file_path, processed_files_set, logger): # logger instance passed
    # 將已處理的檔案名稱集合儲存到狀態檔案
    temp_state_file_path = state_file_path + ".tmp" # 使用臨時檔案以確保原子性寫入
    try:
        # 確保父目錄存在
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        with open(temp_state_file_path, 'w', encoding='utf-8') as f:
            # 將集合轉換為列表以便 JSON 序列化
            json.dump(list(processed_files_set), f, ensure_ascii=False, indent=4)
        # 原子性重命名 (在 POSIX 系統上)
        os.replace(temp_state_file_path, state_file_path)
        logger.debug(f"狀態已成功儲存至 '{state_file_path}'。")
    except Exception as e:
        logger.error(f"儲存狀態至 '{state_file_path}' 時發生錯誤: {e}", exc_info=True)
        if os.path.exists(temp_state_file_path):
            try:
                os.remove(temp_state_file_path)
            except OSError as oe:
                logger.error(f"移除臨時狀態檔案 '{temp_state_file_path}' 時發生錯誤: {oe}", exc_info=True)

def format_srt_time(seconds):
    """將秒數格式化為 SRT 格式的時間字串 (HH:MM:SS,ms)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def main():
    # --- 日誌配置 (使用具名 logger) ---
    logger = logging.getLogger('LocalTranscriberLogger')
    logger.setLevel(logging.INFO)

    # 檢查 logger 是否已有 handlers，防止重複添加 (在 Colab 中多次運行儲存格時可能發生)
    if not logger.handlers:
        # 創建控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 創建 formatter 並添加到 handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # 將 handler 添加到 logger
        logger.addHandler(ch)

    # 可選：阻止日誌消息傳播到 root logger，如果擔心 root logger 的配置
    # logger.propagate = False

    logger.info("local_transcriber.py 腳本已啟動。")

    # --- 獲取用戶輸入的初始提示詞 ---
    user_prompt_input = input(f"請輸入 Whisper 轉錄時使用的初始提示詞 (默認值: '{DEFAULT_INITIAL_PROMPT}'): ")
    current_initial_prompt = user_prompt_input if user_prompt_input else DEFAULT_INITIAL_PROMPT
    logger.info(f"將使用以下初始提示詞進行轉錄: '{current_initial_prompt}'")

    # --- 載入狀態 ---
    processed_files = load_processed_files(STATE_FILE_PATH, logger) # Pass logger
    logger.info(f"從狀態檔案 '{STATE_FILE_PATH}' 載入了 {len(processed_files)} 個已處理檔案的記錄。")

    # --- 掛載 Google Drive ---
    logger.info("嘗試掛載 Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        logger.info("Google Drive 掛載成功。")
    except ImportError:
        logger.warning("跳過 Drive 掛載，因為不在 Colab 環境或 google.colab 不可用。")
        # 取決於 Drive 是否絕對必要，您可能需要在此處退出
        # 對於沒有 Colab 的本地執行，INPUT_AUDIO_DIR 需要是本地路徑
    except Exception as e:
        logger.error(f"掛載 Google Drive 時發生錯誤: {e}", exc_info=True)
        # 如果 Drive 至關重要且掛載失敗，則可能需要退出
        return # 如果 Drive 掛載失敗且被認為是關鍵操作，則退出

    # --- 加載 Faster Whisper 模型 ---
    logger.info(f"正在加載 Faster Whisper 模型: {MODEL_SIZE}...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        logger.info("Faster Whisper 模型加載成功。")
    except Exception as e:
        logger.error(f"加載 Faster Whisper 模型時發生錯誤: {e}", exc_info=True)
        logger.error("此腳本需要支持 CUDA 的 GPU 和相應的庫。")
        logger.error("請確保已正確安裝 PyTorch 和支持 CUDA 的 CTranslate2。")
        return

    # --- 檢查輸入目錄 ---
    if not os.path.exists(INPUT_AUDIO_DIR):
        logger.error(f"輸入音頻目錄 '{INPUT_AUDIO_DIR}' 未找到。正在退出。")
        return

    logger.info(f"輸入音頻目錄: {INPUT_AUDIO_DIR}")
    logger.info(f"輸出根目錄: {OUTPUT_TRANSCRIPTIONS_ROOT_DIR}")
    logger.info(f"狀態檔案路徑: {STATE_FILE_PATH}")

    # --- 遍歷音頻檔案 ---
    audio_files_to_process = [f for f in os.listdir(INPUT_AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.mp4'))]

    if not audio_files_to_process:
        logger.info(f"在 '{INPUT_AUDIO_DIR}' 中未找到任何音頻檔案。")
        return

    logger.info(f"找到 {len(audio_files_to_process)} 個音頻檔案待處理。")

    unwanted_phrase = "字幕由 Amara.org 社群提供" # 根據原始腳本

    for audio_file_name in audio_files_to_process:
        # 使用 audio_file_name 作為已處理狀態的唯一標識符
        if audio_file_name in processed_files:
            logger.info(f"跳過 '{audio_file_name}'，因為它先前已被處理。")
            continue

        base_name = os.path.splitext(audio_file_name)[0]
        audio_path = os.path.join(INPUT_AUDIO_DIR, audio_file_name)

        logger.info(f"--- 正在處理檔案: {audio_path} ---")

        # --- 轉錄 ---
        try:
            logger.info(f"開始轉錄檔案: {audio_file_name}...")
            # VAD 參數來自原始腳本 (可以設為可配置)
            vad_parameters = {
                "min_speech_duration_ms": 50,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 500,
            }
            segments_generator, info = model.transcribe(
                audio_path,
                beam_size=5,
                initial_prompt=current_initial_prompt, # 使用用戶定義或默認的提示詞
                vad_filter=True,
                vad_parameters=vad_parameters
            )
            segments_list = list(segments_generator) # 使用生成器獲取列表
            logger.info(f"檔案 '{audio_file_name}' 轉錄完成。語言: {info.language}，概率: {info.language_probability:.2f}")
            logger.info(f"為 '{audio_file_name}'檢測到 {len(segments_list)} 個片段。")

        except Exception as e:
            logger.error(f"檔案 '{audio_file_name}' 轉錄過程中發生錯誤: {e}", exc_info=True)
            continue # 跳到下一個檔案

        # --- 生成 "一般文本" ---
        whisper_transcription_lines = []
        for segment in segments_list:
            cleaned_text = segment.text.strip().replace(unwanted_phrase, "").strip()
            if cleaned_text: # 清理後有實際文本才添加
                whisper_transcription_lines.append(cleaned_text)

        normal_text_content = "\n".join(whisper_transcription_lines)
        # logger.debug(f"\n檔案 '{base_name}' 的一般文本預覽 (前100字符):\n'{normal_text_content[:100]}...'") # 用於調試

        # --- 生成 SRT 內容 ---
        srt_content = ""
        srt_sequence_number = 1
        for segment in segments_list:
            start_time_srt = format_srt_time(segment.start)
            end_time_srt = format_srt_time(segment.end)
            cleaned_segment_text = segment.text.strip().replace(unwanted_phrase, "").strip()

            if cleaned_segment_text: # SRT 中僅包含有實際文本的片段
                srt_content += f"{srt_sequence_number}\n"
                srt_content += f"{start_time_srt} --> {end_time_srt}\n"
                srt_content += f"{cleaned_segment_text}\n\n"
                srt_sequence_number += 1
        # logger.debug(f"\n檔案 '{base_name}' 的 SRT 內容預覽 (前100字符):\n'{srt_content[:100]}...'") # 用於調試


        # --- 輸出到檔案 ---
        output_dir_for_file = os.path.join(OUTPUT_TRANSCRIPTIONS_ROOT_DIR, base_name)
        try:
            os.makedirs(output_dir_for_file, exist_ok=True)
        except OSError as e:
            logger.error(f"創建輸出目錄 {output_dir_for_file} 時發生錯誤: {e}", exc_info=True)
            continue # 如果目錄創建失敗，跳到下一個檔案

        normal_text_filename = f"{base_name}_normal.txt"
        normal_text_path = os.path.join(output_dir_for_file, normal_text_filename)

        srt_filename = f"{base_name}.srt"
        srt_path = os.path.join(output_dir_for_file, srt_filename)

        try:
            with open(normal_text_path, "w", encoding="utf-8") as f:
                f.write(normal_text_content)
            logger.info(f"一般文本已成功寫入: {normal_text_path}")
        except IOError as e:
            logger.error(f"寫入一般文本至 {normal_text_path} 時發生錯誤: {e}", exc_info=True)

        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            logger.info(f"SRT 字幕已成功寫入: {srt_path}")
        except IOError as e:
            logger.error(f"寫入 SRT 字幕至 {srt_path} 時發生錯誤: {e}", exc_info=True)
            continue # 如果檔案寫入失敗，則不標記為已處理

        # 如果此檔案的所有輸出都已成功保存，則標記為已處理
        processed_files.add(audio_file_name)
        save_processed_files(STATE_FILE_PATH, processed_files, logger) # Pass logger
        logger.info(f"已將 '{audio_file_name}' 標記為已處理並更新狀態檔案。")

    logger.info("所有音頻檔案處理完畢。")
    logger.info("local_transcriber.py 腳本已完成。")

if __name__ == "__main__":
    main()
