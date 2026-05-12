import argparse
import json
from openai import OpenAI

# 1. 設定您的 API Key
client = OpenAI(api_key="your_api_key_here")

# 2. 設定輸入與輸出檔案路徑
INPUT_FILE = "multi_turn_miss_param_zh_tw_function_mix0.jsonl"
OUTPUT_FILE = "classified_results_gpt5_mini_0.jsonl"

# 3. 最終完全體系統提示詞 (包含 7 大分類、強化版 Few-shot 與多重錯誤偵測)
SYSTEM_PROMPT = """
# Role: 對話資料品質審核專家 (Multi-turn Tool Use)

# Task:
全面掃描 Multi-turn 對話資料中的所有輪次。判定是否存在邏輯缺陷或幻覺。請務必找出「所有」潛在問題。
# 核心判定原則：
1. **字面一致性**：判定「重複詢問」的前提是，該參數內容曾精確出現在先前的 User 對話中。
2. **腦補判定**：若助理在 Tool Call 中填入了 User 沒提過、歷史紀錄也沒出現過的特定資訊（如 Email、時間），即判定為參數腦補。
3. **引導無罪**：若資訊不齊全，助理詢問缺失參數是高品質表現，嚴禁判錯。

# Classification Categories:
1. **Information Hallucination (預知未來)**: 在工具結果回傳前，提前在參數中填入該結果才有的具體細節。
   - *例：同時調用創建與更新，且更新參數已填入尚未生成的 ID。*
2. **State Tracking Error (狀態追蹤錯誤)**: 未能同步對話狀態。
   - *例：使用者已刪除紀錄，但後續查詢卻顯示該紀錄依然存在。*
3. **Entity & Parameter Brain-filling (實體與參數腦補)**: 憑空捏造對話未出現的人名、ID或必要參數。
   - *例：未報姓名卻出現「張靜」；未給地址卻自動填入「文化路456號」。*
4. **Redundant Clarification (已知訊息重複詢問)**: 無視使用者已提供的明確資訊，強行發起詢問。
   - *例：用戶已給日期範圍，模型仍問「結束日期是哪天？」。*
5. **Low-quality/Nonsensical Parameter (無意義參數)**: 填入佔位符或不合常理的簡化參數。
   - *例：訓練數據填寫「數據1」、「value1」，而非實際內容。*
6. **Logic & Inconsistency (邏輯與數據不一致)**: 模型回覆與工具數據矛盾，或數據推理錯誤。
   - *例：工具回傳金額加總錯誤，或趨勢判斷與數據相反。*
7. **Tool Interaction Violation (工具調用規範錯誤)**: 未處理 Missing Function 或調用不存在於清單中的工具。

# Few-shot Examples:

## Example 1: Information Hallucination (預知未來)
- **Context**: Assistant 調用 `set_character_dialogue` 時填入 `character_id: 1`。
- **Error**: 該 ID 應為同輪次中 `create_virtual_character` 執行後才產生，助理提前預知了結果。
- **Label**: Information Hallucination

## Example 2: Entity & Parameter Brain-filling (參數腦補)
- **Context**: User 說「幫我查詢牙科保險」，未提及姓名。
- **Error**: Assistant 調用工具時自動填入 `"patient_name": "張靜"`。
- **Label**: Entity & Parameter Brain-filling

## Example 3: Redundant Clarification (重複詢問)
- **Context**: User 說「幫我搜尋海灘.jpg」。
- **Error**: Assistant 卻問「請問您要搜尋的檔案名稱是什麼呢？」。
- **Label**: Redundant Clarification

## Example 4: Low-quality/Nonsensical Parameter (無意義參數)
- **Context**: User 請求訓練機器學習模型。
- **Error**: Assistant 調用 `train_ml_model` 時，數據填入 `[{"特徵": "數據1", "標籤": "正常"}]`。
- **Label**: Low-quality/Nonsensical Parameter

# Output Format (Strict JSON):
你必須針對每一筆資料回傳標準 JSON 物件。若有多個問題，請全部列出。
{
  "id": "原始資料的 ID",
  "has_error": true/false,
  "errors": [
    {
      "error_type": "分類名稱",
      "error_turn_index": [錯誤發生的輪次索引，從 0 開始],
      "reason": "具體描述原因，指出哪一輪出現了什麼邏輯問題。"
    }
  ]
}
# 注意：若無錯誤，"has_error" 填 false，"errors" 回傳空列表 []。
"""
def process_data_stream(input_path, output_path):
    # 使用 'a' (append) 模式可以方便中斷後續傳，或 'w' 重新開始
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8", buffering=1) as outfile: # buffering=1 開啟行緩衝
        
        count = 0
        for line in infile:
            if not line.strip(): continue
            try:
                raw_data = json.loads(line)
                data_id = raw_data.get("id", "unknown")
                
                # 過濾 tools 欄位，僅發送 messages 以節省 Token 並防止記憶體壓力
                payload = {"id": data_id, "messages": raw_data.get("messages", [])}
                
                response = client.chat.completions.create(
                    model="gpt-5-mini", 
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"分析 ID: {data_id}\n{json.dumps(payload, ensure_ascii=False)}"}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # 取得結果並確保 ID 正確
                result = json.loads(response.choices[0].message.content)
                result["id"] = data_id 
                
                # 【核心邏輯】一筆處理完立即寫入硬碟，不留存於記憶體
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush() # 強制將緩衝區數據寫入磁碟
                
                count += 1
                error_num = len(result.get("errors", []))
                print(f"[{count}] 已完成並寫入: {data_id} | 發現問題數: {error_num}")
                
            except Exception as e:
                print(f"Error processing {data_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn Tool Use 資料審核工具")
    
    # 定義參數
    parser.add_argument("--input", "-i", default=INPUT_FILE, help=f"輸入 JSONL (預設: {INPUT_FILE})")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help=f"輸出 JSONL (預設: {OUTPUT_FILE})")

    args = parser.parse_args()

    print(" 開始執行即時寫入任務...")
    print(f" 輸入檔案: {args.input}")
    print(f" 輸出路徑: {args.output}")

    # 執行處理
    process_data_stream(args.input, args.output)
    
    print(f" 任務完成！結果已即時儲存於: {args.output}")