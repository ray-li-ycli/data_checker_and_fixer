import json
from openai import OpenAI
import argparse
# ================= 1. 初始化 Client =================
client = OpenAI(api_key="your_api_key_here")


def get_gpt_surgical_correction_with_diff(combined_reasons, messages_segment):
    system_prompt = """你是一個嚴格的資料修補專家。
你的任務是根據診斷理由修正對話錯誤，並確保整段對話的邏輯連貫性。

[修正策略]：
1. **全量輸出**：請務必回傳完整的對話清單（所有輪次），包含修正過與未修正的部分。
2. **連鎖修正**：若修正某一輪會導致後續輪次（Turn N+1...）的邏輯、術語或脈絡出現斷層，你必須同步微調後續內容以維持一致。
3. **資訊編織**：若為「腦補/預知未來」，必須將該資訊回溯編織進第一輪 User Query。
4. **嚴禁無關改動**：除了為了「修補錯誤」與「維持邏輯連貫」外，禁止改動原始對話的語氣、專有名詞、標點格式。
5. **Function 標記**：若你的改動涉及 `tool_calls` 或 `function` 內容（如修改參數名、補齊參數、變更工具名稱），請務必在 `changes` 中以 "FUNCTION_CHANGE:" 開頭描述。

[輸出要求]：
請務必回傳 JSON，包含：
1. "fixed_messages": 修正且調整連貫性後的完整對話清單。
2. "changes": 描述你改了哪幾輪，以及改了什麼（例如: "Turn 0: 補齊資訊; Turn 3: 配合 Turn 0 調整術語"）。若動到 Function，格式為 "Turn X: FUNCTION_CHANGE: [具體改動內容]"""

    user_prompt = f"[診斷理由]:\n{combined_reasons}\n\n[對話歷史]:\n{json.dumps(messages_segment, ensure_ascii=False)}"

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", # 
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f" API 錯誤: {e}")
        return None

def run_ultimate_refinement_with_diff(dataset_path, error_log_path, output_path, diff_log_path, correct_path, missing_path):
    print(" 啟動精細化分流修正管道...")
    
    # 讀取診斷日誌
    with open(error_log_path, 'r', encoding='utf-8') as f:
        # 轉為字串 ID 確保匹配穩定
        diagnostics = {str(json.loads(line)['id']): json.loads(line) for line in f}

    with open(dataset_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out, \
         open(diff_log_path, 'w', encoding='utf-8') as f_diff, \
         open(correct_path, 'w', encoding='utf-8') as f_correct, \
         open(missing_path, 'w', encoding='utf-8') as f_missing:

        count_fixed = 0
        count_correct = 0
        count_missing = 0

        for line in f_in:
            data = json.loads(line)
            doc_id = str(data.get('id'))
            
            # 分流邏輯
            if doc_id not in diagnostics:
                # --- 狀況 A: 找不到日誌 ---
                f_missing.write(json.dumps(data, ensure_ascii=False) + '\n')
                count_missing += 1
                continue

            report = diagnostics[doc_id]
            
            if report.get('has_error') is False:
                # --- 狀況 B: 診斷無誤 ---
                f_correct.write(json.dumps(data, ensure_ascii=False) + '\n')
                count_correct += 1
                continue

            # --- 狀況 C: 有錯誤，執行修正 ---
            combined_reasons = "\n".join([f"Turn {e['error_turn_index']}: {e['reason']}" for e in report.get('errors', [])])
            messages_to_fix = data['messages'] 

            result = get_gpt_surgical_correction_with_diff(combined_reasons, messages_to_fix)

            if result and 'fixed_messages' in result:
                if len(result['fixed_messages']) >= len(data['messages']):
                    data['messages'] = result['fixed_messages']
                    
                    # 紀錄 Diff
                    changes = result.get('changes', [])
                    f_diff.write(json.dumps({
                        "id": doc_id,
                        "changes_made": changes,
                        "needs_tool_update": any("FUNCTION_CHANGE" in str(c) for c in changes)
                    }, ensure_ascii=False) + '\n')
                    
                    # 寫入結果
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count_fixed += 1
                    print(f" ID: {doc_id} 修復完成")
                    continue

            # 如果修正失敗（例如 API 斷線或回傳格式不對），暫時存入缺失/錯誤區
            f_missing.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f" ID: {doc_id} 修正過程發生異常")

    print(f"\n 處理完畢！統計結果：")
    print(f"- 成功修正: {count_fixed} 筆")
    print(f"- 診斷無誤: {count_correct} 筆")
    print(f"- 日誌缺失: {count_missing} 筆")

# ================= 執行調用 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動精細化分流修正管道")
    
    # 定義參數，預設值可以設為 None 或者你原本的檔名格式
    parser.add_argument("--suffix", type=str, default="2", help="檔案名稱的後綴編號 (例如: 2)")
    
    args = parser.parse_args()
    s = args.suffix  # 取得輸入的編號

    # 使用 f-string 動態組合檔名
    run_ultimate_refinement_with_diff(
        dataset_path=f'multi_turn_miss_param_zh_tw_function_mix_{s}.jsonl',
        error_log_path=f'classified_results_gpt5_mini_{s}.jsonl',
        output_path=f'./final_refined_dataset/final_refined_dataset_{s}.jsonl',
        diff_log_path=f'./diff_log/diff_log_{s}.jsonl',
        correct_path=f'./verified_correct/verified_correct_{s}.jsonl',
        missing_path=f'./not_found_in_log/not_found_in_log_{s}.jsonl'
    )