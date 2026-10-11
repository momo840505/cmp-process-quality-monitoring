# Step 02A — 資料重複、時間戳與晶圓品質配對檢查

這個步驟只是診斷，不是真正清理。先找出異常，再決定規則，避免刪除有效資料。

## 粒度
- Sensor event：每一列為一筆原始設備感測紀錄。可能有同一時間戳的不同狀態紀錄。
- Wafer-stage：`WAFER_ID + STAGE`，對應一筆 MRR 品質結果。不可以把多筆 sensor event 當成多個獨立 MRR 樣本。

## 重複與時間
- 完全重複：原始 25 欄都一樣，標記但暫不刪除。
- 同時間不同值：相同 wafer、stage、machine、MACHINE_DATA、chamber、TIMESTAMP，但其他欄不同。可能是合法多筆訊號，**不能按時間戳直接刪除**。
- 時間倒退：只在同一檔案且同一通道原始順序下檢查。日後趨勢分析必須先訂出正確時間順序。

## 品質指標
- AVG_REMOVAL_RATE：材料去除率（MRR），不是已知規格內/規格外標籤。
- `MRR > 1000` 只是待查標記，不是 USL。

## 輸出檔
- `02_diagnostics_summary.json`：各資料組檢查結果
- `02_wafer_stage_inventory.csv`：每個晶圓階段一列、檢查筆數與品質量測
- `02_exact_duplicate_examples.csv`：完全重複資料位置的樣例
- `02_same_timestamp_different_values_examples.csv`：同時間不同感測值的樣例

## 下一階段才做
先核對異常值和相同時間戳的內容，再決定是否去重、如何排序、如何在訓練/測試做晶圓分組切分；不能預先用測試品質結果設定篩選規則。
