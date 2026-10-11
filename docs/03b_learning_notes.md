# Step 03B — 追查 Stage A 的兩群 MRR

## 目的
Step 03A 看到 Stage A 的 MRR 有兩群。這一步不是找「好晶圓與壞晶圓」，而是調查：兩群是否對應不同的設備位置/訊號代碼？如果是，應先依條件分組，再評估 SPC 管制基準的適用性。

## 先複習三個觀念
- 一筆品質值：`WAFER_ID + STAGE` 對應一筆 MRR。
- 多筆感測值：同一晶圓可跨多個 `MACHINE_DATA` 代碼與時間點。
- `MACHINE_DATA` 是公開資料內的匿名代碼；**不能擅自宣稱是某種確切 Recipe 或不同機台**。

## 安裝、執行
在 PowerShell 中開啟 `C:\Users\momo8\Desktop\CMP-Process-QE`，把 Step03B ZIP 解壓縮在專案根目錄：

```powershell
cd "$HOME\Desktop\CMP-Process-QE"
Expand-Archive -Path ".\CMP_Process_QE_Step03B.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe .\src\03b_investigate_stageA_modes.py
```

程式預設讀取：
- `outputs/processed/02b_wafer_stage_quality.csv`
- `outputs/processed/02b_sensor_events_training.csv.gz`

## 輸出
- `outputs/analysis/03b_stageA_wafer_code_mrr.csv`：一片晶圓階段一筆，MRR 與代碼組合
- `outputs/analysis/03b_code_group_vs_mrr_band.csv`：代碼組 vs MRR 描述區間（交叉表）
- `outputs/analysis/03b_sensor_activity_by_group.csv`：每組晶圓的感測數值非零比例分布
- `outputs/analysis/03b_investigation_summary.json`：所有檢核與解釋警語
- `outputs/analysis/figures/03b_stageA_mrr_by_code_group.png`
- `outputs/analysis/figures/03b_stageA_record_order_by_code_group.png`

## 圖表怎麼看
- **圖 1**：X 軸是根據感測資料算出來的代碼組，Y 軸是 MRR。分布差異只能支持「兩組條件相關」，不能支持「這個代碼造成 MRR 差異」。
- **圖 2**：X 軸依最早感測時間排序，不是確定的日曆生產日期。主要看兩群有無隨紀錄順序交錯出現。

## 特別重要的資料分析規則
1. `MRR < 90`、`135 <= MRR < 1000`、`MRR >= 1000` 都只是 Training EDA 的**描述區間**。不是工廠規格界限、異常真值、模型推論結果。
2. `MACHINE_DATA` 分組來自感測資料；未來若評估預測模型，不可將**事後看 MRR 才分出的群標籤**當作輸入。
3. 感測器數值為零，可能是機台階段、量測通道或編碼問題；不能直接說「機台故障」。
4. 四筆大於 1000 的結果都保留。圖表為便於顯示只把它們放在可視範圍外，不會修改 CSV。
5. 本步驟只用 Training 資料，不讀 Test 或 Validation 的 MRR 做決策。

## 下一步
核對本次結果後，再評估同一群內是否有足夠穩定的基準來建立合理的 SPC 圖。**切記 SPC 控制界限不等於工程規格 USL/LSL。**
