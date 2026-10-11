# Step 08B：整合驗收（中文學習筆記）

## 為什麼做這一步？

之前每一階段各自驗證過，但是最後上傳 GitHub 前，要確定各報告仍然能互相對應：

- **資料守恆**：原始列數 = 衍生列數 + 完全去重列數；品質結果不能平白消失。
- **資料洩漏檢查**：模型特徵目錄和已凍結的模型設定，不得出現 MRR 答案、晶圓 ID 或測試分組。
- **實驗凍結**：Step 06D 最終內部 Test、Step 07C 模擬已完成，不能為了變漂亮而重跑選成績。
- **報告一致性**：Dashboard 是唯讀的展示，不是新的一輪實驗。
- **發布檢查**：源碼可公開，但原始數據、模型二進位、密鑰、私人設定不可不經審查直接提交 GitHub。

## 執行

```powershell
cd "$HOME\Desktop\CMP-Process-QE"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_08b_integrated_audit.py" -v
.\.venv\Scripts\python.exe .\src\08b_integrated_audit.py
```

`PASS` 表示一致性驗證通過；`WARN` 是公開前要處理的事項；`FAIL` 表示關鍵檔案缺少或互相矛盾，須先修復再公開。

驗收程式只會新增兩份文件：`outputs/audit/08b_integration_audit.json` 與 `.md`。不改動 Git、不覆蓋舊結果，也不重新訓練模型。

## 如何在面試說明？

「我除了完成 SPC、FDC、VM 與 R2R 的分析，還做了跨階段資料一致性檢查。像是每片晶圓品質結果只能出現一次、模型的 Test 結果不能因重新調參而更改、R2R 要確認使用同樣的人工干擾，避免把模擬成果誤說成真實產線改善。」
