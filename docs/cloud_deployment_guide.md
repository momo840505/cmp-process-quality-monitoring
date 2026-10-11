# Streamlit Cloud 部署與本機完整分析（CMP 專案）

## 先區分兩個版本

- **本機完整 `dashboard.py`：** 使用本機 `outputs/`，可查看 SPC、逐片 FDC、VM、R2R。GitHub 已排除 `outputs/`，因此直接部署 `dashboard.py` 會缺少 14 份報告。
- **公開 `streamlit_app.py`：** 保留 SPC、FDC、VM、R2R 和資料範圍五頁展示，以及代碼切換、R2R 輪數滑桿。實際數據來自**已驗證的彙總分析報告**；逐片 PHM 檔案、模型與品質標籤均未公開。R2R 可視化的 120 筆資料為**獨立人工模擬**。

公開展示版能在 Community Cloud 完整啟動與操作，但**不等於**把本機所有逐片追蹤功能搬上網。若日後要把逐片真實資料加入網頁，請先審查 PHM 原始與衍生資料的再散布授權、隱私及存取權限；不可只為了讓功能完整就將 `outputs/` 整包提交 GitHub。

## Windows 本機測試

在 `CMP-Process-QE` 根目錄：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_public_deployment.py" -v
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

應能切換 5 個頁籤、SPC/VM 匿名代碼組、FDC 圖表及 10～120 輪的模擬滑桿。

## 安全提交更新

```powershell
cd "$HOME\Desktop\CMP-Process-QE"
git status --short
git add README.md requirements.txt dashboard.py streamlit_app.py src/public_demo_data.py tests/test_public_deployment.py docs/cloud_deployment_guide.md docs/figures/ public_data/
git diff --cached --name-only
git diff --cached --check
```

**請人工檢查 staged 清單**：不應有 `outputs/`、`data/`、`.venv/`、第三方原始 ZIP、逐片晶圓 CSV、訓練模型 `joblib` 或 `.env`。另請確認公開的 `docs/figures` 圖表使用方式符合 PHM 資料來源條件；原始資料授權目前尚未完全核實。確認後：

```powershell
git commit -m "Add public Streamlit dashboard and project figures"
git push origin main
```

## 在 Streamlit Community Cloud 部署

1. 前往 https://share.streamlit.io/ 登入並連結 GitHub。
2. 右上角選 **Create app**，選擇既有 GitHub repository。
3. Repository：`momo840505/cmp-process-quality-monitoring`。
4. Branch：`main`。
5. Main file path（應用程式入口）：**`streamlit_app.py`**，**不要選 `dashboard.py`**。
6. 依畫面選擇 App URL（可選），按 Deploy，等候首次安裝。Cloud 預設會安裝專案根目錄 `requirements.txt` 的套件；程式完全不需要 API secrets。
7. 開啟實際產生的 `https://...streamlit.app`，驗證五個頁籤與變更不同群組、R2R 輪數後，才把這個網址加到 README。**尚未實際部署前不可虛構 Demo URL。**

若安裝出錯，在 Cloud 右下角或 App 管理頁讀取 **Manage app / Logs**。常見原因：`requirements.txt` 安裝失敗、main 路徑選錯、`public_data` 沒提交、GitHub 尚未同步。可用官方文件 https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy 逐項排查。

## 目前界線

- 不需要 DB、AWS、Render、API 密鑰；公開版從 repo 讀取極小的 JSON/模擬 CSV。
- 這只是公開的研究成果展示，沒有模型推論 API、真實感測串流或製程控制。
- Streamlit Community Cloud 可能有資源與休眠限制；連結能用不等於 24 小時生產級服務 SLA。
- 若未來需要完整逐片資料功能，應先確認授權，再設計受權限控制的資料來源與部署方式。
