# Step 08C｜GitHub 公開前檢查與第一次提交

此文件供已完成 Step 01～08B 的專案使用。**這一步只整理說明文件及 Git 追蹤清單；不重新訓練模型、修改原始輸出或重跑已凍結的 06D／07C。**

## 0. 確認文件

本工具包只會提供 `README.md`、`.gitignore`、`docs/08c_publication_guide.md`。原有 `src/`、`tests/`、`dashboard.py` 和 `outputs/` 不需要覆蓋。若你採用 `Expand-Archive -Force`，先備份舊 README 與 `.gitignore`。

先看新的 README 是否符合你實際跑出的數據：

```powershell
cd "$HOME\Desktop\CMP-Process-QE"
Get-Content .\README.md -Encoding UTF8
Get-Content .\.gitignore -Encoding UTF8
```

## 1. 建立 Git（不會上傳網路）

```powershell
git --version
git init
git status --short
```

如果之前已經執行過 `git init`，不要反覆建立新的目錄；直接從下一段繼續。

## 2. 查看排除規則是否真的有效

```powershell
git check-ignore -v .venv/ outputs/ outputs/models/ data/raw/ "CMP_Process_QE_Step08B.zip"
```

這個命令可用來測試**指定路徑是否符合忽略規則**，即使該路徑目前不存在也可檢查。若沒有顯示某個路徑，應檢查拼寫與規則，不能單憑沒有輸出認定安全。特別是 `.venv`、`outputs` 與資料夾 `data` 不應進入 staged 檔案。

## 3. 只加入有意公開的內容

**不要先使用 `git add .`。** 本專案根目錄有二十多份學習階段的 `README_STEP...md`，先不全部提交。建議明確加入：

```powershell
git add README.md .gitignore requirements.txt requirements_dashboard.txt dashboard.py

git add src/ tests/ docs/

git diff --cached --name-only
```

注意：`git add docs/` 會加入 `docs/` 內目前所有**未被排除**的檔案。若其中含不想公開的文件，改用逐檔 `git add`，或先移出該資料夾。

請逐項檢查 staged 清單，確定沒有 `.venv/`、`outputs/`、`data/`、`.env`、大型模型、第三方研究資料及下載壓縮檔。

可進一步核對：

```powershell
git diff --cached --stat

git diff --cached --check

git status --short
```

`git diff --cached --check` 檢查的是空白等格式問題，**不是**洩密掃描；Git 無法保證所有原始碼都沒有密碼或金鑰。提交前仍要人工檢查，包括 GitHub token、API key、密碼、個人資料和本機絕對路徑。

## 4. 檢查完成後才 commit

確定清單及 README 無誤後：

```powershell
git commit -m "Add CMP process analysis and dashboard"

git status
```

若 Git 提示尚未設定身分，依提示設定 `git config user.name`、`git config user.email`，設定後再 commit。不要把 GitHub access token 寫進程式或 README。

## 5. 建立 GitHub 遠端倉庫（等確認後）

在 GitHub 網站建立新的空白 repository，**不要勾選自動加入 README、.gitignore 或 License**（因為本地已有 README 和 `.gitignore`，而你目前還沒有確定資料及程式的完整授權處理方式）。取得自己新建的 repo URL 後，才執行：

```powershell
git branch -M main

git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git

git push -u origin main
```

`YOUR_ACCOUNT`、`YOUR_REPO` 是示意字串，必須換成你建立的真實倉庫位址。如果 `origin` 已存在，先以 `git remote -v` 查核，不要重複 `remote add`。

## 6. 公開作品的可重現性限制

因為 `outputs/` 被排除，新的 GitHub 訪客**無法下載後直接啟動有完整數據的 Dashboard**。主 README 已提供重建順序。若未來想提供線上展示，應先決定可合法公開的最小衍生示範資料或另行製作與真實 PHM 數據明確區分的模擬展示資料，並修改 Dashboard 的資料讀取流程；**目前尚未實作公開免資料 Demo**。

此外，PHM 比賽的公開網站／第三方 mirror 和授權資訊必須持續核對；本儲存庫沒有聲稱自己具有原始資料的再散布權。

## 7. 下次工作

- 可把根目錄中的 `README_STEP...md` 整理至 `docs/steps/`，保留它們作為學習筆記；不要直接刪除。
- 可以在 `docs/figures/` 放少量經確認可公開的自製分析圖片，以便 GitHub README 直接展示，不要上傳全部輸出。
- 介面欄位仍有部分程式英文標籤，可在不更動統計成果的情況下，個別改善 Dashboard 文字。
