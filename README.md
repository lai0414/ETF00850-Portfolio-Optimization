# ETF00850-Portfolio-Optimization
ETF00850成分股投資組合優化與滾動視窗回測系統

![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

這是一個完整的投資組合優化與回測框架，利用數學規劃（Mathematical Programming）與滾動視窗（Rolling Window）方法論，實作並比較五種核心投資策略。

## 📌 專案核心目標
1.  **策略對比**：比較「最大報酬」、「最小風險」、「風險收益平衡」、「等權重」與「定期定額」五種策略。
2.  **大盤基準 (Benchmark)**：以台灣加權股價指數 (**^TWII**) 為基準，檢驗超額報酬 (**Alpha**)。
3.  **穩定性驗證**：透過「滾動視窗 (**Rolling Window**)」模擬每半年調倉一次的真實交易行為，避免後見之明偏誤。

---

## 🛠 系統架構與邏輯拆解

### 1. 資料處理層 (Data Processing)
* **數據獲取**：使用 `quantmod` API 串接 Yahoo Finance，抓取 2016 至今的還原權值股價，確保回測考慮了配息因子。
* **資產相關性分析**：
    系統計算全期間日報酬的相關係數矩陣 $\rho_{i,j} = \frac{\text{cov}(r_i, r_j)}{\sigma_i \sigma_j}$。
    * **設計邏輯**：相關性是分散風險的核心。模型透過識別低相關資產（如防禦性的中華電與攻擊性的台積電）來優化風險邊界。



### 2. 優化模型層 (Optimization Engine)
本系統調用 `nloptr` 中的 **COBYLA** 演算法，在給定約束條件下求解非線性目標函數：

* **最大報酬 (Max Return)**：
    * 🎯 **目標**：$\max \sum (w_i \cdot \mu_i)$
    * ⚠️ **限制**：年化風險 $\sqrt{w^T \Sigma w} \leq 30\%$
    * *設計邏輯：在風險可接受範圍內，最大化資本利得，權重通常集中於高 Beta 動能股。*

* **最小風險 (Min Risk)**：
    * 🎯 **目標**：$\min \sqrt{w^T \Sigma w}$
    * ⚠️ **限制**：年化期望報酬 $\sum (w_i \cdot \mu_i) \geq 10\%$
    * *設計邏輯：尋找資產間的負相關性，抵銷系統波動，適合保守型投資規劃。*

* **平衡策略 (Trade-Off)**：
    * 🎯 **目標**：$\max (w' \mu - \sqrt{w' \Sigma w})$
    * ⚠️ **限制**：$\sigma_p \leq 25\%$ 且 $\mu_p \geq 15\%$
    * *設計邏輯：追求單位風險收益比（類似夏普比率）的最大化。*

### 3. 回測執行層 (Backtesting Framework)
採用專業級「滾動視窗」機制，模擬真實世界的市場適應力：
* **訓練窗 (Training)**：2 年數據（約 500 個交易日），用於估算回報向量 $\mu$ 與共變異矩陣 $\Sigma$。
* **測試窗 (Testing)**：0.5 年（半年），模擬真實持有表現，不使用未來資訊。
* **複利邏輯**：每期結束後的資產現值滾入下一期，精確計算累積財富走勢。

---

## 📊 關鍵績效指標與公式 (Performance Metrics)

| 指標 | 計算邏輯與公式 | 金融意義 |
| :--- | :--- | :--- |
| **Alpha (超額報酬)** | $\alpha = R_p - R_b$ | 策略贏過大盤指數的百分點，衡量「實力」收益。 |
| **Sharpe Ratio** | $\frac{E[R_p]}{\sigma_p} \cdot \sqrt{2}$ | 每承擔一單位總風險所換取的報酬。 |
| **Information Ratio** | $\frac{Avg(R_p - R_b)}{SD(R_p - R_b)} \cdot \sqrt{2}$ | 衡量超越大盤的穩定度，IR 越高代表勝過大盤並非偶然。 |
| **Max Drawdown** | $\min \left( \frac{\text{Wealth}_t - \text{Peak}_t}{\text{Peak}_t} \right)$ | 衡量策略最差情況下的抗跌能力（壓力測試）。 |
| **p-value (t-test)** | 配對樣本 $t$ 檢定 | 統計學意義上的「測謊機」，判斷 Alpha 是否顯著大於 0。 |

---

## 📈 實際產出結果分析 (Visual Analysis)

### 1. 財富累積走勢 (**01_wealth_performance.png**)
展示策略與 **^TWII (大盤)** 的長期增長對比。
* **產出分析**：若策略曲線斜率持續大於大盤，代表優化模型有效捕捉到 Alpha。觀察 2022 年大盤重挫時，各策略的回檔深度可判斷其防守特質。



### 2. 超額報酬 (Alpha) 與勝率 (**02_alpha_comparison.png**)
* **分析**：直方圖展示各策略相對大盤的年化溢酬。若 **Equal_Weight** 的 Alpha 高達 10% 且 p-value < 0.05，說明「分散投資 + 定期再平衡」具備強大實力。

### 3. 風險-報酬散佈圖 (**05_risk_return_scatter.png**)
將策略標註於風險(X軸)與報酬(Y軸)平面。
* **分析**：越往「左上方」移動的策略越理想。理想的優化策略應落在「效率前緣 (Efficient Frontier)」上，提供比大盤更高的回報，同時維持更低的風險。



### 4. 回撤分析 (**07_drawdown_vs_benchmark.png**)
* **分析**：比較 2022 年空頭市場各策略的抗震力。**Min_Risk** 策略因配置大量防禦股，其回撤幅度通常遠小於大盤的 -28.5%，標註點可識別最大壓力位置。

---

## 💡 總結分析結論
* **再平衡的價值**：`Equal Weight` 策略透過每半年強制將權重調回 1/N，實現了「高賣低買」的自動化獲利，Sharpe Ratio 往往領先。
* **選股勝於一切**：選出的 15 檔標的均為優質龍頭，整體素質優異，因此多數策略皆能產生顯著的正向 Alpha。
* **低波防禦力**：`Min_Risk` 策略雖年化報酬較低，但其極低的 MDD (-3.82%) 證明了其在空頭市場中作為「避風港」的極佳價值。

---

## 🚀 如何使用
1.  **安裝必要套件**：
    ```r
    install.packages(c("quantmod", "nloptr", "dplyr", "tidyr", "lubridate", "scales", "ggplot2"))
    ```
2.  **執行步驟**：直接在 R 環境中執行腳本，系統將自動下載數據並執行 16 輪滾動優化。
3.  **獲取結果**：查看 `Portfolio_Backtest_Results` 資料夾，內含所有 CSV 數據報表與高清分析圖表。
