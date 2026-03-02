# ==============================================================================
# ETF00850 Portfolio Optimization with Rolling Window Backtest（Python 版）
# 投資組合優化：ETF00850成分股投資組合優化與滾動視窗回測系統
# ==============================================================================
# 安裝需求（執行前請先安裝）：
#   pip install yfinance scipy pandas numpy matplotlib seaborn python-dateutil
#
# 五種策略：
#   1. Max_Return  ── 最大化報酬（年化風險 < 30%）
#   2. Min_Risk    ── 最小化風險（年化報酬 > 10%）
#   3. Trade_Off   ── 最大化（報酬 - 風險）（風險 < 25% 且 報酬 > 15%）
#   4. Equal_Weight── 等權重（每檔 1/15）
#   5. DCA         ── 定期定額（每月 5,000 元）
# ==============================================================================
#%%
# ── 載入套件 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize, Bounds

warnings.filterwarnings("ignore")

# ── 中文字型設定 ──────────────────────────────────────────────────────────────
# Windows: "Microsoft JhengHei" | Mac: "Arial Unicode MS" | Linux: "Noto Sans CJK TC"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS",
                                    "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

#%% ==============================================================================
# 一、基礎配置
# ==============================================================================

# 從 ETF 00850 成分股中挑選 15 檔標的
STOCKS = ["2330.TW", "2317.TW", "2454.TW", "3231.TW", "2357.TW",
          "2379.TW", "2327.TW", "2881.TW", "2891.TW", "2885.TW",
          "2890.TW", "1102.TW", "2618.TW", "1216.TW", "2412.TW"]

STOCK_NAMES = ["台積電", "鴻海", "聯發科", "緯創", "華碩",
               "瑞昱", "國巨", "富邦金", "中信金", "元大金",
               "永豐金", "亞泥", "長榮航", "統一", "中華電"]

# 中文名稱對照字典（用於圖表標籤）
NAME_MAP = dict(zip(STOCKS, STOCK_NAMES))
N_ASSETS = len(STOCKS)

START_DATE     = "2016-01-01"   # 下載起始日（訓練期需往前推 2 年）
END_DATE       = "2026-01-01"   # 下載結束日
MONTHLY_INVEST = 5000           # DCA 每月投入金額（元）

#%% ==============================================================================
# 二、資料下載與前處理
# ==============================================================================

def get_stock_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    從 Yahoo Finance 下載調整後收盤價（Adjusted Close）。

    使用 auto_adjust=True 自動還原股利與股票分割，
    確保歷史價格序列可直接計算真實報酬率，不因除權息而失真。
    對應 R 的 Ad()（Adjusted Price）函數。

    回傳：
        DataFrame，index 為交易日，欄位為各 ticker 代碼
    """
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    # yfinance 回傳 MultiIndex，取 Close 層（auto_adjust 後即為還原收盤價）
    prices = raw["Close"][tickers] if isinstance(raw.columns, pd.MultiIndex) \
             else raw[["Close"]]
    return prices.dropna()


print("下載股票資料中...")
price_matrix = get_stock_data(STOCKS, START_DATE, END_DATE)
price_matrix.index = pd.to_datetime(price_matrix.index)

print("下載大盤資料（^TWII）中...")
benchmark_raw = get_stock_data(["^TWII"], START_DATE, END_DATE)
benchmark_raw.columns = ["Benchmark"]
benchmark_raw.index = pd.to_datetime(benchmark_raw.index)

# ── 計算日報酬率 ──────────────────────────────────────────────────────────────
# 公式：R_t = P_t / P_{t-1} - 1  ≡  (今天 - 昨天) / 昨天
# pct_change() 對應 R 的 lag() 寫法；第一列為 NaN，dropna() 自動移除
data_returns   = price_matrix.pct_change().dropna()
benchmark_rets = benchmark_raw.pct_change().dropna()

print(f"\n✓ 成功下載 {N_ASSETS} 檔股票，"
      f"期間 {price_matrix.index[0].date()} ～ {price_matrix.index[-1].date()} "
      f"（共 {len(price_matrix)} 個交易日）\n")

#%% ==============================================================================
# 三、相關係數矩陣視覺化
# ==============================================================================

# 計算全期間日報酬率相關係數矩陣
# 用於了解資產間的共同波動程度，作為分散效果的參考
cor_matrix = data_returns.corr()

# 軸標籤：「代碼\n中文名」（對應 R 的 paste(Var1, Name1)）
labels = [f"{t}\n{NAME_MAP[t]}" for t in STOCKS]

fig0, ax0 = plt.subplots(figsize=(14, 11))
sns.heatmap(cor_matrix, annot=True, fmt=".2f",
            cmap="RdBu_r",          # 藍（低相關）→ 紅（高相關）
            vmin=0, vmax=1, center=0.5,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax0,
            annot_kws={"size": 8})
ax0.set_title("Asset Correlation Matrix (00850 Components)\n"
              "Based on daily returns (2016-2025)",
              fontsize=13, fontweight="bold", pad=14)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

#%% ==============================================================================
# 四、投資組合優化函數（對應 R 的 nloptr + COBYLA）
# ==============================================================================

def run_optimization(train_ret: pd.DataFrame, opt_type: str,
                     threshold1: float = None,
                     threshold2: float = None) -> np.ndarray:
    """
    非線性投資組合優化（使用 scipy SLSQP，功能對應 R 的 COBYLA）。

    數學模型
    ─────────────────────────────────────────────────────────────────
    符號定義：
        w      = 權重向量（n×1），wi ≥ 0，Σwi = 1
        μ      = 年化期望報酬向量（複利）：(1 + r̄)^af - 1
        Σ      = 年化共變異矩陣：cov(日報酬) × annual_factor
        af     = 年化因子 = 訓練集交易日數 / 2（訓練期約 2 年）
        σ(w)   = √(w'Σw)（投資組合年化標準差）
        μ(w)   = w'μ  （投資組合年化期望報酬）

    三種優化目標：
        max_return : min -μ(w)             s.t. σ(w) ≤ threshold1
        min_risk   : min  σ(w)             s.t. μ(w) ≥ threshold1
        trade_off  : min -(μ(w) - σ(w))   s.t. σ(w) ≤ threshold1
                                               AND μ(w) ≥ threshold2
    ─────────────────────────────────────────────────────────────────

    參數
    ----
    train_ret  : 訓練期日報酬率 DataFrame（列=日期，欄=股票）
    opt_type   : "max_return" | "min_risk" | "trade_off"
    threshold1 : 主要約束閾值（max_return: 風險上限；min_risk: 報酬下限）
    threshold2 : 第二約束閾值，僅 trade_off 使用（報酬下限）

    回傳
    ----
    最佳權重向量 ndarray，shape = (n_assets,)
    """
    n  = train_ret.shape[1]
    # 年化因子：訓練集實際交易日數 ÷ 2（假設訓練期恰好為 2 年）
    af = len(train_ret) / 2

    # 年化期望報酬率（複利邏輯，與 R 的 (1+colMeans)^af - 1 完全一致）
    mu    = (1 + train_ret.mean()) ** af - 1
    mu    = mu.values   # 轉 ndarray 加速矩陣運算

    # 年化共變異矩陣（線性縮放，假設報酬率 i.i.d.）
    # 對應 R 的 cov(train_ret) * annual_factor
    sigma = train_ret.cov().values * af

    # ── 輔助函數 ────────────────────────────────────────────────────────────
    def port_return(w): return float(w @ mu)
    def port_risk(w):   return float(np.sqrt(w @ sigma @ w))

    # ── 目標函數 & 約束條件 ─────────────────────────────────────────────────
    if opt_type == "max_return":
        # 最大化報酬 ⟺ 最小化負報酬（nloptr 習慣最小化）
        objective   = lambda w: -port_return(w)
        constraints = [{"type": "ineq",
                        "fun": lambda w: threshold1 - port_risk(w)}]   # σ ≤ th1

    elif opt_type == "min_risk":
        # 最小化標準差，報酬需超過下限
        objective   = lambda w: port_risk(w)
        constraints = [{"type": "ineq",
                        "fun": lambda w: port_return(w) - threshold1}]  # μ ≥ th1

    elif opt_type == "trade_off":
        # 最大化（報酬 - 風險）⟺ 最小化 -(報酬 - 風險)
        objective   = lambda w: -(port_return(w) - port_risk(w))
        constraints = [
            {"type": "ineq", "fun": lambda w: threshold1 - port_risk(w)},   # σ ≤ th1
            {"type": "ineq", "fun": lambda w: port_return(w) - threshold2},  # μ ≥ th2
        ]
    else:
        raise ValueError(f"未知的 opt_type: {opt_type}")

    # 等式約束：權重加總 = 1（對應 R 的 eval_g_eq）
    constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    # 初始值：等權重（對應 R 的 rep(1/n_assets, n_assets)）
    w0     = np.ones(n) / n
    bounds = Bounds(lb=0.0, ub=1.0)   # 禁止放空（wi ∈ [0, 1]）

    # SLSQP：支援等式 + 不等式約束，為 scipy 中最接近 COBYLA 的選擇
    res = minimize(objective, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 5000})

    # 數值修正：確保非負且加總嚴格為 1
    w_opt = np.maximum(res.x, 0)
    w_opt = w_opt / w_opt.sum()
    return w_opt


def calc_max_drawdown(wealth_series: np.ndarray) -> float:
    """
    計算財富序列的最大回撤（Maximum Drawdown）。

    公式：MDD = min_t { (W_t - max_{s≤t} W_s) / max_{s≤t} W_s }
    即每個時間點相對於歷史最高點的跌幅，取最大者（最大損失）。
    對應 R 的 calc_max_drawdown 函數。
    """
    peak     = np.maximum.accumulate(wealth_series)   # 滾動最高點
    drawdown = (wealth_series - peak) / peak           # 各期回撤率
    return float(drawdown.min())                       # 最大跌幅（負值）

#%% ==============================================================================
# 五、滾動視窗回測設定
# ==============================================================================

# 產生測試起始日：每半年一期，2018-01 ～ 2025-07，共 16 期
# 對應 R 的 seq(as.Date("2018-01-01"), as.Date("2025-07-01"), by="6 months")
test_starts = pd.date_range(start="2018-01-01", end="2025-07-01", freq="6MS")
n_periods   = len(test_starts)

# 初始資本：等於 DCA 總投入金額，確保單筆策略與 DCA 公平比較
# DCA：每月 5,000 × 6 個月 × 16 期 = 480,000 元
initial_wealth  = int(n_periods * 6 * MONTHLY_INVEST)  # 480,000
total_dca_cost  = initial_wealth

# 單筆策略當前淨值初始化（四種策略各持有相同初始資本）
current_wealth = {
    "Max_Return":   initial_wealth,
    "Min_Risk":     initial_wealth,
    "Trade_Off":    initial_wealth,
    "Equal_Weight": initial_wealth,
}
benchmark_wealth = initial_wealth

# 初始化結果儲存 DataFrame（對應 R 的 results_wealth）
results_wealth = pd.DataFrame(
    0.0,
    index=range(n_periods),
    columns=["Date", "Max_Return", "Min_Risk", "Trade_Off",
             "Equal_Weight", "DCA", "Benchmark"]
)
results_wealth["Date"] = test_starts

# 初始化權重歷史（list of Series，回測結束後轉 DataFrame）
weights_history = {"max": [], "min": [], "trade": []}

# 初始化半年期報酬歷史（用於計算 Sharpe、勝率等績效指標）
returns_history = {k: [] for k in ["max", "min", "trade", "equal", "dca", "benchmark"]}

# DCA 累積持股數（每月扣款後逐期累加，跨期不重置）
dca_shares = np.zeros(N_ASSETS)

print("開始執行滾動視窗回測（2018–2025）...")
print("=" * 65)

#%% ==============================================================================
# 六、滾動視窗回測主迴圈
# ==============================================================================

for i, test_start in enumerate(test_starts):

    # ── 6.1 定義時間視窗 ──────────────────────────────────────────────────────
    # 測試期：從 test_start 起算半年
    test_end    = test_start + relativedelta(months=6) - timedelta(days=1)
    # 訓練期：測試期前 2 年（滾動視窗往前推）
    train_end   = test_start - timedelta(days=1)
    train_start = train_end  - relativedelta(years=2) + timedelta(days=1)

    print(f"  期 {i+1:2d}/{n_periods} | "
          f"訓練: {train_start.date()} ～ {train_end.date()} | "
          f"測試: {test_start.date()} ～ {test_end.date()}")

    # ── 6.2 提取訓練期日報酬率 ─────────────────────────────────────────────
    # 對應 R 的 data_returns %>% filter(date >= train_start & date <= train_end)
    train_mask   = (data_returns.index >= train_start) & \
                   (data_returns.index <= train_end)
    train_matrix = data_returns.loc[train_mask, STOCKS]

    # ── 6.3 執行各策略優化，若優化失敗則退回等權重 ─────────────────────────
    # 對應 R 的 tryCatch(run_optimization(...), error = function(e) rep(1/15, 15))
    try:
        w_max   = run_optimization(train_matrix, "max_return", threshold1=0.30)
    except Exception:
        w_max   = np.ones(N_ASSETS) / N_ASSETS

    try:
        w_min   = run_optimization(train_matrix, "min_risk",   threshold1=0.10)
    except Exception:
        w_min   = np.ones(N_ASSETS) / N_ASSETS

    try:
        w_trade = run_optimization(train_matrix, "trade_off",
                                   threshold1=0.25, threshold2=0.15)
    except Exception:
        w_trade = np.ones(N_ASSETS) / N_ASSETS

    # 等權重策略：固定 1/15，無需優化
    w_eq = np.ones(N_ASSETS) / N_ASSETS

    # ── 6.4 記錄本期各策略權重 ─────────────────────────────────────────────
    # 對應 R 的 rbind(weights_history$max, data.frame(Date=..., t(w_max)))
    weights_history["max"].append(
        pd.Series(w_max,   index=STOCKS, name=test_start))
    weights_history["min"].append(
        pd.Series(w_min,   index=STOCKS, name=test_start))
    weights_history["trade"].append(
        pd.Series(w_trade, index=STOCKS, name=test_start))

    # ── 6.5 計算測試期間個股半年報酬率 ────────────────────────────────────
    # 取測試期第一個與最後一個交易日的收盤價
    # 對應 R 的 data_prices %>% filter(date >= test_start) %>% head(1)
    p_start_mask = price_matrix.index >= test_start
    p_end_mask   = price_matrix.index <= test_end

    if not p_start_mask.any() or not p_end_mask.any():
        continue   # 若無資料（例如假日）則跳過本期

    p_start   = price_matrix.loc[p_start_mask].iloc[0][STOCKS].values.astype(float)
    p_end     = price_matrix.loc[p_end_mask  ].iloc[-1][STOCKS].values.astype(float)
    # 半年期簡單報酬率：(期末價 / 期初價) - 1
    stock_ret = (p_end / p_start) - 1

    # ── 6.6 計算大盤半年報酬率 ────────────────────────────────────────────
    bm_start_mask = benchmark_raw.index >= test_start
    bm_end_mask   = benchmark_raw.index <= test_end

    if bm_start_mask.any() and bm_end_mask.any():
        bm_p_start       = float(benchmark_raw.loc[bm_start_mask].iloc[0]["Benchmark"])
        bm_p_end         = float(benchmark_raw.loc[bm_end_mask  ].iloc[-1]["Benchmark"])
        bench_period_ret = (bm_p_end / bm_p_start) - 1
    else:
        bench_period_ret = 0.0

    # ── 6.7 更新單筆投入策略淨值 ──────────────────────────────────────────
    # 投資組合半年報酬率 = Σ (wi × 個股半年報酬率)
    port_ret = {
        "max":   float(w_max   @ stock_ret),
        "min":   float(w_min   @ stock_ret),
        "trade": float(w_trade @ stock_ret),
        "equal": float(w_eq    @ stock_ret),
    }

    # 記錄各策略半年報酬（後續計算 Sharpe、勝率等）
    for k in ["max", "min", "trade", "equal"]:
        returns_history[k].append(port_ret[k])
    returns_history["benchmark"].append(bench_period_ret)

    # 複利更新策略淨值：V_t = V_{t-1} × (1 + R_t)
    # 對應 R 的 current_wealth <- current_wealth * (1 + unlist(port_ret))
    for key, rkey in [("Max_Return",   "max"),
                      ("Min_Risk",     "min"),
                      ("Trade_Off",    "trade"),
                      ("Equal_Weight", "equal")]:
        current_wealth[key] *= (1 + port_ret[rkey])

    benchmark_wealth *= (1 + bench_period_ret)

    # ── 6.8 更新 DCA 定期定額策略 ─────────────────────────────────────────
    # 每個月月初扣款，5,000 元均分 15 檔，按當月開盤價買入對應股數
    semester_input = 0.0
    # seq(test_start, test_end, by="month") 的等效寫法
    month_dates    = pd.date_range(start=test_start, end=test_end, freq="MS")

    for m_date in month_dates:
        mask_m = price_matrix.index >= m_date
        if mask_m.any():
            p_month       = price_matrix.loc[mask_m].iloc[0][STOCKS].values.astype(float)
            # 每月買入股數 = (每月投入 / 檔數) / 當月股價
            dca_shares   += (MONTHLY_INVEST / N_ASSETS) / p_month
            semester_input += MONTHLY_INVEST

    # DCA 期末總市值 = Σ (累積股數 × 期末股價)
    dca_val_end   = float(dca_shares @ p_end)

    # DCA 半年報酬率（近似時間加權）
    dca_val_start = float(results_wealth.loc[i - 1, "DCA"]) if i > 0 else 0.0
    denom         = dca_val_start + semester_input
    dca_ret       = (dca_val_end - dca_val_start - semester_input) / denom \
                    if denom > 0 else 0.0
    returns_history["dca"].append(dca_ret)

    # ── 6.9 儲存本期末結果 ────────────────────────────────────────────────
    # 對應 R 的 results_wealth[i, 2:7] <- c(...)
    results_wealth.loc[i, "Max_Return"]   = current_wealth["Max_Return"]
    results_wealth.loc[i, "Min_Risk"]     = current_wealth["Min_Risk"]
    results_wealth.loc[i, "Trade_Off"]    = current_wealth["Trade_Off"]
    results_wealth.loc[i, "Equal_Weight"] = current_wealth["Equal_Weight"]
    results_wealth.loc[i, "DCA"]          = dca_val_end
    results_wealth.loc[i, "Benchmark"]    = benchmark_wealth

print("\n✓ 回測執行完成！\n")

#%% ==============================================================================
# 七、績效指標計算
# ==============================================================================

# 將 list 轉為 ndarray 方便向量運算
rh = {k: np.array(v) for k, v in returns_history.items()}

# ── 7.1 Sharpe Ratio（年化）──────────────────────────────────────────────────
# 公式：Sharpe = (E[R] / std[R]) × √2
# √2 為年化調整因子：半年期序列 → 年化（1/0.5 = 2，√2 = √(1/0.5)）
# 對應 R 的 lapply(returns_history, function(r) mean(r)/sd(r) * sqrt(2))
sharpe = {k: (rh[k].mean() / rh[k].std()) * np.sqrt(2) for k in rh}

# ── 7.2 最大回撤（MDD）───────────────────────────────────────────────────────
mdd = {col: calc_max_drawdown(results_wealth[col].values.astype(float))
       for col in ["Max_Return", "Min_Risk", "Trade_Off",
                   "Equal_Weight", "DCA", "Benchmark"]}

# ── 7.3 累積報酬 & 年化報酬率 ─────────────────────────────────────────────
final_values = results_wealth.iloc[-1][
    ["Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA", "Benchmark"]
].values.astype(float)

# 注意：DCA 的績效基準是總投入成本，其餘策略基準是初始資本（480,000）
cumulative_ret = np.array([
    (final_values[0] - initial_wealth) / initial_wealth,   # Max_Return
    (final_values[1] - initial_wealth) / initial_wealth,   # Min_Risk
    (final_values[2] - initial_wealth) / initial_wealth,   # Trade_Off
    (final_values[3] - initial_wealth) / initial_wealth,   # Equal_Weight
    (final_values[4] - total_dca_cost) / total_dca_cost,   # DCA（用總投入成本）
    (final_values[5] - initial_wealth) / initial_wealth,   # Benchmark
])

# 年化報酬率：(1 + 累積報酬)^(1/年數) - 1
# 16 期 × 0.5 年 = 8 年
n_years    = n_periods * 0.5
annual_ret = (1 + cumulative_ret) ** (1 / n_years) - 1

# ── 7.4 Alpha（超額報酬）─────────────────────────────────────────────────────
# 定義：策略年化報酬 - 大盤年化報酬（簡單差值，非 CAPM 回歸 Alpha）
benchmark_annual_ret = annual_ret[5]
alpha       = annual_ret[:5] - benchmark_annual_ret
alpha_names = ["Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA"]

# ── 7.5 資訊比率（Information Ratio）────────────────────────────────────────
# 公式：IR = E[超額報酬] / std[超額報酬] × √2（年化）
# 超額報酬 = 策略半年報酬 - 大盤半年報酬
ir_keys = ["max", "min", "trade", "equal", "dca"]
information_ratio = np.array([
    ((rh[k] - rh["benchmark"]).mean() /
     (rh[k] - rh["benchmark"]).std()) * np.sqrt(2)
    for k in ir_keys
])

# ── 7.6 勝率（Win Rate）──────────────────────────────────────────────────────
# 各期策略報酬 > 大盤報酬的期數比例
win_rate = np.array([
    (rh[k] > rh["benchmark"]).sum() / n_periods
    for k in ir_keys
])

# ── 7.7 列印分析結果 ──────────────────────────────────────────────────────────
print("【超額報酬分析（Alpha vs Benchmark）】")
alpha_df = pd.DataFrame({
    "Strategy":         alpha_names,
    "Annual_Return":    [f"{x:.2%}" for x in annual_ret[:5]],
    "Benchmark_Return": f"{benchmark_annual_ret:.2%}",
    "Alpha":            [f"{x:.2%}" for x in alpha],
    "Outperform":       ["✓ 勝出" if x > 0 else "✗ 落後" for x in alpha],
})
print(alpha_df.to_string(index=False))

print("\n【資訊比率（Information Ratio）】")
def ir_label(v):
    if v > 0.5:  return "優秀"
    if v > 0:    return "良好"
    if v > -0.5: return "普通"
    return "不佳"

ir_df = pd.DataFrame({
    "Strategy":       alpha_names,
    "Info_Ratio":     np.round(information_ratio, 3),
    "Interpretation": [ir_label(v) for v in information_ratio],
})
print(ir_df.to_string(index=False))

print("\n【勝率分析】")
wr_df = pd.DataFrame({
    "Strategy":   alpha_names,
    "Win_Rate":   [f"{x:.1%}" for x in win_rate],
    "Win_Count":  [f"{round(x * n_periods)}/{n_periods}" for x in win_rate],
    "Loss_Count": [f"{n_periods - round(x * n_periods)}/{n_periods}" for x in win_rate],
})
print(wr_df.to_string(index=False))

print("\n【完整績效表現摘要（含大盤比較）】")
strats_all   = ["Max_Return", "Min_Risk", "Trade_Off",
                "Equal_Weight", "DCA", "Benchmark"]
sharpe_order = ["max", "min", "trade", "equal", "dca", "benchmark"]
perf_df = pd.DataFrame({
    "Strategy":   strats_all,
    "Final":      [f"{v:,.0f}" for v in final_values],
    "Ann_Return": [f"{v:.2%}" for v in annual_ret],
    "Sharpe":     [round(sharpe[k], 3) for k in sharpe_order],
    "Max_DD":     [f"{mdd[s]:.2%}" for s in strats_all],
    "Alpha":      [f"{v:.2%}" if not np.isnan(v) else "—"
                   for v in list(alpha) + [float("nan")]],
    "Info_Ratio": [round(v, 3) if not np.isnan(v) else "—"
                   for v in list(information_ratio) + [float("nan")]],
})
print(perf_df.to_string(index=False))

#%% ==============================================================================
# 八、輸出目錄建立
# ==============================================================================

OUTPUT_DIR   = r"C:\Users\USER\Desktop\Portfolio_Backtest_Results"    # 在桌面建立新資料夾儲存結果
PLOT_DIR     = os.path.join(OUTPUT_DIR, "Plots")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "Benchmark_Analysis")

for d in [OUTPUT_DIR, PLOT_DIR, ANALYSIS_DIR]:
    os.makedirs(d, exist_ok=True)

# 各策略顏色（與 R 版視覺風格對應）
STRATEGY_COLORS = {
    "Max_Return":   "#00BCD4",
    "Min_Risk":     "#4CAF50",
    "Trade_Off":    "#FF69B4",
    "Equal_Weight": "#9C27B0",
    "DCA":          "#808000",
    "Benchmark":    "#FF8C69",
}

def save_fig(fig, filename: str) -> None:
    """儲存圖表至 PLOT_DIR，並列印確認訊息。"""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 圖表已儲存：{path}")

#%% ==============================================================================
# 九、視覺化（對應 R 的 p0 ～ p8）
# ==============================================================================

dates = pd.to_datetime(results_wealth["Date"])

# ── 圖零：相關係數矩陣（已在第三節繪製，直接儲存）───────────────────────────
save_fig(fig0, "00_correlation_matrix.png")

# ── 圖一：財富累積 vs 大盤 ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 6))
for col in ["Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA"]:
    ax1.plot(dates, results_wealth[col], marker="o", markersize=4,
             color=STRATEGY_COLORS[col], label=col, linewidth=1.5)
# 大盤用虛線區分（對應 R 的 linetype = "dashed"）
ax1.plot(dates, results_wealth["Benchmark"], marker="o", markersize=4,
         color=STRATEGY_COLORS["Benchmark"], label="Benchmark",
         linewidth=2, linestyle="--")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.set_title("投資組合績效對比（含大盤基準）\n虛線為大盤表現（加權指數）",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Date"); ax1.set_ylabel("Portfolio Value (TWD)")
ax1.legend(loc="upper left", fontsize=9); ax1.grid(True, alpha=0.3)
fig1.tight_layout()
save_fig(fig1, "01_wealth_vs_benchmark.png")

# ── 圖二：Alpha 橫向長條圖 ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
colors_alpha = ["#2ecc71" if a > 0 else "#e74c3c" for a in alpha]
bars = ax2.barh(alpha_names, alpha * 100, color=colors_alpha, height=0.6)
for bar, val in zip(bars, alpha):
    ax2.text(val * 100 + (0.2 if val >= 0 else -0.2),
             bar.get_y() + bar.get_height() / 2,
             f"{val:.2%}", va="center",
             ha="left" if val >= 0 else "right",
             fontweight="bold", fontsize=10)
ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title("超額報酬分析（Alpha）\n相對於大盤的年化報酬差異",
              fontsize=12, fontweight="bold")
ax2.set_xlabel("超額報酬（年化）")
fig2.tight_layout()
save_fig(fig2, "02_alpha_comparison.png")

# ── 圖三：夏普比率比較 ───────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 6))
sharpe_vals   = [sharpe[k] for k in sharpe_order]
display_names = ["Max_Return", "Min_Risk", "Trade_Off",
                 "Equal_Weight", "DCA", "Benchmark"]
bm_sharpe   = sharpe["benchmark"]
bar_colors  = ["#3498db" if v > bm_sharpe else "#95a5a6" for v in sharpe_vals]

sorted_idx    = np.argsort(sharpe_vals)
sorted_names  = [display_names[i] for i in sorted_idx]
sorted_vals   = [sharpe_vals[i]   for i in sorted_idx]
sorted_colors = [bar_colors[i]    for i in sorted_idx]

bars3 = ax3.barh(sorted_names, sorted_vals, color=sorted_colors, height=0.6)
for bar, val in zip(bars3, sorted_vals):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=9)
ax3.axvline(bm_sharpe, color="red", linestyle="--", linewidth=1.5,
            label=f"大盤基準（{bm_sharpe:.3f}）")
ax3.set_title("夏普比率比較（含大盤基準線）\n虛線為大盤夏普比率",
              fontsize=12, fontweight="bold")
ax3.set_xlabel("Sharpe Ratio"); ax3.legend(fontsize=9)
fig3.tight_layout()
save_fig(fig3, "03_sharpe_vs_benchmark.png")

# ── 圖四：逐期勝負熱力圖 ─────────────────────────────────────────────────────
# 對應 R 的 p4_winrate（ggplot2 geom_tile）
win_matrix = pd.DataFrame({
    "Max_Return":   (rh["max"]   > rh["benchmark"]).astype(int),
    "Min_Risk":     (rh["min"]   > rh["benchmark"]).astype(int),
    "Trade_Off":    (rh["trade"] > rh["benchmark"]).astype(int),
    "Equal_Weight": (rh["equal"] > rh["benchmark"]).astype(int),
    "DCA":          (rh["dca"]   > rh["benchmark"]).astype(int),
}).T   # 策略為列，期數為欄

fig4, ax4 = plt.subplots(figsize=(14, 5))
sns.heatmap(win_matrix, annot=False,
            cmap=sns.color_palette(["#e67e22", "#27ae60"], as_cmap=True),
            vmin=0, vmax=1, linewidths=1, linecolor="white",
            cbar=False, ax=ax4)
# 加上 V / X 符號（對應 R 的 geom_text(label=ifelse(Win, "V", "X"))）
for row_i in range(len(win_matrix.index)):
    for col_j in range(n_periods):
        mark = "V" if win_matrix.iloc[row_i, col_j] == 1 else "X"
        ax4.text(col_j + 0.5, row_i + 0.5, mark,
                 ha="center", va="center",
                 color="white", fontsize=11, fontweight="bold")
ax4.set_xticks(np.arange(n_periods) + 0.5)
ax4.set_xticklabels(range(1, n_periods + 1), fontsize=9)
ax4.set_title("逐期勝負記錄（相對於大盤）\nV = 該期優於大盤，X = 該期落後大盤",
              fontsize=12, fontweight="bold")
ax4.set_xlabel("期數")
from matplotlib.patches import Patch
ax4.legend(handles=[Patch(facecolor="#27ae60", label="勝出"),
                    Patch(facecolor="#e67e22", label="落後")],
           loc="upper right", bbox_to_anchor=(1.12, 1))
fig4.tight_layout()
save_fig(fig4, "04_win_rate_heatmap.png")

# ── 圖五：風險-報酬散佈圖 ────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 7))
risk_arr = np.array([rh[k].std() * np.sqrt(2) * 100 for k in sharpe_order])
ret_arr  = annual_ret * 100

for label, risk, ret in zip(display_names, risk_arr, ret_arr):
    if label == "Benchmark":
        # 三角形標記大盤位置
        ax5.scatter(risk, ret, color="#e74c3c", marker="^", s=150, zorder=5)
    else:
        ax5.scatter(risk, ret, color="#3498db", marker="o", s=80, zorder=5)
    ax5.annotate(label, (risk, ret),
                 textcoords="offset points", xytext=(6, 4), fontsize=9)
ax5.set_xlabel("年化風險（標準差 %）"); ax5.set_ylabel("年化報酬（%）")
ax5.set_title("風險-報酬分佈（年化）\n三角形為大盤位置",
              fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.3)
from matplotlib.lines import Line2D
ax5.legend(handles=[
    Line2D([0],[0], marker="^", color="w", markerfacecolor="#e74c3c",
           markersize=10, label="大盤"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#3498db",
           markersize=8, label="策略"),
], fontsize=9)
fig5.tight_layout()
save_fig(fig5, "05_risk_return_scatter.png")

# ── 圖六：相對績效走勢（排除 DCA）────────────────────────────────────────────
# 計算方式：策略市值 / 大盤市值（初始資本相同，直接相除即為相對倍數）
# DCA 因逐月投入，資金節奏與單筆策略不同，不納入此圖（另見圖一）
fig6, ax6 = plt.subplots(figsize=(12, 6))
bm_vals = results_wealth["Benchmark"].values.astype(float)
for col in ["Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight"]:
    rel = results_wealth[col].values.astype(float) / bm_vals
    ax6.plot(dates, rel, marker="o", markersize=4,
             color=STRATEGY_COLORS[col], label=col, linewidth=1.5)
ax6.axhline(1.0, color="black", linestyle="--", linewidth=1)
ax6.annotate("大盤基準線 (= 1.0)", xy=(dates.iloc[-1], 1.0),
             xytext=(-10, 8), textcoords="offset points", fontsize=9)
ax6.set_title("相對績效走勢（單筆策略 / 大盤）\n"
              "大於 1.0 表示優於大盤；DCA 因資金節奏不同不納入此圖",
              fontsize=12, fontweight="bold")
ax6.set_xlabel("Date"); ax6.set_ylabel("相對績效倍數")
ax6.legend(loc="upper left", fontsize=9); ax6.grid(True, alpha=0.3)
fig6.tight_layout()
save_fig(fig6, "06_relative_performance.png")

# ── 圖七：回撤比較（含大盤）─────────────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(12, 6))
for col in ["Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA"]:
    vals = results_wealth[col].values.astype(float)
    peak = np.maximum.accumulate(vals)
    dd   = (vals - peak) / peak
    ax7.plot(dates, dd * 100, color=STRATEGY_COLORS[col], label=col, linewidth=0.9)
    # 標記最大回撤位置（對應 R 的 geom_point + geom_label）
    min_idx = np.argmin(dd)
    ax7.scatter(dates.iloc[min_idx], dd[min_idx] * 100,
                color=STRATEGY_COLORS[col], s=40, zorder=5)
    ax7.annotate(f"{dd[min_idx]:.1%}",
                 (dates.iloc[min_idx], dd[min_idx] * 100),
                 xytext=(4, -12), textcoords="offset points",
                 fontsize=7.5, color=STRATEGY_COLORS[col])

# 大盤回撤用虛線顯示
bm_vals_dd = results_wealth["Benchmark"].values.astype(float)
bm_peak    = np.maximum.accumulate(bm_vals_dd)
bm_dd      = (bm_vals_dd - bm_peak) / bm_peak
ax7.plot(dates, bm_dd * 100, color=STRATEGY_COLORS["Benchmark"],
         label="Benchmark", linewidth=1.5, linestyle="--")
min_idx_bm = np.argmin(bm_dd)
ax7.scatter(dates.iloc[min_idx_bm], bm_dd[min_idx_bm] * 100,
            color=STRATEGY_COLORS["Benchmark"], s=40, zorder=5)
ax7.annotate(f"{bm_dd[min_idx_bm]:.1%}",
             (dates.iloc[min_idx_bm], bm_dd[min_idx_bm] * 100),
             xytext=(4, -12), textcoords="offset points",
             fontsize=7.5, color=STRATEGY_COLORS["Benchmark"])

ax7.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax7.yaxis.set_major_formatter(mtick.PercentFormatter())
ax7.set_title("回撤比較（含大盤）\n虛線粗線為大盤回撤，標註點為最大回撤位置",
              fontsize=12, fontweight="bold")
ax7.set_xlabel("Date"); ax7.set_ylabel("Drawdown (%)")
ax7.legend(loc="lower right", fontsize=9); ax7.grid(True, alpha=0.3)
fig7.tight_layout()
save_fig(fig7, "07_drawdown_vs_benchmark.png")

# ── 圖八：各策略每期持股權重變化（拆成三張獨立圖）────────────────────────────

# 將 weights_history（list of Series）轉為 DataFrame
wh_max_df   = pd.DataFrame(weights_history["max"],   columns=STOCKS)
wh_min_df   = pd.DataFrame(weights_history["min"],   columns=STOCKS)
wh_trade_df = pd.DataFrame(weights_history["trade"], columns=STOCKS)

# 加上日期欄與策略標籤
wh_max_df["Date"]   = test_starts;  wh_max_df["Strategy"]   = "Max_Return"
wh_min_df["Date"]   = test_starts;  wh_min_df["Strategy"]   = "Min_Risk"
wh_trade_df["Date"] = test_starts;  wh_trade_df["Strategy"] = "Trade_Off"

# 合併三張表
df_all_weights = pd.concat([wh_max_df, wh_min_df, wh_trade_df], ignore_index=True)

# 寬格式 → 長格式
df_long_weights = df_all_weights.melt(
    id_vars=["Date", "Strategy"],
    var_name="Stock",
    value_name="Weight"
)

# 加入「代碼\n中文名」標籤
df_long_weights["Label"] = df_long_weights["Stock"].map(
    lambda s: f"{s}\n{NAME_MAP.get(s, s)}"
)

# 過濾趨近於 0 的雜訊（< 0.1%）
df_long_weights = df_long_weights[df_long_weights["Weight"] > 0.001]

# 格式化期別標籤（YYYY-MM）
df_long_weights["Period"] = pd.to_datetime(df_long_weights["Date"]).dt.strftime("%Y-%m")

# 取得所有出現過的標的，分配固定顏色（三張圖顏色一致，方便對照）
all_labels = sorted(df_long_weights["Label"].unique())
palette    = sns.color_palette("tab20", n_colors=len(all_labels))
color_map  = dict(zip(all_labels, palette))

# 各策略對應的檔名
fig8_files = {
    "Max_Return":  "08a_weights_max_return.png",
    "Min_Risk":    "08b_weights_min_risk.png",
    "Trade_Off":   "08c_weights_trade_off.png",
}

for strategy, filename in fig8_files.items():
    df_sub  = df_long_weights[df_long_weights["Strategy"] == strategy]
    periods = sorted(df_sub["Period"].unique())

    # pivot_table 建立堆疊資料矩陣（列=期別，欄=標的）
    pivot = df_sub.pivot_table(index="Period", columns="Label",
                               values="Weight", aggfunc="sum").fillna(0)
    pivot = pivot.loc[periods]

    fig, ax = plt.subplots(figsize=(14, 6))   # 單張圖，寬度足夠

    # 逐標的疊加長條
    bottom = np.zeros(len(periods))
    for lbl in pivot.columns:
        ax.bar(range(len(periods)), pivot[lbl].values,
               bottom=bottom,
               color=color_map.get(lbl, "gray"),
               label=lbl,
               width=0.8,
               edgecolor="white",
               linewidth=0.3)
        bottom += pivot[lbl].values

    ax.set_title(f"{strategy} ── 每期持股權重變化（2018–2025）",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("權重配置")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)

    # 圖例放在圖表右側，字體放大、單欄排列
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(l, "gray"))
               for l in pivot.columns]
    ax.legend(handles, pivot.columns,
              loc="upper left",
              bbox_to_anchor=(1.01, 1),
              borderaxespad=0,
              fontsize=9,       # 比三合一版本大很多
              ncol=1)           # 單欄，避免擠壓

    fig.tight_layout()
    save_fig(fig, filename)

#%% ==============================================================================
# 十、統計顯著性檢定（配對 t-test）
# ==============================================================================

from scipy.stats import ttest_rel

print("\n【統計顯著性檢定（配對 t-test）】")
print("H0: 策略報酬 = 大盤報酬，Ha: 策略報酬 ≠ 大盤報酬\n")

for name, key in zip(alpha_names, ir_keys):
    result = ttest_rel(rh[key], rh["benchmark"])
    diff   = rh[key].mean() - rh["benchmark"].mean()
    sig    = "*** 顯著" if result.pvalue < 0.05 else ""
    conc   = ("顯著優於大盤" if diff > 0 else "顯著落後大盤") \
             if result.pvalue < 0.05 else "與大盤無顯著差異"
    print(f"【{name}】")
    print(f"  平均差異: {diff:.4f}（{diff * 100:.2f}%）")
    print(f"  t-統計量: {result.statistic:.3f}")
    print(f"  p-value:  {result.pvalue:.4f} {sig}")
    print(f"  結論: {conc}\n")

#%% ==============================================================================
# 十一、CSV 匯出
# ==============================================================================

# ── 完整績效表 ────────────────────────────────────────────────────────────────
performance_summary = pd.DataFrame({
    "Strategy":          strats_all,
    "Initial":           [initial_wealth] * 4 + [total_dca_cost, initial_wealth],
    "Final":             final_values,
    "Cumulative_Return": cumulative_ret,
    "Annual_Return":     annual_ret,
    "Sharpe_Ratio":      [sharpe[k] for k in sharpe_order],
    "Max_Drawdown":      [mdd[s] for s in strats_all],
    "Alpha":             list(alpha) + [float("nan")],
    "Info_Ratio":        list(information_ratio) + [float("nan")],
    "Win_Rate":          list(win_rate) + [float("nan")],
})
performance_summary.to_csv(
    os.path.join(OUTPUT_DIR, "01_績效總表_含大盤.csv"),
    index=False, encoding="utf-8-sig")

results_wealth.to_csv(
    os.path.join(OUTPUT_DIR, "02_每期財富淨值序列.csv"),
    index=False, encoding="utf-8-sig")

# ── 平均權重表 ────────────────────────────────────────────────────────────────
avg_weights = pd.DataFrame({
    "Stock":        STOCKS,
    "Stock_Name":   STOCK_NAMES,
    "Max_Return":   wh_max_df[STOCKS].mean().values,
    "Min_Risk":     wh_min_df[STOCKS].mean().values,
    "Trade_Off":    wh_trade_df[STOCKS].mean().values,
    "Equal_Weight": 1 / N_ASSETS,
})
avg_weights.to_csv(
    os.path.join(OUTPUT_DIR, "03_各股票平均權重.csv"),
    index=False, encoding="utf-8-sig")

# ── 詳細權重歷史 ──────────────────────────────────────────────────────────────
for fname, df in [("04_權重歷史_最大報酬", wh_max_df),
                  ("05_權重歷史_最小風險", wh_min_df),
                  ("06_權重歷史_平衡策略", wh_trade_df)]:
    df.to_csv(os.path.join(OUTPUT_DIR, f"{fname}.csv"),
              index=False, encoding="utf-8-sig")

# ── 進階指標分析 ──────────────────────────────────────────────────────────────
alpha_df.to_csv(
    os.path.join(ANALYSIS_DIR, "alpha_分析.csv"),
    index=False, encoding="utf-8-sig")
ir_df.to_csv(
    os.path.join(ANALYSIS_DIR, "資訊比率_分析.csv"),
    index=False, encoding="utf-8-sig")
wr_df.to_csv(
    os.path.join(ANALYSIS_DIR, "勝率_分析.csv"),
    index=False, encoding="utf-8-sig")
win_matrix.T.reset_index().rename(columns={"index": "Period"}).to_csv(
    os.path.join(ANALYSIS_DIR, "逐期勝負紀錄.csv"),
    index=False, encoding="utf-8-sig")
pd.DataFrame({
    "Strategy":     display_names,
    "Risk":         risk_arr,
    "Return":       ret_arr,
    "Is_Benchmark": ["大盤" if s == "Benchmark" else "策略" for s in display_names],}).to_csv(
    os.path.join(ANALYSIS_DIR, "風險報酬分佈數據.csv"),
    index=False, encoding="utf-8-sig")

print("\n✓ 大盤比較分析完成！")
print("✓ 績效報告已儲存至: {OUTPUT_DIR}/")
print("✓ 圖表已儲存至:     {PLOT_DIR}/")
print("✓ 詳細分析已儲存至: {ANALYSIS_DIR}/")
print("\n完成！")
