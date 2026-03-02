# ==============================================================================
# Portfolio Optimization with Rolling Window Backtest
# 投資組合優化：滾動視窗回測
# ==============================================================================
# Description:
#   使用數學規劃比較五種投資策略：
#   - 最大報酬策略 (風險上限 < 30%)
#   - 最小風險策略 (報酬下限 > 10%)
#   - 組合策略 (風險 < 25% 且 報酬 > 15%)
#   - 定期定額 (DCA)
#   - 等權重投資 (Equal Weight)
#
# ==============================================================================

# 載入套件 =====================================================================
library(quantmod)   # 股票數據下載
library(nloptr)     # 非線性優化計算
library(dplyr)      # 資料處理與轉換
library(tidyr)      # 資料整理
library(lubridate)  # 日期處理
library(scales)     # 數據格式化（百分比、千分撇）
library(ggplot2)    # 專業繪圖

# 基礎配置 =====================================================================
# 從 ETF 00850 成分股中挑選15檔標的
STOCKS <- c("2330.TW", "2317.TW", "2454.TW", "3231.TW", "2357.TW", 
            "2379.TW", "2327.TW", "2881.TW", "2891.TW", "2885.TW",
            "2890.TW", "1102.TW", "2618.TW", "1216.TW", "2412.TW")

STOCK_NAMES <- c("台積電", "鴻海", "聯發科", "緯創", "華碩", 
                 "瑞昱", "國巨", "富邦金", "中信金", "元大金",
                 "永豐金", "亞泥", "長榮航", "統一", "中華電")

START_DATE <- as.Date("2016-01-01")
END_DATE <- as.Date("2026-01-01")
MONTHLY_INVEST <- 5000  # 定期定額每月投入金額

# 資料下載 =====================================================================

get_stock_data <- function(ticker) {
  tryCatch({
    data <- getSymbols(ticker, src = "yahoo", from = START_DATE, to = END_DATE, auto.assign = FALSE)
    return(Ad(data))
  }, error = function(e) NULL)
}

price_list <- lapply(STOCKS, get_stock_data)
price_matrix <- do.call(merge, price_list)
names(price_matrix) <- STOCKS
price_matrix <- na.omit(price_matrix)

benchmark_data <- getSymbols("^TWII", src = "yahoo", from = START_DATE, to = END_DATE, auto.assign = FALSE)
benchmark_price <- Ad(benchmark_data)
names(benchmark_price) <- "Benchmark"

data_prices <- data.frame(date = index(price_matrix), 
                         coredata(price_matrix), 
                         check.names = FALSE)

# 計算日報酬率
data_returns <- data_prices
data_returns[, -1] <- (data_prices[, -1] / lag(data_prices[, -1])) - 1
data_returns <- na.omit(data_returns)

data_benchmark_ret <- data.frame(date = index(benchmark_price), coredata(benchmark_price))
data_benchmark_ret$Benchmark_Ret <- (data_benchmark_ret$Benchmark / lag(data_benchmark_ret$Benchmark)) - 1
data_benchmark_ret <- na.omit(data_benchmark_ret)

cat(sprintf("成功下載 %d 檔股票，期間從 %s 到 %s (共 %d 個交易日)\n\n",
            length(STOCKS), min(data_prices$date), 
            max(data_prices$date), nrow(data_prices)))


# 相關係數矩陣計算與視覺化 =====================================================

# 計算全期間相關係數矩陣
cor_matrix <- cor(data_returns[, -1])
print(cor_matrix)

df_cor <- as.data.frame(cor_matrix) %>%
  mutate(Var1 = rownames(.)) %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "Correlation")

name_map <- data.frame(Stock = STOCKS, Name = STOCK_NAMES)

df_cor <- df_cor %>%
  left_join(name_map, by = c("Var1" = "Stock")) %>%
  rename(Name1 = Name) %>%
  left_join(name_map, by = c("Var2" = "Stock")) %>%
  rename(Name2 = Name) %>%
  mutate(Label1 = paste(Var1, Name1),
         Label2 = paste(Var2, Name2))

p0 <- ggplot(df_cor, aes(x = Label1, y = Label2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#4575b4", mid = "white", high = "#d73027", 
                       midpoint = 0.5, limit = c(0, 1), space = "Lab", 
                       name="Correlation") +
  geom_text(aes(label = round(Correlation, 2)), color = "black", size = 3) +
  labs(title = "Asset Correlation Matrix (00850 Components)",
       subtitle = "Based on daily returns (2016-2025)",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5))

print(p0)

# 定義優化函數 =================================================================

run_optimization <- function(train_ret, type, threshold1 = NULL, threshold2 = NULL) {
  # 使用數學規劃執行投資組合優化
  # 
  # 參數：
  #   train_ret: 訓練期間的報酬率矩陣
  #   type: 優化策略 "max_return", "min_risk", "trade_off"
  #   threshold1: 限制1
  #   threshold2: 限制2 (只用在 trade_off 策略)
  # 
  # 回傳值：
  #   最佳資產權重向量
  
  n_assets <- ncol(train_ret)
  annual_factor <- nrow(train_ret) / 2  # 年化因子 (假設訓練期為2年)
  
  # 年化期望報酬率 (採用複利邏輯)
  mu <- (1 + colMeans(train_ret))^annual_factor - 1
  # 年化共變異矩陣
  sigma <- cov(train_ret) * annual_factor
  
  # 定義目標函數與限制函數
  if (type == "max_return") {
    eval_f <- function(w) -sum(w * mu)                                          # 目標：最小化「負報酬」等於最大化報酬
    eval_g_ineq <- function(w) sqrt(t(w) %*% sigma %*% w) - threshold1          # 風險限制
  } else if (type == "min_risk") {
    eval_f <- function(w) sqrt(t(w) %*% sigma %*% w)                            # 目標：最小化標準差
    eval_g_ineq <- function(w) threshold1 - sum(w * mu)                         # 報酬限制
  } else if (type == "trade_off") {
    eval_f <- function(w) -(sum(w * mu) - sqrt(t(w) %*% sigma %*% w))           # 目標：最大化(報酬-風險)
    eval_g_ineq <- function(w) {
      c(sqrt(t(w) %*% sigma %*% w) - threshold1, threshold2 - sum(w * mu))      # 雙重限制
    }
  }
  
  eval_g_eq <- function(w) sum(w) - 1  # 權重總和= 1
  
  # 執行規劃求解初始設定
  opts <- list(algorithm = "NLOPT_LN_COBYLA", xtol_rel = 1e-6, maxeval = 5000)
  res <- nloptr(x0 = rep(1/n_assets, n_assets), 
                eval_f = eval_f, 
                eval_g_ineq = eval_g_ineq, 
                eval_g_eq = eval_g_eq, 
                lb = rep(0, n_assets), 
                ub = rep(1, n_assets), 
                opts = opts)
  
  return(res$solution)
}

calc_max_drawdown <- function(wealth_series) {
  # 從財富序列計算最大回撤
  peak <- cummax(wealth_series)
  drawdown <- (wealth_series - peak) / peak
  return(min(drawdown))
}

# 回測設定 =====================================================================

test_starts <- seq(as.Date("2018-01-01"), as.Date("2025-07-01"), by = "6 months")
n_periods <- length(test_starts)

initial_wealth <- n_periods * 6 * MONTHLY_INVEST   # 初始化投入資本 (為確保公平對比，設定單筆投入等於 DCA 總投入金額)
current_wealth <- setNames(rep(initial_wealth, 4), 
                          c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight"))
benchmark_wealth <- initial_wealth

# 初始化結果儲存df
results_wealth <- data.frame(Date = test_starts, 
                            Max_Return = 0, Min_Risk = 0, Trade_Off = 0, 
                            Equal_Weight = 0, DCA = 0, Benchmark = 0)

weights_history <- list(max = data.frame(), min = data.frame(), trade = data.frame())
returns_history <- list(max = numeric(), min = numeric(), trade = numeric(),
                       equal = numeric(), dca = numeric(), benchmark = numeric())

dca_shares <- rep(0, length(STOCKS))  # DCA 累積持股數

cat("開始執行滾動視窗回測 (2018-2025)...\n")

# 滾動視窗回測主迴圈 ===========================================================

for (i in seq_along(test_starts)) {
  # 定義時間視窗 (2年訓練，0.5年測試)
  test_start <- test_starts[i]
  test_end <- test_start + months(6) - days(1)
  train_end <- test_start - days(1)
  train_start <- train_end - years(2) + days(1)
  
  # 提取訓練期數據
  train_matrix <- data_returns %>% 
    filter(date >= train_start & date <= train_end) %>% 
    dplyr::select(dplyr::all_of(STOCKS))
  
  # 執行
  w_max <- tryCatch(run_optimization(train_matrix, "max_return", 0.3), 
                   error = function(e) rep(1/15, 15))
  w_min <- tryCatch(run_optimization(train_matrix, "min_risk", 0.1), 
                   error = function(e) rep(1/15, 15))
  w_trade <- tryCatch(run_optimization(train_matrix, "trade_off", 0.25, 0.15), 
                     error = function(e) rep(1/15, 15))
  w_eq <- rep(1/15, 15)
  
  # 紀錄各策略權重變化
  weights_history$max <- rbind(weights_history$max, 
                               data.frame(Date = test_start, t(w_max)))
  weights_history$min <- rbind(weights_history$min, 
                               data.frame(Date = test_start, t(w_min)))
  weights_history$trade <- rbind(weights_history$trade, 
                                 data.frame(Date = test_start, t(w_trade)))
  
  # 計算測試期間報酬
  price_start <- as.numeric(data_prices %>% 
                           filter(date >= test_start) %>% 
                           head(1) %>% 
                           dplyr::select(dplyr::all_of(STOCKS)))
  price_end <- as.numeric(data_prices %>% 
                         filter(date <= test_end) %>% 
                         tail(1) %>% 
                         dplyr::select(dplyr::all_of(STOCKS)))
  stock_ret <- (price_end / price_start) - 1
  
  # 計算測試期間大盤報酬
  bench_start_df <- data_benchmark_ret %>% filter(date >= test_start) %>% head(1)
  bench_end_df <- data_benchmark_ret %>% filter(date <= test_end) %>% tail(1)
  
  if(nrow(bench_start_df) > 0 && nrow(bench_end_df) > 0) {
    bench_price_start <- as.numeric(bench_start_df$Benchmark)
    bench_price_end <- as.numeric(bench_end_df$Benchmark)
    bench_period_ret <- (bench_price_end / bench_price_start) - 1
  } else {
    bench_period_ret <- 0
  }
  
  # 更新單筆投入策略之市值
  port_ret <- list(max = sum(w_max * stock_ret),
                  min = sum(w_min * stock_ret),
                  trade = sum(w_trade * stock_ret),
                  equal = sum(w_eq * stock_ret))
  
  returns_history$max <- c(returns_history$max, port_ret$max)
  returns_history$min <- c(returns_history$min, port_ret$min)
  returns_history$trade <- c(returns_history$trade, port_ret$trade)
  returns_history$equal <- c(returns_history$equal, port_ret$equal)
  returns_history$benchmark <- c(returns_history$benchmark, bench_period_ret)
  
  # 更新 DCA 定期定額策略市值
  dca_val_start <- if(i == 1) 0 else results_wealth$DCA[i-1]
  semester_input <- 0
  
  months_in_semester <- seq(test_start, test_end, by = "month")
  for (m_date in months_in_semester) {
    p_month <- data_prices %>% filter(date >= m_date) %>% head(1) %>% dplyr::select(dplyr::all_of(STOCKS))
    if(nrow(p_month) > 0) {
      dca_shares <- dca_shares + (MONTHLY_INVEST / 15) / as.numeric(p_month)
      semester_input <- semester_input + MONTHLY_INVEST
    }
  }
  
  dca_val_end <- sum(dca_shares * price_end)
  dca_ret <- (dca_val_end - dca_val_start - semester_input) / 
             (dca_val_start + semester_input)
  returns_history$dca <- c(returns_history$dca, dca_ret)

  current_wealth <- current_wealth * (1 + unlist(port_ret))
  benchmark_wealth <- benchmark_wealth * (1 + bench_period_ret)

  # 儲存本期結果
  results_wealth[i, 2:7] <- c(current_wealth["Max_Return"], 
                              current_wealth["Min_Risk"], 
                              current_wealth["Trade_Off"], 
                              current_wealth["Equal_Weight"], 
                              dca_val_end, 
                              benchmark_wealth)
}

cat("\n回測執行完成！\n\n")

# 績效指標計算（含大盤比較） ===================================================

## Sharpe Ratio (年化計算：半年報酬率之均值/標準差 * 根號2)
sharpe <- lapply(returns_history, function(r) mean(r) / sd(r) * sqrt(2))

## 最大回撤
mdd <- lapply(results_wealth[, -1], calc_max_drawdown)

## 累積報酬與年化報酬
final_values <- as.numeric(tail(results_wealth, 1)[, -1])
total_dca_cost <- n_periods * 6 * MONTHLY_INVEST

cumulative_ret <- c(
  (final_values[1] - initial_wealth) / initial_wealth,  # Max_Return
  (final_values[2] - initial_wealth) / initial_wealth,  # Min_Risk
  (final_values[3] - initial_wealth) / initial_wealth,  # Trade_Off
  (final_values[4] - initial_wealth) / initial_wealth,  # Equal_Weight
  (final_values[5] - total_dca_cost) / total_dca_cost,  # DCA
  (final_values[6] - initial_wealth) / initial_wealth   # Benchmark
)

n_years <- n_periods * 0.5
annual_ret <- (1 + cumulative_ret)^(1/n_years) - 1

## 相對於大盤的超額報酬 (Alpha)
benchmark_annual_ret <- annual_ret[6]

alpha <- annual_ret[1:5] - benchmark_annual_ret
alpha_names <- c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA")

cat("\n【超額報酬分析 (Alpha vs Benchmark)】\n")
alpha_df <- data.frame(
  Strategy = alpha_names,
  Annual_Return = percent(annual_ret[1:5], 0.01),
  Benchmark_Return = percent(benchmark_annual_ret, 0.01),
  Alpha = percent(alpha, 0.01),
  Outperform = ifelse(alpha > 0, "✓ 勝出", "✗ 落後")
)
print(alpha_df, row.names = FALSE)


## 資訊比率 (Information Ratio)
# Information Ratio = (策略報酬 - 大盤報酬) / 追蹤誤差
# 追蹤誤差 = sd(策略報酬 - 大盤報酬)

information_ratio <- numeric(5)
for (i in 1:5) {
  strategy_name <- c("max", "min", "trade", "equal", "dca")[i]
  tracking_error <- returns_history[[strategy_name]] - returns_history$benchmark
  information_ratio[i] <- mean(tracking_error) / sd(tracking_error) * sqrt(2)
}

cat("\n【資訊比率 (Information Ratio)】\n")
ir_df <- data.frame(
  Strategy = alpha_names,
  Info_Ratio = round(information_ratio, 3),
  Interpretation = case_when(
    information_ratio > 0.5 ~ "優秀",
    information_ratio > 0 ~ "良好",
    information_ratio > -0.5 ~ "普通",
    TRUE ~ "不佳"
  )
)
print(ir_df, row.names = FALSE)

## 勝率分析 (Win Rate)
cat("\n【檢查資料完整性】\n")
lengths_check <- data.frame(
  Data = c("max", "min", "trade", "equal", "dca", "benchmark"),
  Length = c(length(returns_history$max),
             length(returns_history$min),
             length(returns_history$trade),
             length(returns_history$equal),
             length(returns_history$dca),
             length(returns_history$benchmark)),
  Expected = n_periods
)
print(lengths_check)

 # 修正長度不一致的問題
for (name in c("max", "min", "trade", "equal", "dca", "benchmark")) {
  if (length(returns_history[[name]]) > n_periods) {
    cat(sprintf("警告：%s 長度 %d > %d，截取前 %d 個\n", 
                name, length(returns_history[[name]]), n_periods, n_periods))
    returns_history[[name]] <- returns_history[[name]][1:n_periods]
  } else if (length(returns_history[[name]]) < n_periods) {
    cat(sprintf("錯誤：%s 長度 %d < %d\n", 
                name, length(returns_history[[name]]), n_periods))
  }
}

 # 計算勝率
win_rate <- numeric(5)
strategy_names_short <- c("max", "min", "trade", "equal", "dca")

for (i in 1:5) {
  wins <- sum(returns_history[[strategy_names_short[i]]] > returns_history$benchmark)
  win_rate[i] <- wins / n_periods
}

cat("\n【勝率分析】\n")
wr_df <- data.frame(
  Strategy = c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA"),
  Win_Rate = percent(win_rate, 0.1),
  Win_Count = paste0(round(win_rate * n_periods), "/", n_periods),
  Loss_Count = paste0(n_periods - round(win_rate * n_periods), "/", n_periods)
)
print(wr_df, row.names = FALSE)

# ==============================================================================
# 建立完整績效表（含大盤比較）
# ==============================================================================

performance_summary <- data.frame(
  Strategy = c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA", "Benchmark"),
  Initial = c(rep(initial_wealth, 4), total_dca_cost, initial_wealth),
  Final = final_values,
  Cumulative_Return = cumulative_ret,
  Annual_Return = annual_ret,
  Sharpe_Ratio = c(sharpe$max, sharpe$min, sharpe$trade, 
                   sharpe$equal, sharpe$dca, sharpe$benchmark),
  Max_Drawdown = unlist(mdd),
  Alpha = c(alpha, NA),  # 大盤沒有 Alpha
  Info_Ratio = c(information_ratio, NA),  # 大盤沒有 IR
  Win_Rate = c(win_rate, NA)  # 大盤沒有勝率
)

cat("\n【完整績效表現摘要（含大盤比較）】\n")
print(data.frame(
  Strategy = performance_summary$Strategy,
  Final = comma(performance_summary$Final),
  Ann_Return = percent(performance_summary$Annual_Return, 0.01),
  Sharpe = round(performance_summary$Sharpe_Ratio, 3),
  Max_DD = percent(performance_summary$Max_Drawdown, 0.01),
  Alpha = percent(performance_summary$Alpha, 0.01),
  Info_Ratio = round(performance_summary$Info_Ratio, 3)
), row.names = FALSE)

# ==============================================================================
# 視覺化 1：財富累積 vs 大盤（強調比較）
# ==============================================================================

df_wealth <- pivot_longer(results_wealth, -Date, names_to = "Strategy", values_to = "Wealth")

# 標記大盤
df_wealth <- df_wealth %>%
  mutate(Is_Benchmark = ifelse(Strategy == "Benchmark", "大盤", "策略"))

p1 <- ggplot(df_wealth, aes(x = Date, y = Wealth, color = Strategy, 
                            linetype = Is_Benchmark, size = Is_Benchmark)) +
  geom_line() +
  geom_point(size = 2) +
  scale_size_manual(values = c("策略" = 1, "大盤" = 1.5)) +
  scale_linetype_manual(values = c("策略" = "solid", "大盤" = "dashed")) +
  scale_y_continuous(labels = comma) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "投資組合績效對比 (含大盤基準)",
       subtitle = "虛線為大盤表現 (加權指數)",
       x = "Date", y = "Portfolio Value (TWD)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        legend.position = "bottom") +
  guides(linetype = "none", size = "none")

print(p1)

# ==============================================================================
# 視覺化 2：超額報酬 (Alpha) 長條圖
# ==============================================================================

alpha_data <- data.frame(Strategy = alpha_names, Alpha = alpha) %>%
  mutate(Positive = Alpha > 0)

p2_alpha <- ggplot(alpha_data, aes(x = reorder(Strategy, Alpha), y = Alpha, fill = Positive)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray30") +
  # 使用較小的字體並優化 hjust，expand 指定 y 軸範圍避免文字超出
  geom_text(aes(label = percent(Alpha, 0.01),
                hjust = ifelse(Alpha >= 0, -0.1, 1.1)), 
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#e74c3c")) +
  scale_y_continuous(labels = percent, expand = expansion(mult = c(0.2, 0.2))) + 
  coord_flip() +
  labs(title = "超額報酬分析 (Alpha)",
       subtitle = "相對於大盤的年化報酬差異",
       x = "", y = "超額報酬 (年化)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        plot.margin = margin(10, 30, 10, 10),
        legend.position = "none")

print(p2_alpha)

# ==============================================================================
# 視覺化 3：風險調整後報酬比較（夏普比率 vs 大盤）
# ==============================================================================

sharpe_data <- data.frame(Strategy = names(sharpe), Sharpe = unlist(sharpe)) %>%
  mutate(Better_Than_Benchmark = Sharpe > sharpe$benchmark)

p3_sharpe <- ggplot(sharpe_data, aes(x = reorder(Strategy, Sharpe), y = Sharpe, fill = Better_Than_Benchmark)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = sharpe$benchmark, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_text(aes(label = round(Sharpe, 3)), hjust = -0.3, size = 3.5) +
  # 動態計算標註位置，避免寫死
  annotate("text", x = length(sharpe), y = sharpe$benchmark, 
           label = paste0("大盤基準 (", round(sharpe$benchmark, 3), ")"),
           color = "red", hjust = -0.1, vjust = -1, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#3498db", "FALSE" = "#95a5a6")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) + 
  coord_flip() +
  labs(title = "夏普比率比較 (含大盤基準線)",
       subtitle = "虛線為大盤夏普比率", x = "", y = "Sharpe Ratio") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        plot.margin = margin(10, 40, 10, 10),
        legend.position = "none")

print(p3_sharpe)

# ==============================================================================
# 視覺化 4：勝率熱力圖（每期勝負情況）
# ==============================================================================

win_loss_matrix <- data.frame(
  Period = 1:n_periods,
  Date = test_starts,
  Max_Return = returns_history$max > returns_history$benchmark,
  Min_Risk = returns_history$min > returns_history$benchmark,
  Trade_Off = returns_history$trade > returns_history$benchmark,
  Equal_Weight = returns_history$equal > returns_history$benchmark,
  DCA = returns_history$dca > returns_history$benchmark
) %>%
  pivot_longer(cols = -c(Period, Date), names_to = "Strategy", values_to = "Win") %>%
  mutate(
    Result = ifelse(Win, "勝出", "落後"),
    Result_Detail = paste0(Strategy, "\n", format(Date, "%Y-%m"))
  )

p4_winrate <- ggplot(win_loss_matrix, aes(x = factor(Period), y = Strategy, fill = Result)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = ifelse(Win, "✓", "✗")), 
            color = "white", size = 4, fontface = "bold") +
  scale_fill_manual(values = c("勝出" = "#27ae60", "落後" = "#e67e22")) +
  labs(title = "逐期勝負記錄（相對於大盤）",
       subtitle = "✓ = 該期優於大盤，✗ = 該期落後大盤",
       x = "期數", y = "", fill = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        axis.text.x = element_text(size = 9),
        axis.text.y = element_text(size = 11, face = "bold"),
        legend.position = "bottom",
        panel.grid = element_blank())

print(p4_winrate)

# ==============================================================================
# 視覺化 5：風險-報酬散佈圖（含大盤位置）
# ==============================================================================

risk_return_data <- data.frame(
  Strategy = c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA", "Benchmark"),
  Risk = sapply(returns_history, sd) * sqrt(2) * 100,  # 年化標準差（百分比）
  Return = annual_ret * 100,  # 年化報酬（百分比）
  Is_Benchmark = c(rep("策略", 5), "大盤")
)

p5_scatter <- ggplot(risk_return_data, aes(x = Risk, y = Return, 
                                           color = Is_Benchmark, 
                                           shape = Is_Benchmark,
                                           size = Is_Benchmark)) +
  geom_point(alpha = 0.7) +
  geom_text(aes(label = Strategy), vjust = -1, size = 3.5, show.legend = FALSE) +
  scale_size_manual(values = c("策略" = 4, "大盤" = 6)) +
  scale_color_manual(values = c("策略" = "#3498db", "大盤" = "#e74c3c")) +
  scale_shape_manual(values = c("策略" = 16, "大盤" = 17)) +
  labs(title = "風險-報酬分佈（年化）",
       subtitle = "三角形為大盤位置",
       x = "年化風險 (標準差 %)", y = "年化報酬 (%)",
       color = "", shape = "", size = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

print(p5_scatter)

# ==============================================================================
# 視覺化 6：相對績效走勢（策略 / 大盤）
# ==============================================================================

# 計算相對績效：僅保留單筆投入策略（排除 DCA，因為資金節奏不同無法公平比較）
# 公式：策略市值 / 大盤市值（初始資本相同，直接相除即為相對倍數）
relative_perf <- results_wealth %>%
  select(Date, Max_Return, Min_Risk, Trade_Off, Equal_Weight, Benchmark) %>%
  mutate(
    Max_Return   = Max_Return   / Benchmark,
    Min_Risk     = Min_Risk     / Benchmark,
    Trade_Off    = Trade_Off    / Benchmark,
    Equal_Weight = Equal_Weight / Benchmark
  ) %>%
  select(-Benchmark) %>%
  pivot_longer(-Date, names_to = "Strategy", values_to = "Relative_Value")

p6_relative <- ggplot(relative_perf, aes(x = Date, y = Relative_Value, color = Strategy)) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  annotate("text", x = max(relative_perf$Date), y = 1, 
           label = "大盤基準線 (= 1.0)", hjust = 1, vjust = -0.5, size = 3.5) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "相對績效走勢（單筆策略 / 大盤）",
       subtitle = "大於 1.0 表示優於大盤，小於 1.0 表示落後大盤（DCA 因資金節奏不同，不納入此圖）",  # ← 更新
       x = "Date", y = "相對績效倍數", color = "Strategy") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        legend.position = "bottom")

print(p6_relative)

# ==============================================================================
# 視覺化 7：回撤比較（含大盤）
# ==============================================================================

dd_data <- results_wealth %>%
  mutate(across(-Date, ~ (. - cummax(.)) / cummax(.))) %>%
  pivot_longer(-Date, names_to = "Strategy", values_to = "Drawdown") %>%
  mutate(Is_Benchmark = ifelse(Strategy == "Benchmark", "大盤", "策略"))

dd_min_points <- dd_data %>%
  group_by(Strategy) %>%
  filter(Drawdown == min(Drawdown)) %>%
  slice(1)

p7_drawdown <- ggplot(dd_data, aes(x = Date, y = Drawdown, color = Strategy,
                                   linetype = Is_Benchmark, size = Is_Benchmark)) +
  geom_line(alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_point(data = dd_min_points, size = 3, show.legend = FALSE) +
  geom_label(data = dd_min_points, 
             aes(label = percent(Drawdown, 0.1)),
             nudge_y = -0.02, size = 2.5, show.legend = FALSE) +
  scale_size_manual(values = c("策略" = 0.8, "大盤" = 1.2)) +
  scale_linetype_manual(values = c("策略" = "solid", "大盤" = "dashed")) +
  scale_y_continuous(labels = percent) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(title = "回撤比較（含大盤）",
       subtitle = "虛線粗線為大盤回撤，標註點為最大回撤位置",
       x = "Date", y = "Drawdown", color = "Strategy") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        legend.position = "bottom") +
  guides(linetype = "none", size = "none")

print(p7_drawdown)

# ==============================================================================
# 統計檢定：策略是否顯著優於大盤
# ==============================================================================

cat("\n【統計顯著性檢定（t-test）】\n")
cat("H0: 策略報酬 = 大盤報酬，Ha: 策略報酬 ≠ 大盤報酬\n\n")

for (i in 1:5) {
  strategy_name <- c("Max_Return", "Min_Risk", "Trade_Off", "Equal_Weight", "DCA")[i]
  strategy_returns <- returns_history[[c("max", "min", "trade", "equal", "dca")[i]]]
  
  test_result <- t.test(strategy_returns, returns_history$benchmark, paired = TRUE)
  
  cat(sprintf("【%s】\n", strategy_name))
  cat(sprintf("  平均差異: %.4f (%.2f%%)\n", 
              test_result$estimate, test_result$estimate * 100))
  cat(sprintf("  t-統計量: %.3f\n", test_result$statistic))
  cat(sprintf("  p-value: %.4f %s\n", 
              test_result$p.value,
              ifelse(test_result$p.value < 0.05, "*** 顯著", "")))
  cat(sprintf("  結論: %s\n\n", 
              ifelse(test_result$p.value < 0.05,
                     ifelse(test_result$estimate > 0, "顯著優於大盤", "顯著落後大盤"),
                     "與大盤無顯著差異")))
}

# ==============================================================================
# 匯出結果（含大盤比較）
# ==============================================================================

OUTPUT_DIR <- "Portfolio_Backtest_Results"
PLOT_DIR <- file.path(OUTPUT_DIR, "Plots")
ANALYSIS_DIR <- file.path(OUTPUT_DIR, "Benchmark_Analysis")

dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(PLOT_DIR, showWarnings = FALSE)
dir.create(ANALYSIS_DIR, showWarnings = FALSE)

 # 產生平均權重表 (avg_weights_table)
avg_weights_table <- data.frame(
  Stock = STOCKS,
  Stock_Name = STOCK_NAMES,
  Max_Return = colMeans(weights_history$max[, -1]),
  Min_Risk = colMeans(weights_history$min[, -1]),
  Trade_Off = colMeans(weights_history$trade[, -1]),
  Equal_Weight = 1/length(STOCKS)
)

 # 準備權重歷史變數
weights_history_max   <- weights_history$max
weights_history_min   <- weights_history$min
weights_history_trade <- weights_history$trade
column_names <- c("Date", STOCKS)
colnames(weights_history_max)   <- column_names
colnames(weights_history_min)   <- column_names
colnames(weights_history_trade) <- column_names

# CSV 匯出
# --- 核心績效與數據 ---
write.csv(performance_summary, file.path(OUTPUT_DIR, "01_績效總表_含大盤.csv"), row.names = FALSE)
write.csv(results_wealth,      file.path(OUTPUT_DIR, "02_每期財富淨值序列.csv"), row.names = FALSE)
write.csv(avg_weights_table,   file.path(OUTPUT_DIR, "03_各股票平均權重.csv"), row.names = FALSE)

# --- 詳細權重歷史 ---
write.csv(weights_history_max,   file.path(OUTPUT_DIR, "04_權重歷史_最大報酬.csv"), row.names = FALSE)
write.csv(weights_history_min,   file.path(OUTPUT_DIR, "05_權重歷史_最小風險.csv"), row.names = FALSE)
write.csv(weights_history_trade, file.path(OUTPUT_DIR, "06_權重歷史_平衡策略.csv"), row.names = FALSE)

# --- 進階指標分析 (Benchmark_Analysis 子資料夾) ---
write.csv(alpha_df,        file.path(ANALYSIS_DIR, "alpha_分析.csv"), row.names = FALSE)
write.csv(ir_df,           file.path(ANALYSIS_DIR, "資訊比率_分析.csv"), row.names = FALSE)
write.csv(wr_df,           file.path(ANALYSIS_DIR, "勝率_分析.csv"), row.names = FALSE)
write.csv(win_loss_matrix, file.path(ANALYSIS_DIR, "逐期勝負紀錄.csv"), row.names = FALSE)
write.csv(risk_return_data,file.path(ANALYSIS_DIR, "風險報酬分佈數據.csv"), row.names = FALSE)

# 匯出圖表
ggsave(file.path(PLOT_DIR, "00_correlation_matrix.png"), p0, width = 10, height = 8, dpi = 300)
ggsave(file.path(PLOT_DIR, "01_wealth_vs_benchmark.png"), p1, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "02_alpha_comparison.png"), p2_alpha, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "03_sharpe_vs_benchmark.png"), p3_sharpe, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "04_win_rate_heatmap.png"), p4_winrate, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "05_risk_return_scatter.png"), p5_scatter, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "06_relative_performance.png"), p6_relative, width = 10, height = 6, dpi = 300)
ggsave(file.path(PLOT_DIR, "07_drawdown_vs_benchmark.png"), p7_drawdown, width = 10, height = 6, dpi = 300)

cat(sprintf("\n✓ 大盤比較分析完成！\n"))
cat(sprintf("✓ 績效報告已儲存至: %s\n", OUTPUT_DIR))
cat(sprintf("✓ 圖表已儲存至: %s\n", PLOT_DIR))
cat(sprintf("✓ 詳細分析已儲存至: %s\n", ANALYSIS_DIR))
cat("\n完成！\n")

# ==============================================================================

