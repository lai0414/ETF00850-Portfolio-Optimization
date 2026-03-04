# ============================================================
## Q6-1
## 以 electricity_data 資料集為基礎（需額外保留 10% 作為緩衝，且核能比例不得超過 10%），
## 使用 動態最佳化（四年訓練、一季測試），比較三種政策下的最佳解：
## 成本導向（安裝成本 70%、碳排放 30%）
## 碳排導向（安裝成本 30%、碳排放 70%）
## 平均權重（各 50%）

## Q6-2
## 以 electricity dataset 為基礎，針對上述兩種政策，模擬以下三種容量擴充情境：
## 太陽能增加 100%
## 風力發電增加 100%
## 核能增加 100%
## 並與原始最佳解（僅一次性最佳化）進行比較。

# ============================================================
# 電力最佳化分析 - 【程式整體流程說明】
# ============================================================
#   Step 1  讀資料，換算每月每小時平均需求（converted_demand）
#   Step 2  定義 LP 函數 bi_objective_lp()
#   Step 3  定義動態分配函數 dynamic_alloc()
#   Step 4  定義滑動視窗函數 rolling_dynamic()
#   Step 5  定義一次性最佳化函數 oneshot_lp()
#   Step 6  定義摘要函數 summarize_results()
#   Step 7  執行 6-1：三種政策 × 動態最佳化
#   Step 8  執行 6-2：兩種政策 × 四種情境 × 一次性+動態比較
#   Step 9  輸出 Excel
# ==============================================================

library(lpSolve)    # 線性規劃：lp()
library(writexl)    # 寫出 Excel：write_xlsx()
library(dplyr)      # 資料整理：mutate / filter / summarise / %>%
library(readxl)     # 讀入 Excel：read_xlsx()
library(lubridate)  # 日期工具：ymd() / days_in_month() / year()


# ==============================================================
# 全域常數
# ==============================================================

# 能源名稱向量（順序固定，後面所有向量與矩陣欄位都依此對齊）
energy_names <- c("coal", "gas", "nuclear", "wind", "solar")

# 碳排放係數（g CO₂ / kWh），來源：Excel 雙目標分頁 Row 3
# gas = 400（Excel 值）；原始 R 程式碼為 490，以 Excel 為準
carbon_g_kwh <- c(820, 400, 12, 12, 48)
#               coal  gas  nuc wind solar


# ==============================================================
# Step 1：讀資料與需求換算
# ==============================================================

raw  <- read_xlsx("electricity_data.xlsx")

# raw 讀入後的結構（132 列 × 17 欄，每列 = 一個月）：
#
#   Date        coal_cost  gas_cost  ...  demand      coal_capacity  ...
#   2014-01-01  1.79       3.00           19274214    17401469
#   2014-02-01  1.74       2.98           17553510    17401469
#   ...（到 2024-12-01，共 132 列）
#
# demand 欄位單位：MWh / 月（當月總用電量）
# capacity 欄位單位：kWh/hr（裝置容量，代表每小時最多能發多少電）

data <- raw %>%
  mutate(
    # ymd()：將日期字串轉為 R 的 Date 物件
    #   例如 "2014-01-01" → <Date> 2014-01-01
    Date_ad = ymd(Date),
    
    # days_in_month()：回傳該月的天數
    #   例如 2014-01 → 31，2014-02 → 28，2016-02 → 29（閏年）
    days_in_month = days_in_month(Date_ad),
    
    # converted_demand：換算成每小時平均需求（kWh/hr），與 capacity 單位一致
    #
    # 換算邏輯（逐步說明）：
    #   demand（MWh/月）
    #   × 1000             → kWh/月    （1 MWh = 1000 kWh）
    #   × 1.1              → 加入 10% 備轉緩衝（題目要求「額外保留10%」）
    #   ÷ 24               → kWh/天    （一天 24 小時）
    #   ÷ days_in_month    → kWh/hr    （除以當月天數，得每小時均值）
    #
    # 數值範例（2014-01，demand = 19,274,214 MWh，31天）：
    #   = 1.1 × 19,274,214 × 1000 / 24 / 31
    #   ≈ 28,496,822  kWh/hr
    converted_demand = 1.1 * demand * 1000 / 24 / days_in_month
  )

# mutate 後的 data 多了三欄：Date_ad、days_in_month、converted_demand
# 其餘欄位（成本、容量等）維持原樣，共 132 列 × 20 欄


# ==============================================================
# Step 2：雙目標 LP 函數 bi_objective_lp()
# ==============================================================
#
# ──────────────────────────────────────────────────────────────
# 【LP 問題設定】
#
# 決策變數（6 個）：
#   x1=coal, x2=gas, x3=nuclear, x4=wind, x5=solar（發電量，kWh/hr）
#   x6=slack（再生能源違規量，軟性限制用，kWh/hr）
#
# 目標函數（最小化加權成本）：
#   min  Σᵢ [ratio × cost_i + (1-ratio) × co_cost_i] × xᵢ  +  1000 × slack
#
# 限制式（9 條）：
#
#   硬性（必須滿足）：
#     C1~C5  xi ≤ capacity_i          各能源不超過容量         Excel Row11~15
#     C6     Σxi ≤ 1.15 × demand      總發電不超過需求 115%    Excel Row16
#     C7     Σxi ≥ 1.07 × demand      總發電至少需求 107%      Excel Row19
#     C8     x3 ≤ 0.10 × demand       核能不超過需求 10%        題目要求
#
#   軟性（盡力達到，違反時 slack > 0 被懲罰）：
#     C9     -0.15×(x1+x2+x3) + 0.85×(x4+x5) + slack ≥ 0
#            ↑ 整理自「再生能源(wind+solar)佔比 ≥ 15%」，係數來自 Excel Row20
#
#   【C9 為何軟性？】
#   視窗 1~17（2014~2021）wind+solar 容量僅 5%~15% 需求，
#   若設硬性前 17 輪全部無解。slack 讓 LP 永遠可解，
#   再生不足時罰 1000×slack，強制盡量提高再生比例。
#
# 限制矩陣示意（9 列 × 6 欄）：
#
#          coal   gas   nuc  wind solar slack    方向    RHS
#   C1:  [  1     0     0    0    0     0  ]  <=  coal_cap
#   C2:  [  0     1     0    0    0     0  ]  <=  gas_cap
#   C3:  [  0     0     1    0    0     0  ]  <=  nuc_cap  ← 已含 10% 上限
#   C4:  [  0     0     0    1    0     0  ]  <=  wind_cap
#   C5:  [  0     0     0    0    1     0  ]  <=  solar_cap
#   C6:  [  1     1     1    1    1     0  ]  <=  1.15×D
#   C7:  [  1     1     1    1    1     0  ]  >=  1.07×D
#   C8:  [  0     0     1    0    0     0  ]  <=  0.10×D
#   C9:  [-0.15 -0.15 -0.15 0.85 0.85  1  ]  >=  0
# ──────────────────────────────────────────────────────────────

bi_objective_lp <- function(ratio, cost_list, co_cost_list, capacity, demand) {
  # 參數說明：
  #   ratio       : 安裝成本權重（0~1），碳排成本權重 = 1-ratio
  #   cost_list   : 各能源單位安裝成本，長度 5 的向量
  #   co_cost_list: 各能源單位碳排成本，長度 5 的向量
  #   capacity    : 各能源容量上限，長度 5 的向量
  #   demand      : 當期需求總量（純量）
  
  names(cost_list)    <- energy_names
  names(co_cost_list) <- energy_names
  names(capacity)     <- energy_names
  
  # 核能容量上限 = min(實際容量, 需求×10%)
  # 例如：capacity["nuclear"]=1.53e8，0.1×demand=1.53e8 → 取較小者
  # 這樣 LP 本身的 C3 就已隱含核能不超過 10%，C8 是雙重保障
  capacity["nuclear"] <- min(capacity["nuclear"], 0.10 * demand)
  
  # ── 目標函數係數向量（長度 6）────────────────────────────────
  # 前 5 個：各能源的加權成本係數
  #   obj[i] = ratio × cost_i + (1-ratio) × co_cost_i
  #
  # 數值範例（ratio=0.7，成本導向）：
  #   cost_list    = [1.85, 3.00, 1.18, 2.65, 4.68]
  #   co_cost_list = [1.91, 1.14, 0.03, 0.03, 0.11]
  #   obj[1~5] = 0.7×[1.85,3.00,1.18,2.65,4.68] + 0.3×[1.91,1.14,0.03,0.03,0.11]
  #            = [1.87, 2.44, 0.83, 1.86, 3.31]
  #   obj[6]   = 1000（slack 懲罰，固定值）
  obj <- c(ratio * cost_list + (1 - ratio) * co_cost_list, 1000)
  
  # ── 限制矩陣（9 列 × 6 欄）──────────────────────────────────
  #
  # 建構方式：
  #   diag(5)            → 5×5 單位矩陣（對角線全1，其餘0）
  #   cbind(diag(5), 0)  → 右側加一欄 0（slack 不影響 C1~C5）
  #   rep(1, 5)          → 長度5的全1向量，代表「各能源發電量加總」
  #   rbind(...)         → 把各條限制式垂直堆疊成完整矩陣
  #
  # diag(5) 長這樣：
  #   [1 0 0 0 0]
  #   [0 1 0 0 0]
  #   [0 0 1 0 0]
  #   [0 0 0 1 0]
  #   [0 0 0 0 1]
  const_mat <- rbind(
    cbind(diag(5), rep(0, 5)),              # C1~C5
    c(rep(1, 5), 0),                        # C6
    c(rep(1, 5), 0),                        # C7
    c(0, 0, 1, 0, 0, 0),                   # C8
    c(-0.15, -0.15, -0.15, 0.85, 0.85, 1)  # C9
  )
  # const_mat 印出來（9×6）：
  #      [,1]  [,2]  [,3]  [,4]  [,5]  [,6]
  # [1,]  1.00  0.00  0.00  0.00  0.00  0.00  ← C1
  # [2,]  0.00  1.00  0.00  0.00  0.00  0.00  ← C2
  # [3,]  0.00  0.00  1.00  0.00  0.00  0.00  ← C3
  # [4,]  0.00  0.00  0.00  1.00  0.00  0.00  ← C4
  # [5,]  0.00  0.00  0.00  0.00  1.00  0.00  ← C5
  # [6,]  1.00  1.00  1.00  1.00  1.00  0.00  ← C6
  # [7,]  1.00  1.00  1.00  1.00  1.00  0.00  ← C7
  # [8,]  0.00  0.00  1.00  0.00  0.00  0.00  ← C8
  # [9,] -0.15 -0.15 -0.15  0.85  0.85  1.00  ← C9
  
  const_dir <- c(rep("<=", 5), "<=", ">=", "<=", ">=")
  #              C1~C5        C6    C7    C8    C9
  
  # const_rhs：各條限制式的右手邊數值（長度 9）
  # c(capacity, ...) 先展開 capacity 向量的 5 個數值，再接後面 4 個
  const_rhs <- c(
    capacity,       # C1~C5 的 RHS（5 個容量值）
    1.15 * demand,  # C6 的 RHS（Excel Row16）
    1.07 * demand,  # C7 的 RHS（Excel Row19）
    0.10 * demand,  # C8 的 RHS（核能上限）
    0               # C9 的 RHS（再生能源係數不等式 ≥ 0）
  )
  
  # lp("min", 目標係數, 限制矩陣, 方向向量, RHS 向量)
  res <- lp("min", obj, const_mat, const_dir, const_rhs)
  
  if (res$status == 0) {
    # res$solution 長度 6：前 5 個是各能源發電量，第 6 個是 slack
    sol      <- res$solution[1:5]
    names(sol) <- energy_names
    slack_re <- res$solution[6]   # 0 = 再生達標；> 0 = 不足（被罰）
    
    # sol 數值範例（視窗 1，成本導向 ratio=0.7）：
    #   coal       gas   nuclear      wind     solar
    #   8.31e8    0.00   1.53e8    3.16e7    4.28e7
    # （煤炭用滿、核能達 10% 上限、氣和再生依成本分配）
    
    total_cost      <- sum(sol * cost_list)
    total_co_cost   <- sum(sol * co_cost_list)
    total_output    <- sum(sol)
    # sol 單位 kWh/hr，carbon_g_kwh 單位 g/kWh
    # sol × carbon_g_kwh = g/hr，÷ 1e6 → 公噸/hr
    carbon_emission <- sum(sol * carbon_g_kwh) / 1e6
    
    # 回傳 1 列 × 17 欄 data.frame
    return(data.frame(
      Objective_Value     = sum(sol * (ratio * cost_list + (1 - ratio) * co_cost_list)),
      Installation_Cost   = total_cost,
      Carbon_Cost         = total_co_cost,
      Slack_RE            = slack_re,
      Total_Power_Output  = total_output,
      Total_Cost_per_kWh  = (total_cost + total_co_cost) / total_output,
      Carbon_Emission_ton = carbon_emission,
      coal          = sol["coal"],
      gas           = sol["gas"],
      nuclear       = sol["nuclear"],
      wind          = sol["wind"],
      solar         = sol["solar"],
      ratio_coal    = sol["coal"]    / total_output,
      ratio_gas     = sol["gas"]     / total_output,
      ratio_nuclear = sol["nuclear"] / total_output,
      ratio_wind    = sol["wind"]    / total_output,
      ratio_solar   = sol["solar"]   / total_output
    ))
  } else {
    # 有 slack 機制理論上必定有解；若仍無解代表容量不足
    warning(sprintf("LP 無解：demand=%.2e, sum_cap=%.2e", demand, sum(capacity)))
    return(NULL)
  }
}


# ==============================================================
# Step 3：動態分配函數 dynamic_alloc()
# ==============================================================
#
# 【做什麼】
#   測試期的每一個月：
#   ① 初始配置 = 當月需求 × 訓練集求得的最佳比例 tr_ratio
#   ② 若某能源超出容量上限 → 封頂，把超出量依成本倒數比例
#      分配給還有空間的能源
#   ③ 重複②直到沒有超出為止（repeat 迴圈）
#
# 【repeat 迴圈每一輪的流程】
#   ① exceed_mask  = 哪些能源超出容量（邏輯向量）
#   ② exceed_sum   = 超出總量
#   ③ 若 exceed_sum ≈ 0 → break，結束迴圈
#   ④ 超出能源封頂（設為容量上限）
#   ⑤ 找有空間的能源（available_mask）
#   ⑥ 以加權成本倒數為權重，計算分配比例 ts_ratio
#   ⑦ alloc += exceed_sum × ts_ratio
#   ⑧ 若補充後又超出 → 回到①繼續

dynamic_alloc <- function(tr_ratio, row, inst_weight, carb_weight) {
  # tr_ratio    : 訓練集最佳能源比例，長度 5，加總 = 1
  # row         : 測試集某一月的資料列（1 列 data.frame）
  # inst_weight : 安裝成本權重
  # carb_weight : 碳排放成本權重（= 1 - inst_weight）
  
  demand <- row$converted_demand
  
  # paste0(energy_names, "_cost") 展開為：
  #   c("coal_cost", "gas_cost", "nuclear_cost", "wind_cost", "solar_cost")
  # row[ 向量 ] 一次取出多欄，as.numeric() 轉成純數值向量
  cost_list    <- as.numeric(row[paste0(energy_names, "_cost")])
  co_cost_list <- as.numeric(row[paste0(energy_names, "_carbon_cost")])
  capacity_lim <- as.numeric(row[paste0(energy_names, "_capacity")])
  names(cost_list)    <- energy_names
  names(co_cost_list) <- energy_names
  names(capacity_lim) <- energy_names
  
  # 核能容量上限（與 LP 訓練端保持一致）
  capacity_lim["nuclear"] <- min(capacity_lim["nuclear"], 0.10 * demand)
  
  # 初始配置
  # 範例：demand=28,496,822，tr_ratio=c(0.52,0.20,0.10,0.02,0.16)
  #   alloc = c(14,818,347, 5,699,364, 2,849,682, 569,936, 4,559,492)
  alloc <- demand * tr_ratio
  
  # ── repeat 迴圈：迭代再分配 ──────────────────────────────────
  repeat {
    # exceed_mask：邏輯向量，TRUE = 超出容量
    # 數值範例：
    #   capacity_lim = c(17401469, 14469422, 1740147, 2811634, 4282000)
    #   alloc        = c(14818347,  5699364, 2849682,  569936, 4559492)
    #   exceed_mask  = c(FALSE,    FALSE,    TRUE,     FALSE,  TRUE   )
    #                                         ↑ 核能超      ↑ 太陽能超
    exceed_mask <- (alloc > capacity_lim)
    
    # 超出總量（只加 exceed_mask=TRUE 的部分）
    # alloc - capacity_lim = c(-2583122, -8770058, +1109535, -2241698, +277492)
    # 只取正值加總：1109535 + 277492 = 1387027
    exceed_sum <- sum((alloc - capacity_lim)[exceed_mask])
    
    if (exceed_sum <= 1e-6) break  # 無超出 → 完成
    
    alloc[exceed_mask] <- capacity_lim[exceed_mask]  # 超出能源封頂
    
    # 找有空間的能源及其剩餘空間
    available_mask <- (capacity_lim > alloc)
    available_room <- rep(0, length(capacity_lim))
    available_room[available_mask] <- capacity_lim[available_mask] - alloc[available_mask]
    
    if (sum(available_room) <= 1e-6) {
      warning(paste0("期別 ", row$Date, "：總容量不足，需求無法完全滿足"))
      break
    }
    
    # 計算分配比例（成本倒數加權）
    # 參照原始 R 程式碼：先建全零向量，只填 available 位置
    #
    # 範例：available_mask = c(TRUE,TRUE,FALSE,FALSE,FALSE)（只有煤和氣有空間）
    #   total_cost_vec[available] = [1.87, 2.44]（加權成本）
    #   weight_cost = [1/1.87, 1/2.44] = [0.535, 0.410]
    #   sum(weight_cost) = 0.945
    #   ts_ratio[available] = [0.535/0.945, 0.410/0.945] = [0.566, 0.434]
    #
    # 最後：alloc += 1387027 × ts_ratio
    #   alloc["coal"] += 1387027 × 0.566 ≈ 785,057
    #   alloc["gas"]  += 1387027 × 0.434 ≈ 601,970
    total_cost_vec  <- inst_weight * cost_list + carb_weight * co_cost_list
    weight_cost     <- 1 / total_cost_vec[available_mask]
    ts_ratio        <- rep(0, length(energy_names))
    names(ts_ratio) <- energy_names
    ts_ratio[available_mask] <- weight_cost / sum(weight_cost)
    
    alloc <- alloc + exceed_sum * ts_ratio
    # 若補充後又有超出 → 下一輪 repeat 繼續
  }
  
  # ── 計算統計量 ────────────────────────────────────────────────
  total_cost      <- sum(alloc * cost_list)
  total_co_cost   <- sum(alloc * co_cost_list)
  total_output    <- sum(alloc)
  carbon_emission <- sum(alloc * carbon_g_kwh) / 1e6
  ratio_vec       <- alloc / total_output
  
  # 回傳 1 列 data.frame（對應測試期一個月）
  # 多個月整合後每列 = 一個月，大概長這樣：
  #   Date        Demand     Obj_Value  Install  Carbon  Output  ...  r_coal r_gas ...
  #   2018-01-01  28496822   5.8e10     3.5e10   2.3e10  2.85e7  ...  0.52   0.20  ...
  data.frame(
    Date                = row$Date,
    Demand              = demand,
    Objective_Value     = inst_weight * total_cost + carb_weight * total_co_cost,
    Installation_Cost   = total_cost,
    Carbon_Cost         = total_co_cost,
    Total_Power_Output  = total_output,
    Total_Cost_per_kWh  = (total_cost + total_co_cost) / total_output,
    Carbon_Emission_ton = carbon_emission,
    coal          = alloc["coal"],
    gas           = alloc["gas"],
    nuclear       = alloc["nuclear"],
    wind          = alloc["wind"],
    solar         = alloc["solar"],
    ratio_coal    = ratio_vec["coal"],
    ratio_gas     = ratio_vec["gas"],
    ratio_nuclear = ratio_vec["nuclear"],
    ratio_wind    = ratio_vec["wind"],
    ratio_solar   = ratio_vec["solar"]
  )
}


# ==============================================================
# Step 4：滑動視窗主函數 rolling_dynamic()
# ==============================================================
#
# 【滑動視窗示意（固定 48 月訓練 / 3 月測試）】
#
#  視窗  訓練集（48 個月）              測試集（3 個月）
#  ────────────────────────────────────────────────────────
#   1    月 01~48  2014-01 ~ 2017-12    月 49~51  2018-01~03
#   2    月 04~51  2014-04 ~ 2018-03    月 52~54  2018-04~06
#   3    月 07~54  2014-07 ~ 2018-06    月 55~57  2018-07~09
#   ...
#  28    月 82~129 2020-10 ~ 2024-09    月 130~132  2024-10~12
#
#  總共 28 個視窗，84 個測試月（= 7 年 × 12 個月）
#
# 【每輪做的事】
#   ① 從訓練集計算統計量（median 安裝成本、最後2年碳成本、容量和、需求和）
#   ② 呼叫 bi_objective_lp() → 得到最佳比例 tr_ratio
#   ③ 對測試季 3 個月各呼叫一次 dynamic_alloc()，收集結果
#   ④ train_start_idx += 3，視窗往後滑動一季

rolling_dynamic <- function(data, inst_weight, carb_weight,
                            capacity_modifier = NULL) {
  # capacity_modifier：命名數值向量，NULL = 不修改
  #   例如 c(solar = 2.0) → 太陽能容量全部乘以 2（+100%）
  
  carb_weight     <- 1 - inst_weight  # 強制確保兩者加總 = 1
  train_window    <- 48               # 訓練視窗長度（月）
  test_window     <- 3                # 測試視窗長度（月）
  results_all     <- list()           # 收集每月結果（list of data.frame）
  period_idx      <- 0                # 測試季計數器
  train_start_idx <- 1                # 訓練視窗起始列號
  
  # while 迴圈：每次檢查「訓練集結束後還有測試資料」
  # 條件 (train_start_idx + 48) <= 132：
  #   視窗 1：(1+48)=49 ≤ 132  ✓
  #   視窗 28：(82+48)=130 ≤ 132 ✓
  #   視窗 29：(85+48)=133 > 132  ✗ → 停止
  while ((train_start_idx + train_window) <= nrow(data)) {
    
    # 計算各索引
    train_end_idx  <- train_start_idx + train_window - 1   # 訓練集最後一列
    test_start_idx <- train_end_idx + 1                    # 測試集第一列
    test_end_idx   <- min(test_start_idx + test_window - 1, nrow(data))
    
    train_data <- data[train_start_idx:train_end_idx, ]   # 固定 48 列
    test_data  <- data[test_start_idx:test_end_idx, ]     # 通常 3 列
    
    # ── 訓練集統計量 ──────────────────────────────────────────
    
    # sapply(向量, 函數)：對向量每個元素套用函數，回傳等長命名向量
    # 結果範例（視窗 1）：
    #   coal   gas  nuclear  wind  solar
    #   1.85  3.00    1.18  2.65   4.68
    avg_cost <- sapply(energy_names, function(x)
      median(train_data[[paste0(x, "_cost")]]))
    
    # 碳排成本：只取訓練集最後兩年（反映近期碳價水準）
    # year() 從 Date 物件提取年份整數；%in% 判斷是否屬於向量中的值
    end_yr      <- year(max(train_data$Date_ad))
    last_two    <- train_data %>% filter(year(Date_ad) %in% c(end_yr - 1, end_yr))
    avg_co_cost <- sapply(energy_names, function(x)
      mean(last_two[[paste0(x, "_carbon_cost")]], na.rm = TRUE))
    
    # 容量加總（48 個月的月容量全部加總）
    # 例如 coal：17,401,469 × 48 ≈ 8.35e8
    sum_capacity <- sapply(energy_names, function(x)
      sum(train_data[[paste0(x, "_capacity")]]))
    
    # 容量擴充情境：對指定能源乘以修改係數
    if (!is.null(capacity_modifier)) {
      for (nm in names(capacity_modifier)) {
        sum_capacity[nm] <- sum_capacity[nm] * capacity_modifier[nm]
      }
    }
    
    # 需求加總（48 個月的 converted_demand 全部加總）
    sum_demand <- sum(train_data$converted_demand)
    
    # ── 訓練集 LP ─────────────────────────────────────────────
    opt_sol <- bi_objective_lp(
      ratio        = inst_weight,
      cost_list    = avg_cost,
      co_cost_list = avg_co_cost,
      capacity     = sum_capacity,
      demand       = sum_demand
    )
    
    # 有 slack 必定有解；若無解立即 stop（不跳過）
    if (is.null(opt_sol)) {
      stop(sprintf("LP 無解：訓練期 %s ~ %s，請檢查資料",
                   min(train_data$Date_ad), max(train_data$Date_ad)))
    }
    
    # 提取最佳比例（以各能源發電量佔總量的比例表示）
    # as.numeric(opt_sol[, energy_names])：從 1列 df 取出 5 個欄的值 → 向量
    tr_alloc        <- as.numeric(opt_sol[, energy_names])
    names(tr_alloc) <- energy_names
    tr_ratio        <- tr_alloc / sum(tr_alloc)  # 正規化，加總 = 1
    
    # ── 測試季：逐月動態分配 ──────────────────────────────────
    period_idx <- period_idx + 1  # 測試季編號 +1
    
    # for 迴圈跑 i = 1, 2, 3（測試季的 3 個月）
    # 每一輪取出一列（test_data[i, ]），呼叫 dynamic_alloc()
    for (i in 1:nrow(test_data)) {
      row <- test_data[i, ]
      
      # 測試期容量也要同步修改
      if (!is.null(capacity_modifier)) {
        for (nm in names(capacity_modifier)) {
          row[[paste0(nm, "_capacity")]] <-
            row[[paste0(nm, "_capacity")]] * capacity_modifier[nm]
        }
      }
      
      res_row             <- dynamic_alloc(tr_ratio, row, inst_weight, carb_weight)
      res_row$Period      <- period_idx
      res_row$Train_Start <- min(train_data$Date_ad)
      res_row$Train_End   <- max(train_data$Date_ad)
      
      # append(list, list(新元素))：把 1 列結果加入 list 末尾
      results_all <- append(results_all, list(res_row))
    }
    
    train_start_idx <- train_start_idx + test_window  # 視窗往後滑動 3 個月
  }
  
  # do.call(rbind, list)：把 list 中的 84 個 1列 data.frame 垂直合併
  # 最終得到 84 列 × 21 欄 data.frame，大概長這樣：
  #
  #   Date        Demand   Obj_Value  ...  r_coal r_gas ... Period  Train_End
  #   2018-01-01  28496822  5.8e10   ...    0.52  0.20       1     2017-12-01
  #   2018-02-01  27123000  5.7e10   ...    0.52  0.20       1     2017-12-01
  #   2018-03-01  28001000  5.7e10   ...    0.52  0.20       1     2017-12-01
  #   2018-04-01  28500000  5.8e10   ...    0.51  0.21       2     2018-03-01  ← 視窗2
  #   ...（共 84 列）
  final           <- do.call(rbind, results_all)
  rownames(final) <- NULL
  final
}


# ==============================================================
# Step 5：一次性最佳化函數 oneshot_lp()
# ==============================================================
# 不做滑動視窗，用全資料一次求解（對照基準）
# 統計量計算邏輯與 rolling_dynamic 的訓練端相同

oneshot_lp <- function(data_use, inst_weight) {
  avg_cost <- sapply(energy_names, function(x)
    median(data_use[[paste0(x, "_cost")]]))
  
  end_yr      <- year(max(data_use$Date_ad))
  last_two    <- data_use %>% filter(year(Date_ad) %in% c(end_yr - 1, end_yr))
  avg_co_cost <- sapply(energy_names, function(x)
    mean(last_two[[paste0(x, "_carbon_cost")]], na.rm = TRUE))
  
  sum_cap    <- sapply(energy_names, function(x) sum(data_use[[paste0(x, "_capacity")]]))
  sum_demand <- sum(data_use$converted_demand)
  
  opt <- bi_objective_lp(inst_weight, avg_cost, avg_co_cost, sum_cap, sum_demand)
  if (!is.null(opt)) {
    opt$inst_weight <- inst_weight
    opt$carb_weight <- 1 - inst_weight
  }
  opt  # 回傳 1 列 data.frame（全資料一次性最佳解）
}


# ==============================================================
# Step 6：摘要統計函數 summarize_results()
# ==============================================================
# 把 84 列逐月結果壓縮成 1 列彙總統計

summarize_results <- function(df, label = "") {
  # summarise()：對整個 df 計算摘要統計，回傳 1 列 data.frame
  s <- df %>% summarise(
    N_Months                  = n(),
    Total_Objective_Value     = sum(Objective_Value,      na.rm = TRUE),
    Total_Installation_Cost   = sum(Installation_Cost,    na.rm = TRUE),
    Total_Carbon_Cost         = sum(Carbon_Cost,          na.rm = TRUE),
    Total_Carbon_Emission_ton = sum(Carbon_Emission_ton,  na.rm = TRUE),
    Total_Power_Output        = sum(Total_Power_Output,   na.rm = TRUE),
    Avg_Cost_per_kWh          = mean(Total_Cost_per_kWh, na.rm = TRUE),
    Avg_ratio_coal            = mean(ratio_coal,    na.rm = TRUE),
    Avg_ratio_gas             = mean(ratio_gas,     na.rm = TRUE),
    Avg_ratio_nuclear         = mean(ratio_nuclear, na.rm = TRUE),
    Avg_ratio_wind            = mean(ratio_wind,    na.rm = TRUE),
    Avg_ratio_solar           = mean(ratio_solar,   na.rm = TRUE)
  )
  # 回傳結果（1 列）範例：
  #   N_Months Total_Obj  Total_Install Total_Carbon ...  Avg_r_coal  Label
  #   84       6.8e12     4.3e12        2.5e12            0.52        成本導向
  if (nchar(label) > 0) s$Label <- label
  s
}


# ==============================================================
# Step 7：執行 6-1：三種政策 × 動態最佳化
# ==============================================================
cat("\n========== 6-1: 動態最佳化 × 三種政策 ==========\n")

policies <- list(
  cost_oriented   = list(inst = 0.7, label = "成本導向(70/30)"),
  carbon_oriented = list(inst = 0.3, label = "碳排導向(30/70)"),
  balanced        = list(inst = 0.5, label = "平均權重(50/50)")
)

results_6_1 <- list()  # key = 政策名，value = 84列 data.frame
summary_6_1 <- list()  # key = 政策名，value = 1列摘要

# for 迴圈：pname 依序取 "cost_oriented"、"carbon_oriented"、"balanced"
# names(policies) 回傳 c("cost_oriented","carbon_oriented","balanced")
# policies[[pname]] 取出對應的設定 list（含 inst 和 label）
for (pname in names(policies)) {
  p <- policies[[pname]]
  cat(sprintf("\n--- 政策: %s ---\n", p$label))
  
  res  <- rolling_dynamic(data, inst_weight = p$inst, carb_weight = 1 - p$inst)
  summ <- summarize_results(res, label = p$label)
  
  results_6_1[[pname]] <- res
  summary_6_1[[pname]] <- summ
  print(summ)
}

# do.call(rbind, summary_6_1)：把 3 個 1列摘要合併成 3 列比較表
summary_6_1_df <- do.call(rbind, summary_6_1)
cat("\n----- 6-1 三種政策彙總比較（3 列）-----\n")
print(summary_6_1_df)
# 輸出大概長這樣（每列 = 一種政策的整體績效）：
#
#   N_Months Total_Obj  Avg_r_coal  Avg_r_nuclear  Label
#   84       6.8e12     0.52        0.10           成本導向(70/30)
#   84       7.1e12     0.38        0.10           碳排導向(30/70)
#   84       6.9e12     0.45        0.10           平均權重(50/50)


# ==============================================================
# Step 8：執行 6-2：一次性 vs 動態 × 四種情境
# ==============================================================
cat("\n========== 6-2: 容量擴充情境分析 ==========\n")

scenarios <- list(
  baseline   = list(mod = NULL,              label = "原始（基準）"),
  solar_2x   = list(mod = c(solar   = 2.0),  label = "太陽能+100%"),
  wind_2x    = list(mod = c(wind    = 2.0),  label = "風力+100%"),
  nuclear_2x = list(mod = c(nuclear = 2.0),  label = "核能+100%")
)

target_policies <- list(
  cost_oriented   = list(inst = 0.7, label = "成本導向(70/30)"),
  carbon_oriented = list(inst = 0.3, label = "碳排導向(30/70)")
)

oneshot_rows <- list()  # 一次性最佳化結果（共 2×4=8 列）
dynamic_rows <- list()  # 動態最佳化彙總（共 8 列）
results_6_2  <- list()  # 動態最佳化逐月詳細（共 8 個 84列 df）

# 雙層 for 迴圈：外層 2 種政策，內層 4 種情境，共 8 組
for (pname in names(target_policies)) {
  p <- target_policies[[pname]]
  cat(sprintf("\n=== 政策: %s ===\n", p$label))
  
  for (sname in names(scenarios)) {
    s   <- scenarios[[sname]]
    key <- paste(pname, sname, sep = "_")  # 例如 "cost_oriented_solar_2x"
    cat(sprintf("  情境: %s\n", s$label))
    
    # 建立資料副本並修改容量（不影響原始 data）
    # 一次性最佳化需要用修改後的資料
    data_mod <- data
    if (!is.null(s$mod)) {
      for (nm in names(s$mod)) {
        data_mod[[paste0(nm, "_capacity")]] <-
          data_mod[[paste0(nm, "_capacity")]] * s$mod[nm]
      }
    }
    
    # 一次性最佳化（用 data_mod）
    one <- oneshot_lp(data_mod, p$inst)
    if (!is.null(one)) {
      one$Policy   <- p$label
      one$Scenario <- s$label
      one$Type     <- "一次性最佳化"
      oneshot_rows[[key]] <- one
    }
    
    # 動態最佳化（傳入 capacity_modifier，函數內部自行修改）
    dyn     <- rolling_dynamic(data,
                               inst_weight       = p$inst,
                               carb_weight       = 1 - p$inst,
                               capacity_modifier = s$mod)
    dyn_sum <- summarize_results(dyn, label = paste(p$label, s$label, sep = " | "))
    dyn_sum$Policy   <- p$label
    dyn_sum$Scenario <- s$label
    dyn_sum$Type     <- "動態最佳化"
    
    dynamic_rows[[key]] <- dyn_sum
    results_6_2[[key]]  <- dyn
  }
}

# 整合 8 組結果
# lapply(list, as.data.frame)：把 list 每個元素轉成 df，再 rbind
oneshot_df <- do.call(rbind, lapply(oneshot_rows, as.data.frame))
dynamic_df <- do.call(rbind, dynamic_rows)

cat("\n----- 6-2 一次性最佳化比較（8 列）-----\n")
oneshot_show <- oneshot_df %>%
  select(Policy, Scenario, Type,
         Objective_Value, Installation_Cost, Carbon_Cost,
         Carbon_Emission_ton, Total_Power_Output,
         ratio_coal, ratio_gas, ratio_nuclear, ratio_wind, ratio_solar)
print(as.data.frame(oneshot_show))

cat("\n----- 6-2 動態最佳化彙總比較（8 列）-----\n")
dynamic_show <- dynamic_df %>%
  select(Policy, Scenario, Type,
         Total_Objective_Value, Total_Installation_Cost, Total_Carbon_Cost,
         Total_Carbon_Emission_ton, Total_Power_Output,
         Avg_ratio_coal, Avg_ratio_gas, Avg_ratio_nuclear,
         Avg_ratio_wind, Avg_ratio_solar)
print(as.data.frame(dynamic_show))


# ==============================================================
# Step 9：輸出 Excel
# ==============================================================
cat("\n========== 輸出 Excel ==========\n")

# setNames(list, 名稱向量)：為 list 元素命名（用作 Excel 分頁名稱）
# lapply(names, function(n) list[[n]])：以名稱一一取出 df 組成新 list
sheets_6_1 <- setNames(
  lapply(names(results_6_1), function(n) results_6_1[[n]]),
  paste0("6-1_", names(results_6_1))
)
sheets_6_1[["6-1_Summary"]] <- summary_6_1_df

sheets_6_2 <- setNames(
  lapply(names(results_6_2), function(n) results_6_2[[n]]),
  paste0("6-2_", names(results_6_2))
)
sheets_6_2[["6-2_Oneshot_Compare"]] <- as.data.frame(oneshot_show)
sheets_6_2[["6-2_Dynamic_Compare"]] <- as.data.frame(dynamic_show)

# write_xlsx()：把命名 list 寫入 Excel，每個 list 元素 = 一個分頁
# c() 合併兩個命名 list
write_xlsx(c(sheets_6_1, sheets_6_2), "electricity_optimization_results.xlsx")
cat("完成！結果已儲存至 electricity_optimization_results.xlsx\n")

# 輸出的 Excel 包含以下分頁（共 14 個）：
#
#   6-1_cost_oriented          成本導向，84 列逐月結果
#   6-1_carbon_oriented        碳排導向，84 列逐月結果
#   6-1_balanced               平均權重，84 列逐月結果
#   6-1_Summary                三種政策彙總比較（3 列）
#   6-2_cost_oriented_baseline         成本×基準
#   6-2_cost_oriented_solar_2x         成本×太陽能+100%
#   6-2_cost_oriented_wind_2x          成本×風力+100%
#   6-2_cost_oriented_nuclear_2x       成本×核能+100%
#   6-2_carbon_oriented_baseline       碳排×基準
#   6-2_carbon_oriented_solar_2x       碳排×太陽能+100%
#   6-2_carbon_oriented_wind_2x        碳排×風力+100%
#   6-2_carbon_oriented_nuclear_2x     碳排×核能+100%
#   6-2_Oneshot_Compare        一次性最佳化 8 組比較
#   6-2_Dynamic_Compare        動態最佳化 8 組彙總比較