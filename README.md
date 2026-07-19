# PI-X-STGAT：物理引導、可解釋的交通壅塞預測

本專案提出 **Physics-Informed Explainable Spatial-Temporal Graph Attention Network（PI-X-STGAT）**，目標是在複雜城市路網中，同時提升交通壅塞預測的準確度、極端情境穩定性與可解釋性。研究以洛杉磯 HERE Maps API 資料為例，將道路區段建模為圖節點，預測未來的 **Jam Factor** 與 **Speed**。

> 論文題目：*A Physics-Informed Spatial-Temporal Graph Attention Model for Traffic Forecasting and Interpretable Congestion Propagation Analysis*

![PI-X-STGAT 模型架構](<paper/figure/Fig_Model Architecture Diagram.png>)

## 研究問題與方法

純資料驅動的時空圖神經網路在一般資料上雖有良好表現，但面對事故、假日或極端天氣等分布外（Out-of-Distribution, OoD）狀況時，可能產生不符合交通物理直覺的結果，而且很難說明哪些路段影響了預測。

PI-X-STGAT 由四個部分組成：

1. **Context-Aware GAT**：聚合道路鄰居的空間資訊與交通、天氣、時間等多模態特徵。
2. **Node Adaptive Parameter Learning（NAPL）**：以可學習的節點嵌入補足固定地理拓樸未涵蓋的隱藏關聯。
3. **GRU**：建模壅塞狀態隨時間演變的動態。
4. **Physics-guided loss**：以 Jam Factor 作為密度代理量，加入軟式交通流守恆殘差；它是正則化項，不是嚴格求解 LWR 偏微分方程。

模型保留空間注意力矩陣，用來提供候選影響路段與壅塞傳播關係的輔助線索。注意力代表模型學到的統計關聯，**不能直接視為因果證據**。

## 資料與實驗設定

| 項目 | 設定 |
| --- | --- |
| 資料來源 | HERE Maps API，洛杉磯都市道路網 |
| 路網規模 | 87 個道路節點、59,194 個時間點 |
| 輸入 | 18 維交通、道路屬性、天氣與週期時間特徵 |
| 輸出 | Jam Factor、Speed |
| 資料切分 | 依時間順序前 80% 訓練、後 20% 評估 |
| 預設歷史窗 | `seq_len=12` |
| 預測窗 | `horizon=1, 2, 3, 4` |
| 訓練 | Adam、learning rate `1e-3`、batch size 32 |
| 最終模型設定 | GAT 64、GRU 128、node embedding 10、`lambda_phy=0.015`（論文報告值） |

主要特徵定義在 `constants.py`，資料張量化與路網建構在 `dataset.py`。Jam Factor 在訓練目標中由 0–10 縮放至 0–1，計算 Jam 誤差時再乘回 10。

## 目前結果摘要

以下數值整理自現有的 `all_model_performance_for_plot.csv`，不是本次重新訓練的結果：

| 模型 | Jam MAE | Jam RMSE |
| --- | ---: | ---: |
| HA | 0.6287 | 1.9565 |
| LSTM | 0.1094 | 0.6185 |
| STGCN | 0.0668 | 0.6212 |
| ASTGCN | 0.0805 | 0.6214 |
| AGCRN | 0.0947 | 0.6234 |
| Graph WaveNet | 0.1176 | 0.6184 |
| MTGNN | 0.0613 | 0.6208 |
| **PI-X-STGAT** | **0.0595** | 0.6212 |

在目前實驗設定中，PI-X-STGAT 的 Jam MAE 最低。五個隨機種子實驗的 Jam MAE 為 `0.0669 ± 0.0063`；Labor Day 真實假日切片的 Jam MAE 為 `0.0597`。由於 Jam Factor 接近 0 時 MAPE 會數值爆增，專案中的 MAPE 僅適合當輔助指標，主要比較應以 MAE/RMSE 為主。

![模型效能比較](paper/figure/Fig_Performance_Comparison_All_Models.png)

## 專案結構

```text
CMES/
├── README.md                         # 研究與重現說明
├── constants.py                      # 輸入特徵與預測目標
├── dataset.py                        # 時空張量、KNN 路網與 DataLoader
├── model.py                          # PI-X-STGAT 與各基準模型
├── metrics.py                        # 物理守恆正則化
├── experiment_paths.py               # 統一圖片輸出位置
├── run_*.py / search_best_pix.py     # 可重現的主要實驗入口
├── *.csv                             # 已有實驗結果
├── pix_best_checkpoint.pt            # 目前最佳 PI-X-STGAT 權重
├── all_segments_preprocessed.zip     # 壓縮後資料集
├── paper/
│   ├── CMES_draft_rev.tex            # 最新修改時間的論文原始檔
│   └── figure/                        # 所有研究圖片與後續繪圖輸出
└── literature review/                # 相關文獻
```

所有繪圖程式現在都透過 `experiment_paths.py` 將 PNG/PDF 輸出至 `paper/figure/`，不再散落於專案根目錄。

## 圖片索引

| 圖片 | 內容 |
| --- | --- |
| [Fig_Model Architecture Diagram.png](<paper/figure/Fig_Model Architecture Diagram.png>) | PI-X-STGAT 整體架構 |
| [Fig_Performance_Comparison_All_Models.png](paper/figure/Fig_Performance_Comparison_All_Models.png) | 全模型 Jam MAE/RMSE 比較 |
| [Fig_Multistep_Forecasting.png](paper/figure/Fig_Multistep_Forecasting.png) | 多步預測誤差趨勢 |
| [Fig_Hyperparameter_Sensitivity.png](paper/figure/Fig_Hyperparameter_Sensitivity.png) | 物理損失權重敏感度 |
| [Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.png](paper/figure/Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.png) | GRU、節點嵌入與歷史窗敏感度 |
| [Fig_Ablation_Study.png](paper/figure/Fig_Ablation_Study.png) | 模組消融實驗 |
| [Fig_Error_Analysis.png](paper/figure/Fig_Error_Analysis.png) | 守恆殘差與預測誤差分布 |
| [Fig_Attention_Heatmap.png](paper/figure/Fig_Attention_Heatmap.png) | 全域空間注意力熱圖 |
| [Fig_Top_Attention_Edges.png](paper/figure/Fig_Top_Attention_Edges.png) | 高注意力道路連結 |
| [Fig_Real_Dynamic_Attention.png](paper/figure/Fig_Real_Dynamic_Attention.png) | 上游注意力與下游壅塞動態案例 |
| [Fig_OoD_1_LocalShock.png](paper/figure/Fig_OoD_1_LocalShock.png) | 局部事故擾動的 OoD 測試 |

## 環境與資料準備

建議使用 Python 3.10 以上，並依 CPU/CUDA 環境安裝合適的 PyTorch。專案目前沒有鎖定版號的 dependency manifest；最小依賴如下：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn torch
```

若根目錄沒有解壓後的 CSV：

```powershell
Expand-Archive .\all_segments_preprocessed.zip -DestinationPath . -Force
python .\dataset.py
```

資料集約 445 MB；完整訓練建議使用 GPU，執行前請預留記憶體與時間。

## 建議實驗流程

所有指令均由專案根目錄執行。

### 1. 全模型效能比較

```powershell
python .\run_all_model_metrics.py --seed 42
python .\plot_all_model_performance.py
```

輸出：`all_model_performance_for_plot.csv` 與 `paper/figure/Fig_Performance_Comparison_All_Models.png`。

### 2. PI-X-STGAT 參數搜尋

```powershell
python .\search_best_pix.py --epochs 15 --patience 5
```

輸出：`pix_best_search_results.csv`、`pix_best_checkpoint.pt`，並更新效能表中的 PI-X-STGAT 紀錄。

### 3. 多步預測

```powershell
python .\run_multistep_forecasting_all_models.py --seed 42 --skip-existing
```

輸出：`multistep_forecasting_all_models.csv` 與多步預測圖。

### 4. 超參數、消融與穩定性

```powershell
python .\hyperparameter_sensitivity.py
python .\hyperparameter_sensitivity_gru_embed_seq.py --skip-existing
python .\ablation_study.py
python .\run_seed_stability_comparison.py --skip-existing
```

`mae_score.py` 會用檔案內手動填入的四個 MAE 值製作論文版消融圖；正式更新圖表前，應先把數值替換為當次 `ablation_study.py` 的真實結果。

### 5. OoD 與真實事件

```powershell
python .\ood_simulation.py
python .\real_event_ood.py --scenario holiday --checkpoint .\pix_best_checkpoint.pt
```

`real_event_ood.py` 會輸出事件候選、各模型結果與真實事件比較圖。

### 6. XAI 案例

```powershell
python .\xai_dynamic_case_study.py
```

此案例輸出動態注意力圖。現有的全域注意力熱圖與 Top-edge 圖已收錄於圖片資料夾，但專案中尚未找到直接產生這兩張最終圖的獨立腳本。

## 實驗程式分類

| 類別 | 建議使用 | 說明 |
| --- | --- | --- |
| 核心元件 | `dataset.py`, `model.py`, `metrics.py` | 資料、模型與物理損失 |
| 主比較 | `run_all_model_metrics.py`, `plot_all_model_performance.py` | 15-minute 標籤下的全模型比較 |
| 多步預測 | `run_multistep_forecasting_all_models.py` | 取代較舊的 `main.py` 與 `multi_step_forecasting.py` |
| 最佳模型 | `search_best_pix.py` | 搜尋 PI-X-STGAT 並儲存 checkpoint |
| 穩定性 | `run_seed_stability_comparison.py` | 多隨機種子比較 |
| 超參數 | `hyperparameter_sensitivity*.py` | 物理權重與模型容量分析 |
| OoD | `ood_simulation.py`, `real_event_ood.py` | 合成擾動與真實假日切片 |
| XAI | `xai_dynamic_case_study.py` | 動態注意力案例 |
| 舊版/輔助 | `evaluate_and_plot.py`, `plot_new_performance.py`, `reviewer_response_experiments.py`, `plot_intervention.py`, `xai_plotter.py` | 保留作為過往分析，不建議當主結果入口 |

## 重現前需要核對的事項

目前程式、結果檔與論文敘述間仍有幾個需要在投稿或公開重現前統一的地方：

1. `dataset.py` 實作的是 Haversine 距離、每節點 `k=6` 的加權 KNN、有向列正規化與 self-loop；論文目前描述 2.0 km 距離閾值。兩者應擇一並同步。
2. 程式以排序後的「下一個唯一時間點」定義 `horizon=1`，繪圖則標成 15 分鐘；資料早期可見非固定間隔，應先確認取樣頻率或重採樣，再標示 15/30/45/60 分鐘。
3. `model.py` 目前沒有 ASTGCN 類別；比較表中的 ASTGCN 數值是既有 CSV 紀錄，不是 `run_all_model_metrics.py` 在本專案內重新訓練的結果。
4. `Fig_Ablation_Study.png` 的產生腳本 `mae_score.py` 使用手動填入數值，與 `ablation_study.py` 的自動訓練輸出檔名不同。
5. 單一城市與單一資料來源仍限制跨城市泛化；注意力解釋也應維持「候選影響關係」的表述，避免宣稱因果。

## 研究限制與下一步

目前驗證只涵蓋洛杉磯單一城市；未來可加入 METR-LA、PEMS-BAY、PeMS 等公開資料集進行跨城市驗證。XAI 部分可再搭配反事實解釋、擾動測試或因果發現方法，驗證注意力線索是否穩健。若要進一步支援交通控制，可將預測與候選影響路段作為深度強化學習的環境狀態，延伸至主動式號誌或匝道控制。
