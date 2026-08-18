# PI-X-STGAT

**結合物理引導與可解釋性的時空圖注意力交通預測模型**

[![DOI](https://img.shields.io/badge/DOI-10.32604%2Fcmes.2026.086216-0B7285)](https://doi.org/10.32604/cmes.2026.086216)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

[English](README.md) | 繁體中文

本 repository 是下列正式發表論文的官方實作與實驗封存：

> Yan-Wei Li and David Chunhu Li，〈A Physics-Informed Spatial-Temporal Graph Attention Model for Traffic Forecasting and Interpretable Congestion Propagation Analysis〉，*Computer Modeling in Engineering & Sciences*，2026。[https://doi.org/10.32604/cmes.2026.086216](https://doi.org/10.32604/cmes.2026.086216)

論文已於 2026 年 8 月 14 日由 Tech Science Press 正式線上發行。

![PI-X-STGAT 模型架構](<figures/Fig_Model Architecture Diagram.png>)

## 研究概述

PI-X-STGAT 針對資料驅動都市交通預測的三項實務限制：物理一致性不足、分布轉移下的穩定性有限，以及模型難以解釋。研究將道路區段建模為圖節點，整合交通、道路、天氣與週期時間特徵，預測未來的 **Jam Factor** 與**速度**。

模型包含四個主要元件：

1. **Context-Aware Graph Attention Network（GAT）**：聚合相連道路區段的空間資訊。
2. **Node Adaptive Parameter Learning（NAPL）**：學習地理拓樸未完整表達的潛在節點關係。
3. **Gated Recurrent Unit（GRU）**：建模交通狀態的時間演化。
4. **Physics-guided regularization**：以軟式交通流守恆代理殘差約束不合理的預測。

注意力矩陣用於分析候選壅塞影響路徑；它反映模型學到的統計相依性，不能直接解讀為物理因果證據。

## 研究特色

- 以 18 維多模態輸入共同預測 Jam Factor 與速度。
- 結合既定空間結構與可學習節點關係的情境感知自適應圖。
- 使用軟式物理引導損失抑制不合理預測，但不宣稱直接求解 LWR 偏微分方程。
- 透過空間與時間注意力分析候選壅塞傳播關係。
- 涵蓋標準預測、多步預測、消融、超參數、隨機種子穩定性與 OoD 評估。

## 資料與實驗設定

| 項目 | 設定 |
| --- | --- |
| 研究區域 | 洛杉磯都市道路網 |
| 資料來源 | HERE Maps API |
| 圖規模 | 87 個道路區段節點 |
| 時間資料 | 研究資料集包含 59,194 個時間點 |
| 輸入 | 18 維交通、道路、天氣與週期時間特徵 |
| 預測目標 | Jam Factor、速度 |
| 資料切分 | 依時間順序 80% 訓練、20% 評估 |
| 預設歷史窗 | `seq_len=12` |
| 最佳化器 | Adam，learning rate `1e-3` |
| Batch size | 32 |
| 論文報告物理權重 | `lambda_phy=0.015` |

HERE Maps 資料的存取與再散布受 HERE Technologies 條款約束，因此本 repository **不提供原始或前處理後資料集**。執行資料相關實驗前，請將合法取得並完成前處理的 `all_segments_preprocessed.csv` 放在根目錄；此檔案已由 `.gitignore` 排除。

預測目標與模型特徵定義於 [`constants.py`](constants.py)，張量轉換與圖前處理實作於 [`dataset.py`](dataset.py)。

## 已封存結果

下列數值來自 [`results/all_model_performance_for_plot.csv`](results/all_model_performance_for_plot.csv)。這些是既有實驗封存結果，本次整理並未重新訓練模型。

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

其他封存結果：

- PI-X-STGAT 五組隨機種子的 Jam MAE：`0.0669 ± 0.0063`。
- Labor Day 真實事件切片的 Jam MAE：`0.0597`。
- 真實 Jam Factor 接近 0 時，MAPE 會數值不穩定，因此僅適合作為輔助指標。

![模型效能比較](figures/Fig_Performance_Comparison_All_Models.png)

## Repository 結構

```text
PI-X-STGAT-Research/
├── README.md / README.zh-TW.md       # 英文與繁體中文說明
├── CITATION.cff                      # 機器可讀的引用資訊
├── LICENSE                           # Repository 程式的 MIT 授權
├── requirements.txt                  # 最小 Python 依賴
├── constants.py                      # 特徵與預測目標
├── dataset.py                        # 資料張量、圖建構與 DataLoader
├── model.py                          # PI-X-STGAT 與已實作基準模型
├── metrics.py                        # 物理引導正則化
├── experiment_paths.py               # 統一輸出資料夾
├── run_*.py                          # 主要實驗入口
├── figures/                          # 論文與實驗圖片
├── results/                          # 封存 CSV 結果
└── checkpoints/                      # 已訓練 PI-X-STGAT 權重
```

舊版與單次分析腳本仍留在根目錄，以保留完整實驗歷程。重新執行時，建議優先使用 `run_*.py`、`search_best_pix.py` 與具名的敏感度/OoD 腳本。

## 安裝

建議使用 Python 3.10 以上，並依 CPU/CUDA 環境選擇合適的 PyTorch。

```bash
git clone https://github.com/rexw713ss/PI-X-STGAT-Research.git
cd PI-X-STGAT-Research
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

將合法取得的資料集放到根目錄後，先驗證資料處理：

```bash
python dataset.py
```

完整模型訓練需要較多時間與記憶體，建議使用支援 CUDA 的 GPU。

## 重現實驗

所有指令均由 repository 根目錄執行。產生的圖片、結果與模型權重會分別寫入 `figures/`、`results/` 與 `checkpoints/`。

```bash
# 全模型比較
python run_all_model_metrics.py --seed 42
python plot_all_model_performance.py

# PI-X-STGAT 搜尋與權重
python search_best_pix.py --epochs 15 --patience 5

# 多步預測
python run_multistep_forecasting_all_models.py --seed 42 --skip-existing

# 敏感度、消融與隨機種子穩定性
python hyperparameter_sensitivity.py
python hyperparameter_sensitivity_gru_embed_seq.py --skip-existing
python ablation_study.py
python run_seed_stability_comparison.py --skip-existing

# OoD 與真實事件
python ood_simulation.py
python real_event_ood.py --scenario holiday

# 注意力案例
python xai_dynamic_case_study.py
```

`mae_score.py` 是使用手動輸入消融數值的論文繪圖輔助程式；重新產生正式圖前，應先用對應訓練結果更新檔案內數值。

## 圖片封存

| 圖片 | 說明 |
| --- | --- |
| [模型架構](<figures/Fig_Model Architecture Diagram.png>) | PI-X-STGAT 整體流程 |
| [全模型效能](figures/Fig_Performance_Comparison_All_Models.png) | 預測指標比較 |
| [多步預測](figures/Fig_Multistep_Forecasting.png) | 不同預測 horizon 的誤差 |
| [物理權重敏感度](figures/Fig_Hyperparameter_Sensitivity.png) | 物理正則化權重分析 |
| [容量與時間窗敏感度](figures/Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.png) | GRU、embedding 與 sequence length |
| [消融研究](figures/Fig_Ablation_Study.png) | 模組貢獻比較 |
| [誤差分析](figures/Fig_Error_Analysis.png) | 守恆殘差與預測誤差 |
| [注意力熱圖](figures/Fig_Attention_Heatmap.png) | 全域空間注意力 |
| [Top attention edges](figures/Fig_Top_Attention_Edges.png) | 高權重道路區段配對 |
| [動態注意力](figures/Fig_Real_Dynamic_Attention.png) | 時間注意力案例 |
| [局部衝擊 OoD](figures/Fig_OoD_1_LocalShock.png) | 合成局部擾動 |

## 重現注意事項

在宣稱完全重現前，請先核對下列實作細節：

1. 發行版 `dataset.py` 使用 Haversine 距離建立 `k=6` 加權 KNN 圖，並進行列正規化及加入 self-loop；論文描述的是 2.0 km 候選鄰居閾值。請依欲重現的實驗協定統一圖建構方式。
2. 程式以排序後的唯一時間點定義 horizon，繪圖則標示為 15/30/45/60 分鐘；解讀標籤前應先確認或重採樣實際時間頻率。
3. ASTGCN 數值保留自論文比較表，但 `model.py` 未包含 ASTGCN 實作。
4. 注意力權重是相關性模型訊號，不是交通流因果證據。
5. 研究只評估單一城市與資料來源，跨城市泛化仍是後續工作。

## 引用

GitHub 會讀取 [`CITATION.cff`](CITATION.cff) 並顯示 **Cite this repository**。建議引用正式論文：

```bibtex
@article{Li2026PIXSTGAT,
  author  = {Li, Yan-Wei and Li, David Chunhu},
  title   = {A Physics-Informed Spatial-Temporal Graph Attention Model for Traffic Forecasting and Interpretable Congestion Propagation Analysis},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026},
  doi     = {10.32604/cmes.2026.086216},
  url     = {https://doi.org/10.32604/cmes.2026.086216}
}
```

## 授權與資料權利

Repository 程式採用 [MIT License](LICENSE)。已訓練權重與封存結果用於研究重現；此授權不包含 HERE Maps 原始資料或出版社論文，相關材料仍受各自條款約束。
