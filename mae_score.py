import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 填入你跑出來的 4 個變體的真實 MAE 數據
models = ['Base-GRU\n(No Spatial/Phy/Ctx)', 'STGAT\n(+ Spatial GAT)', 'PI-STGAT\n(+ Physics Loss)', 'PI-X-STGAT\n(+ Context/Weather)']
mae_scores = [0.1019, 0.0896, 0.0785, 0.0713] # 請替換成你實際跑出的數值

df = pd.DataFrame({'Model Variants': models, 'Jam MAE': mae_scores})

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# 使用漸層顏色，象徵模型逐漸變強
palette = sns.color_palette("Blues", len(models))
ax = sns.barplot(x='Model Variants', y='Jam MAE', data=df, palette=palette, edgecolor='black')

plt.title('Ablation Study: Progressive Error Reduction', fontsize=18, fontweight='bold', pad=15)
plt.ylabel('Jam Factor MAE', fontsize=14)
plt.xlabel('')

# 標註誤差下降的百分比
for i in range(1, len(models)):
    drop_pct = (mae_scores[i-1] - mae_scores[i]) / mae_scores[i-1] * 100
    ax.annotate(f'-{drop_pct:.1f}%', 
                xy=(i, mae_scores[i]), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig("Fig_Ablation_Study.png", dpi=300)
print("✅ 消融實驗圖表已儲存！")