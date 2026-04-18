# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# ========================= 字体设置 =========================
import matplotlib.font_manager as fm

def pick_chinese_font():
    cand_fonts = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Micro Hei', 'PingFang SC', 'Heiti SC',
        'Songti SC', 'STHeiti', 'Arial Unicode MS'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for f in cand_fonts:
        if f in available:
            return f
    return None

zh_font = pick_chinese_font()
plt.style.use('seaborn-v0_8-whitegrid')

if zh_font is not None:
    matplotlib.rcParams['font.sans-serif'] = [zh_font, 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = [zh_font, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print('[WARN] 未找到可用中文字体，中文仍可能显示为方框，请安装 Microsoft YaHei / SimHei / Noto Sans CJK SC')

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ========================= 高级库 =========================
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[INFO] lightgbm 未安装，降级使用 GradientBoostingClassifier")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[INFO] shap 未安装，降级使用 permutation_importance")

try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except ImportError:
    HAS_BORUTA = False
    print("[INFO] boruta 未安装，降级使用 RFE")

# ========================= 路径配置 =========================
DATA_PATH = './data.xlsx'
FIG_DIR = './task1/figures'
TABLE_DIR = './task1/tables'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

PALETTE_A = ['#1F4E79', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6',
             '#F79646', '#4F81BD', '#943634', '#76923C']

PALETTE_B = {
    'positive': '#C0504D',   # 正向/高风险/高值
    'negative': '#1F4E79',   # 负向/低风险/低值
    'neutral':  '#7F7F7F',   # 中性
    'highlight':'#F79646',   # 高亮
    'softred':  '#E6B8B7',
    'softblue': '#B8CCE4',
    'softgreen':'#D8E4BC',
    'gold':     '#C9A227'
}

CONSTITUTIONS = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
                 '湿热质', '血瘀质', '气郁质', '特禀质']

# ========================= 工具函数 =========================
def find_col(df, keyword):
    for c in df.columns:
        if keyword in c:
            return c
    return None

def short(name):
    mapping = {
        'HDL-C（高密度脂蛋白）': 'HDL-C',
        'LDL-C（低密度脂蛋白）': 'LDL-C',
        'TG（甘油三酯）': 'TG',
        'TC（总胆固醇）': 'TC',
        '活动量表总分（ADL总分+IADL总分）': '活动量表',
        '空腹血糖': '血糖',
        '血尿酸': '血尿酸',
        'BMI': 'BMI'
    }
    return mapping.get(name, name[:8])

def minmax(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return v
    rng = np.nanmax(v) - np.nanmin(v)
    return (v - np.nanmin(v)) / rng if rng > 1e-12 else np.zeros_like(v)

def safe_spearman(x, y):
    r, _ = stats.spearmanr(x, y)
    if np.isnan(r):
        return 0.0
    return abs(r)

def distance_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = np.mean(A * B)
    dcov2_xx = np.mean(A * A)
    dcov2_yy = np.mean(B * B)
    if dcov2_xx <= 1e-12 or dcov2_yy <= 1e-12:
        return 0.0
    val = np.sqrt(max(dcov2_xy, 0) / np.sqrt(dcov2_xx * dcov2_yy))
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)

def borda_rank(score_list, names):
    n = len(names)
    borda = np.zeros(n, dtype=float)
    for scores in score_list:
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)
        for rank, idx in enumerate(order):
            borda[idx] += (n - rank)
    return borda / len(score_list)

def signal_strength(*arrays):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        vals.append(np.nanmean(arr))
    return float(np.mean(vals))

def robust_fill_median(df_sub):
    out = df_sub.copy()
    for col in out.columns:
        med = out[col].median()
        out[col] = out[col].fillna(med)
    return out

# ========================= 读入数据 =========================
df = pd.read_excel(DATA_PATH)
print(f"数据形状: {df.shape}")

blood_keywords = ['HDL', 'LDL', 'TG', 'TC', '空腹血糖', '血尿酸', 'BMI']
blood_cols = [find_col(df, k) for k in blood_keywords]
blood_cols = [c for c in blood_cols if c is not None]

act_col = find_col(df, '活动量表总分')
target_col = find_col(df, '高血脂')
tanshi_col = find_col(df, '痰湿质')
type_col = find_col(df, '体质标签')

candidates = blood_cols + ([act_col] if act_col else [])

print(f"候选指标 ({len(candidates)} 项): {candidates}")
print(f"高血脂标签列: {target_col}")
print(f"痰湿积分列: {tanshi_col}")
print(f"体质标签列: {type_col}")

if target_col is None or tanshi_col is None or len(candidates) == 0:
    raise ValueError("关键列未识别成功，请检查 Excel 列名是否包含：高血脂、痰湿质、TG/TC/LDL/HDL/空腹血糖/血尿酸/BMI/活动量表总分")

# ========================= 基础数据 =========================
X_raw = robust_fill_median(df[candidates])
X_std = StandardScaler().fit_transform(X_raw)

y = df[target_col].astype(int).values
P_ts = df[tanshi_col].astype(float).values

# 痰湿严重度增强目标：
# 若有主导体质标签，则融合“痰湿积分”和“是否痰湿主导体质”
if type_col is not None:
    dominant_tanshi = (df[type_col].astype(float).values == 5).astype(float)
    Y_ts_enhanced = 0.7 * minmax(P_ts) + 0.3 * dominant_tanshi
    print("痰湿严重度目标: 0.7*标准化痰湿积分 + 0.3*是否痰湿主导体质")
else:
    Y_ts_enhanced = minmax(P_ts)
    print("痰湿严重度目标: 标准化痰湿积分")

# ========================= 任务 A：痰湿严重度 =========================
print("\n" + "=" * 60)
print("任务 A: 表征痰湿严重度的关联指标")
print("=" * 60)

spearman_tsA = np.array([
    safe_spearman(X_std[:, i], Y_ts_enhanced)
    for i in range(X_std.shape[1])
])

np.random.seed(42)
sample_idx = np.random.choice(len(X_std), min(600, len(X_std)), replace=False)
dcorr_tsA = np.array([
    distance_corr(X_std[sample_idx, i], Y_ts_enhanced[sample_idx])
    for i in range(X_std.shape[1])
])

# 连续互信息：修正原来“分箱后用 mutual_info_classif”的问题
mi_tsA = np.array([
    mutual_info_regression(
        X_std[:, i:i+1], Y_ts_enhanced, random_state=42
    )[0]
    for i in range(X_std.shape[1])
])

borda_A = borda_rank([spearman_tsA, dcorr_tsA, mi_tsA], candidates)

sev_df = pd.DataFrame({
    '指标': candidates,
    'Spearman': np.round(spearman_tsA, 4),
    '距离相关': np.round(dcorr_tsA, 4),
    '互信息(连续型)': np.round(mi_tsA, 4),
    'Borda得分': np.round(borda_A, 2)
}).sort_values('Borda得分', ascending=False).reset_index(drop=True)

sev_df.to_csv(f'{TABLE_DIR}/p1_A_severity.csv', index=False, encoding='utf-8-sig')
print(sev_df.to_string(index=False))

# ========================= 任务 B：高血脂风险 =========================
print("\n" + "=" * 60)
print("任务 B: 预警高血脂发病风险的关联指标")
print("=" * 60)

wass_B = np.array([
    wasserstein_distance(X_std[y == 0, i], X_std[y == 1, i])
    for i in range(X_std.shape[1])
])

ks_B = np.array([
    ks_2samp(X_std[y == 0, i], X_std[y == 1, i]).statistic
    for i in range(X_std.shape[1])
])

if HAS_LGBM:
    core_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model_name = "LightGBM"
else:
    core_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model_name = "GradientBoosting"

core_model.fit(X_std, y)
print(f"核心模型: {model_name}")

if HAS_SHAP:
    explainer = shap.TreeExplainer(core_model)
    shap_vals = explainer.shap_values(X_std)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_imp = np.abs(shap_vals).mean(axis=0)
    imp_name = "SHAP均值"
else:
    perm = permutation_importance(core_model, X_std, y, n_repeats=10, random_state=42, n_jobs=-1)
    shap_imp = perm.importances_mean
    shap_vals = None
    imp_name = "Permutation重要性"

if HAS_BORUTA:
    boruta = BorutaPy(
        RandomForestClassifier(n_estimators=200, random_state=42),
        n_estimators='auto',
        random_state=42,
        max_iter=50
    )
    boruta.fit(X_std, y)
    confirmed = boruta.support_
    selection_name = "Boruta"
else:
    rfe = RFE(
        LogisticRegression(max_iter=1000, random_state=42),
        n_features_to_select=max(4, X_std.shape[1] // 2)
    )
    rfe.fit(X_std, y)
    confirmed = rfe.support_
    selection_name = "RFE"

print(f"特征确认方法: {selection_name}")
print(f"确认选择特征数: {confirmed.sum()} / {len(candidates)}")

borda_B = borda_rank([wass_B, ks_B, shap_imp], candidates)

risk_df = pd.DataFrame({
    '指标': candidates,
    'Wasserstein': np.round(wass_B, 4),
    'KS统计量': np.round(ks_B, 4),
    imp_name: np.round(shap_imp, 4),
    f'{selection_name}确认': confirmed,
    'Borda得分': np.round(borda_B, 2)
}).sort_values('Borda得分', ascending=False).reset_index(drop=True)

risk_df.to_csv(f'{TABLE_DIR}/p1_B_risk.csv', index=False, encoding='utf-8-sig')
print(risk_df.to_string(index=False))

# ========================= 双目标融合 =========================
print("\n" + "=" * 60)
print("双目标融合排名（自适应加权）")
print("=" * 60)

strength_A = signal_strength(spearman_tsA, dcorr_tsA, mi_tsA)
strength_B = signal_strength(wass_B, ks_B, shap_imp)

# 如果痰湿侧信号较弱，则自动降低权重
wA = strength_A / (strength_A + strength_B + 1e-12)
wA = float(np.clip(wA, 0.25, 0.50))
wB = 1.0 - wA

borda_final = wA * borda_A + wB * borda_B

combined = pd.DataFrame({
    '指标': candidates,
    'A_Borda_痰湿严重度': np.round(borda_A, 2),
    'B_Borda_发病风险': np.round(borda_B, 2),
    '融合Borda': np.round(borda_final, 2),
    '是否Boruta/RFE确认': confirmed
}).sort_values('融合Borda', ascending=False).reset_index(drop=True)

combined.to_csv(f'{TABLE_DIR}/p1_final_ranking.csv', index=False, encoding='utf-8-sig')
print(f"自适应权重: 任务A={wA:.3f}, 任务B={wB:.3f}")
print(combined.to_string(index=False))

# ========================= 九种体质贡献 =========================
print("\n" + "=" * 60)
print("九种体质对发病风险的贡献度")
print("=" * 60)

available_constitutions = [c for c in CONSTITUTIONS if c in df.columns]
if len(available_constitutions) != 9:
    raise ValueError(f"九种体质积分列未完整识别，当前识别到: {available_constitutions}")

Xc = df[available_constitutions].fillna(0).values
Xc_std = StandardScaler().fit_transform(Xc)

if HAS_LGBM:
    cons_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
else:
    cons_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

cons_model.fit(Xc, y)

if HAS_SHAP:
    cons_expl = shap.TreeExplainer(cons_model)
    cons_shap = cons_expl.shap_values(Xc)
    if isinstance(cons_shap, list):
        cons_shap = cons_shap[1]
    cons_shap_imp = np.abs(cons_shap).mean(axis=0)
else:
    perm_c = permutation_importance(cons_model, Xc, y, n_repeats=10, random_state=42, n_jobs=-1)
    cons_shap_imp = perm_c.importances_mean
    cons_shap = None

wass_c = np.array([
    wasserstein_distance(Xc_std[y == 0, i], Xc_std[y == 1, i])
    for i in range(9)
])

lrc = LogisticRegression(max_iter=1000, random_state=42).fit(Xc_std, y)
lr_coef_c = np.abs(lrc.coef_[0])

# 实际发病率差值辅助校验：该体质积分高组 vs 低组
risk_gap_c = []
for i in range(9):
    vals = Xc[:, i]
    q75 = np.quantile(vals, 0.75)
    q25 = np.quantile(vals, 0.25)
    high_mask = vals >= q75
    low_mask = vals <= q25
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        gap = abs(y[high_mask].mean() - y[low_mask].mean())
    else:
        gap = 0.0
    risk_gap_c.append(gap)
risk_gap_c = np.array(risk_gap_c)

borda_cons = borda_rank([cons_shap_imp, wass_c, lr_coef_c, risk_gap_c], available_constitutions)

cons_df = pd.DataFrame({
    '体质': available_constitutions,
    f'{imp_name}': np.round(cons_shap_imp, 4),
    'Wasserstein': np.round(wass_c, 4),
    'LR|系数|': np.round(lr_coef_c, 4),
    '高低分组发病率差': np.round(risk_gap_c, 4),
    'Borda综合': np.round(borda_cons, 2)
}).sort_values('Borda综合', ascending=False).reset_index(drop=True)

cons_df.to_csv(f'{TABLE_DIR}/p1_constitutions.csv', index=False, encoding='utf-8-sig')
print(cons_df.to_string(index=False))

# ========================= 可视化 1：痰湿严重度 =========================
fig = plt.figure(figsize=(16, 11), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

# ---------- (a) 气泡横向相关图 ----------
ax = fig.add_subplot(gs[0, 0])
order_A = sev_df['指标'].tolist()
y_pos = np.arange(len(order_A))
r_signed = []
r_abs = []
for f in order_A:
    i = candidates.index(f)
    r = stats.spearmanr(X_std[:, i], Y_ts_enhanced)[0]
    r = 0 if np.isnan(r) else r
    r_signed.append(r)
    r_abs.append(abs(r))

for i, (name, r, ra) in enumerate(zip(order_A, r_signed, r_abs)):
    color = PALETTE_B['positive'] if r >= 0 else PALETTE_B['negative']
    ax.scatter(ra, i, s=3500 * ra + 90, color=color, alpha=0.75,
               edgecolor='black', linewidth=0.9, zorder=3)
    ax.plot([0, ra], [i, i], color=color, lw=1.2, alpha=0.55, zorder=2)
    ax.text(ra + 0.004, i, f'{r:+.3f}', va='center', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels([short(x) for x in order_A], fontsize=10)
ax.set_xlabel('|Spearman秩相关系数|', fontsize=10)
ax.set_title('(a) 候选指标与痰湿严重度的相关强度', fontsize=12, fontweight='bold')
ax.set_xlim(0, max(0.08, max(r_abs) * 1.35))
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- (b) Top3 趋势曲线 ----------
ax = fig.add_subplot(gs[0, 1])
top3_A = sev_df.head(3)['指标'].tolist()
curve_colors = [PALETTE_A[0], PALETTE_A[1], PALETTE_A[2]]

for i, f in enumerate(top3_A):
    vals = df[f].astype(float).values
    df_tmp = pd.DataFrame({'v': vals, 'p': Y_ts_enhanced})
    try:
        df_tmp['bin'] = pd.qcut(df_tmp['p'], 10, duplicates='drop')
    except Exception:
        df_tmp['bin'] = pd.cut(df_tmp['p'], 10)
    grp = df_tmp.groupby('bin', observed=True).agg(
        m=('v', 'mean'),
        s=('v', 'std'),
        n=('v', 'size')
    ).reset_index()
    grp['s'] = grp['s'].fillna(0)
    mean_norm = minmax(grp['m'].values)
    std_norm = grp['s'].values / (np.nanmax(grp['m'].values) - np.nanmin(grp['m'].values) + 1e-9)
    x_mid = np.arange(len(grp))
    ax.plot(x_mid, mean_norm, '-o', color=curve_colors[i], lw=2.2,
            markersize=6.5, label=short(f), alpha=0.95)
    ax.fill_between(x_mid,
                    np.clip(mean_norm - 0.25 * std_norm, 0, 1.05),
                    np.clip(mean_norm + 0.25 * std_norm, 0, 1.05),
                    color=curve_colors[i], alpha=0.12)

ax.set_xlabel('痰湿严重度分位组', fontsize=10)
ax.set_ylabel('指标均值（归一化）', fontsize=10)
ax.set_title('(b) Top3 指标随痰湿严重程度变化趋势', fontsize=12, fontweight='bold')
ax.set_xticks(range(10))
ax.set_xticklabels([f'G{i+1}' for i in range(10)], fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(frameon=True, fontsize=9, loc='best')
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- (c) 综合排序条形图 ----------
ax = fig.add_subplot(gs[1, 0])
sev_sorted = sev_df.copy()
y_pos = np.arange(len(sev_sorted))
bar_colors = [PALETTE_A[i % len(PALETTE_A)] for i in range(len(sev_sorted))]
bars = ax.barh(y_pos, sev_sorted['Borda得分'], color=bar_colors,
               edgecolor='black', linewidth=0.8, alpha=0.92)

for rank, (bar, v) in enumerate(zip(bars, sev_sorted['Borda得分'])):
    ax.text(v + 0.08, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}', va='center', fontsize=9)
    if rank < 3:
        ax.text(0.03, bar.get_y() + bar.get_height()/2,
                f'Top{rank+1}', va='center', ha='left', fontsize=8,
                color='white', fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels([short(x) for x in sev_sorted['指标']], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Borda综合得分', fontsize=10)
ax.set_title('(c) 痰湿严重度表征能力综合排序', fontsize=12, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- (d) Top4 多准则雷达图 ----------
ax = fig.add_subplot(gs[1, 1], projection='polar')
top4_A = sev_df.head(4)['指标'].tolist()
categories = ['Spearman', '距离相关', '互信息']
theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
theta += theta[:1]

for idx, f in enumerate(top4_A):
    i = candidates.index(f)
    values = [
        minmax(spearman_tsA)[i],
        minmax(dcorr_tsA)[i],
        minmax(mi_tsA)[i]
    ]
    values += values[:1]
    ax.plot(theta, values, 'o-', linewidth=2.0, label=short(f),
            color=PALETTE_A[idx], markersize=5)
    ax.fill(theta, values, alpha=0.10, color=PALETTE_A[idx])

ax.set_xticks(theta[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_ylim(0, 1.0)
ax.set_title('(d) Top4 指标的多准则对比', fontsize=12, fontweight='bold', pad=18)
ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.12), fontsize=8, frameon=True)

plt.suptitle('问题一（A） 痰湿严重度关联指标分析', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{FIG_DIR}/p1_01_severity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\n✓ 保存图: p1_01_severity.png')

# ========================= 可视化 2：风险 =========================
fig = plt.figure(figsize=(16, 11), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

# 统一排序，后续子图保持一致
risk_sorted = risk_df.copy().reset_index(drop=True)

# ---------- (a) Wasserstein-KS 双指标散点 ----------
ax = fig.add_subplot(gs[0, 0])

for i, name in enumerate(candidates):
    conf = confirmed[i]
    color = PALETTE_B['highlight'] if conf else PALETTE_B['softblue']
    size = 340 if conf else 180

    ax.scatter(
        wass_B[i], ks_B[i],
        s=size, color=color, edgecolor='black',
        linewidth=1.0, alpha=0.88, zorder=3
    )

    ax.annotate(
        short(name), (wass_B[i], ks_B[i]),
        xytext=(6, 5), textcoords='offset points',
        fontsize=9
    )

# 均值参考线
ax.axvline(np.mean(wass_B), color='#999999', linestyle='--', linewidth=1.0, alpha=0.8)
ax.axhline(np.mean(ks_B), color='#999999', linestyle='--', linewidth=1.0, alpha=0.8)

ax.set_xlabel('Wasserstein 距离', fontsize=10)
ax.set_ylabel('KS 统计量', fontsize=10)
ax.set_title('(a) 发病组与未发病组的分布差异识别', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_items = [
    Patch(facecolor=PALETTE_B['highlight'], edgecolor='black', label=f'{selection_name}确认'),
    Patch(facecolor=PALETTE_B['softblue'], edgecolor='black', label='未确认')
]
ax.legend(handles=legend_items, fontsize=9, frameon=True, loc='best')

# ---------- (b) Top2 风险指标分组小提琴 ----------
ax = fig.add_subplot(gs[0, 1])

top2_B = risk_sorted.head(2)['指标'].tolist()
data_list = []
label_list = []
group_colors = []

for f in top2_B:
    vals_0 = df[df[target_col] == 0][f].dropna().astype(float).values
    vals_1 = df[df[target_col] == 1][f].dropna().astype(float).values
    allv = np.concatenate([vals_0, vals_1])

    mn, mx = allv.min(), allv.max()
    vals_0 = (vals_0 - mn) / (mx - mn + 1e-9)
    vals_1 = (vals_1 - mn) / (mx - mn + 1e-9)

    data_list.extend([vals_0, vals_1])
    label_list.extend([f'{short(f)}\n未发病', f'{short(f)}\n发病'])
    group_colors.extend([PALETTE_B['negative'], PALETTE_B['positive']])

vp = ax.violinplot(
    data_list,
    showmeans=True,
    showmedians=True,
    showextrema=True
)

for i, body in enumerate(vp['bodies']):
    body.set_facecolor(group_colors[i])
    body.set_edgecolor('black')
    body.set_alpha(0.62)

for k in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
    vp[k].set_color('#333333')
    vp[k].set_linewidth(1.0)

# 叠加箱线摘要，更适合论文展示
for i, arr in enumerate(data_list, start=1):
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    ax.plot([i - 0.12, i + 0.12], [med, med], color='black', linewidth=1.3, zorder=4)
    ax.plot([i, i], [q1, q3], color='black', linewidth=1.0, zorder=4)

ax.set_xticks(range(1, len(label_list) + 1))
ax.set_xticklabels(label_list, fontsize=9)
ax.set_ylabel('指标值（归一化）', fontsize=10)
ax.set_title('(b) Top2 风险指标在两类人群中的分布对比', fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- (c) 重要性排序 + 特征确认 ----------
ax = fig.add_subplot(gs[1, 0])

y_pos = np.arange(len(risk_sorted))
colors_b = [
    PALETTE_B['highlight'] if c else PALETTE_B['neutral']
    for c in risk_sorted[f'{selection_name}确认']
]

bars = ax.barh(
    y_pos, risk_sorted[imp_name],
    color=colors_b, edgecolor='black',
    linewidth=0.8, alpha=0.92
)

for rank, (bar, v, c) in enumerate(zip(
    bars, risk_sorted[imp_name], risk_sorted[f'{selection_name}确认']
)):
    tag = '（已确认）' if c else ''
    ax.text(
        v + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f'{v:.3f}{tag}',
        va='center', fontsize=9
    )
    if rank < 3:
        ax.text(
            0.002,
            bar.get_y() + bar.get_height() / 2,
            f'Top{rank+1}',
            va='center', ha='left',
            fontsize=8, color='white', fontweight='bold'
        )

ax.set_yticks(y_pos)
ax.set_yticklabels([short(x) for x in risk_sorted['指标']], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel(imp_name, fontsize=10)
ax.set_title(f'(c) 模型重要性排序及{selection_name}筛选结果', fontsize=12, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- (d) SHAP蜂群 / TG分组发病率 ----------
ax = fig.add_subplot(gs[1, 1])

if HAS_SHAP and shap_vals is not None:
    top_n = min(6, len(candidates))
    top_idx = np.argsort(-shap_imp)[:top_n]
    cmap_list = [PALETTE_A[i % len(PALETTE_A)] for i in range(top_n)]

    np.random.seed(42)
    for row_i, feat_idx in enumerate(top_idx):
        sv = shap_vals[:, feat_idx]
        jitter = np.random.uniform(-0.16, 0.16, size=len(sv))
        ax.scatter(
            sv, np.full(len(sv), row_i) + jitter,
            s=10, alpha=0.45, color=cmap_list[row_i],
            edgecolors='none'
        )
        ax.hlines(
            row_i, np.min(sv), np.max(sv),
            color='#DDDDDD', lw=0.8, zorder=0
        )

    ax.axvline(0, color='gray', lw=1.0, ls='--')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([short(candidates[i]) for i in top_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP 值', fontsize=10)
    ax.set_title('(d) 关键风险指标的 SHAP 影响分布', fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

else:
    tg_col = find_col(df, 'TG')
    if tg_col:
        tg_bins = pd.qcut(
            df[tg_col], 5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            duplicates='drop'
        )
        rate = df.groupby(tg_bins, observed=True)[target_col].mean()
        cnts = df.groupby(tg_bins, observed=True)[target_col].size()

        bar_colors = [PALETTE_A[i % len(PALETTE_A)] for i in range(len(rate))]
        bars = ax.bar(
            range(len(rate)), rate.values,
            color=bar_colors, edgecolor='black',
            linewidth=0.8, alpha=0.9
        )

        for b, r, c in zip(bars, rate.values, cnts.values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                r + 0.015,
                f'{r*100:.1f}%\n(n={c})',
                ha='center', fontsize=9
            )

        ax.plot(
            range(len(rate)), rate.values,
            color='#444444', lw=1.5, marker='o'
        )

        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels(rate.index.astype(str), fontsize=9)
        ax.set_xlabel('TG 分位组', fontsize=10)
        ax.set_ylabel('高血脂发病率', fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_title('(d) TG 分组与高血脂发病率关系', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.suptitle('问题一（B）高血脂发病风险关联指标分析', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{FIG_DIR}/p1_02_risk.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('✓ 保存图: p1_02_risk.png')

# ========================= 可视化 3：九种体质 =========================
fig = plt.figure(figsize=(16, 11), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

# ---------- (a) 九种体质贡献排序 ----------
ax1 = fig.add_subplot(gs[0, 0])
cons_sorted = cons_df.copy()
y_pos = np.arange(len(cons_sorted))
bar_colors = [PALETTE_A[i % len(PALETTE_A)] for i in range(len(cons_sorted))]
bars = ax1.barh(y_pos, cons_sorted['Borda综合'],
                color=bar_colors, edgecolor='black',
                linewidth=0.8, alpha=0.93)

for rank, (bar, v) in enumerate(zip(bars, cons_sorted['Borda综合'])):
    ax1.text(v + 0.05, bar.get_y() + bar.get_height()/2,
             f'{v:.1f}', va='center', fontsize=9)
    if rank < 3:
        ax1.text(0.03, bar.get_y() + bar.get_height()/2, f'Top{rank+1}',
                 va='center', ha='left', fontsize=8, color='white', fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(cons_sorted['体质'], fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel('Borda综合得分', fontsize=10)
ax1.set_title('(a) 九种体质对发病风险的贡献排序', fontsize=12, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.35)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ---------- (b) Top4 雷达图 ----------
ax2 = fig.add_subplot(gs[0, 1], projection='polar')
top4_cons = cons_sorted.head(4)['体质'].tolist()
categories = [imp_name, 'Wasserstein', 'LR|系数|', '高低分组发病率差']
theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
theta += theta[:1]

for idx, t in enumerate(top4_cons):
    orig_idx = available_constitutions.index(t)
    values = [
        minmax(cons_shap_imp)[orig_idx],
        minmax(wass_c)[orig_idx],
        minmax(lr_coef_c)[orig_idx],
        minmax(risk_gap_c)[orig_idx]
    ]
    values += values[:1]
    ax2.plot(theta, values, 'o-', linewidth=2.0, label=t,
             color=PALETTE_A[idx], markersize=5)
    ax2.fill(theta, values, alpha=0.10, color=PALETTE_A[idx])

ax2.set_xticks(theta[:-1])
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax2.set_ylim(0, 1.0)
ax2.set_title('(b) Top4体质的多准则综合表现', fontsize=12, fontweight='bold', pad=18)
ax2.legend(loc='upper right', bbox_to_anchor=(1.34, 1.12), fontsize=8, frameon=True)

# ---------- (c) 体质×准则热力图 ----------
ax3 = fig.add_subplot(gs[1, 0])
heat_mat = np.stack([
    minmax(cons_shap_imp),
    minmax(wass_c),
    minmax(lr_coef_c),
    minmax(risk_gap_c)
], axis=0)

order = [available_constitutions.index(t) for t in cons_sorted['体质']]
heat_mat = heat_mat[:, order]

im = ax3.imshow(heat_mat, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
ax3.set_yticks(range(4))
ax3.set_yticklabels([imp_name, 'Wasserstein', 'LR|系数|', '发病率差'], fontsize=9)
ax3.set_xticks(range(len(cons_sorted)))
ax3.set_xticklabels(cons_sorted['体质'], rotation=35, ha='right', fontsize=9)

for i in range(heat_mat.shape[0]):
    for j in range(heat_mat.shape[1]):
        v = heat_mat[i, j]
        txt_color = 'white' if v > 0.55 else 'black'
        ax3.text(j, i, f'{v:.2f}', ha='center', va='center',
                 color=txt_color, fontsize=8)

cbar = plt.colorbar(im, ax=ax3, fraction=0.045, pad=0.03)
cbar.set_label('归一化得分', fontsize=9)
ax3.set_title('(c) 九种体质在各评价准则下的相对表现', fontsize=12, fontweight='bold')

# ---------- (d) 各主导体质的实际发病率 ----------
ax4 = fig.add_subplot(gs[1, 1])
if type_col:
    rates_by_type = []
    counts_by_type = []
    for k in range(1, 10):
        sub = df[df[type_col] == k]
        rates_by_type.append(sub[target_col].mean() if len(sub) else 0)
        counts_by_type.append(len(sub))

    x_pos = np.arange(9)
    bar_colors = [PALETTE_A[i % len(PALETTE_A)] for i in range(9)]
    bars = ax4.bar(x_pos, rates_by_type, color=bar_colors,
                   edgecolor='black', linewidth=0.8, alpha=0.92)

    for bar, r, c in zip(bars, rates_by_type, counts_by_type):
        ax4.text(bar.get_x() + bar.get_width()/2, r + 0.015,
                 f'{r*100:.1f}%\n(n={c})', ha='center', fontsize=8)

    overall_rate = df[target_col].mean()
    ax4.axhline(overall_rate, ls='--', color=PALETTE_B['positive'], lw=1.5,
                label=f'总体均值：{overall_rate*100:.1f}%')
    ax4.plot(x_pos, rates_by_type, color='#444444', marker='o', lw=1.4)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(CONSTITUTIONS, rotation=35, fontsize=9, ha='right')
    ax4.set_ylabel('高血脂发病率', fontsize=10)
    ax4.set_title('(d) 不同主导体质对应的实际发病率', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(1.05, max(rates_by_type) + 0.12))
    ax4.legend(fontsize=9, frameon=True)
    ax4.grid(axis='y', linestyle='--', alpha=0.35)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

plt.suptitle('问题一（C） 九种体质对高血脂发病风险贡献分析', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{FIG_DIR}/p1_03_constitutions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ 保存图: p1_03_constitutions.png')

# ========================= 输出摘要 =========================
print('\n' + '=' * 60)
print('========= 问题一（修正版）完成 =========')
print(f'  核心模型: {model_name}')
print(f'  重要性方法: {imp_name}')
print(f'  特征确认: {selection_name}')
print(f'  高级库可用性: LGBM={HAS_LGBM}, SHAP={HAS_SHAP}, Boruta={HAS_BORUTA}')
print(f'  双目标自适应权重: A={wA:.3f}, B={wB:.3f}')
print('\n[最终建议关注的关键指标 Top5]')
print(combined.head(5).to_string(index=False))