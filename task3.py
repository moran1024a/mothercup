# -*- coding: utf-8 -*-
"""
MathorCup 2026 C题 - 问题3
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

DATA_PATH = './data.xlsx'
FIG_DIR   = './task3/figures'
TABLE_DIR = './task3/tables'
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# =================== 全局字体与绘图风格 ===================
import matplotlib as mpl
from matplotlib import font_manager as fm

def pick_chinese_font():
    cand = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
        'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    for f in cand:
        try:
            path = fm.findfont(f, fallback_to_default=False)
            if path and ('DejaVuSans' not in path):
                return f
        except Exception:
            continue
    return 'SimHei'   # Windows 下优先兜底

CN_FONT = pick_chinese_font()

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [CN_FONT, 'Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.default'] = 'regular'

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.95
mpl.rcParams['legend.edgecolor'] = '#666666'

plt.style.use('seaborn-v0_8-white')

COLORS_CLUST = ['#4C72B0', '#C44E52', '#55A868', '#8172B3', '#64B5CD']
colors_s = ['#4C72B0', '#DD8452', '#55A868']

# =================== 1 ===================
df = pd.read_excel(DATA_PATH)

def find_col(df, kw):
    for c in df.columns:
        if kw in c:
            return c
    return None

COL_ID     = find_col(df, '样本ID') or df.columns[0]
COL_TYPE   = find_col(df, '体质标签')
COL_TANSHI = find_col(df, '痰湿质')
COL_AGEGRP = find_col(df, '年龄组')
COL_ACT    = find_col(df, '活动量表总分')

ts = df[df[COL_TYPE] == 5].copy().reset_index(drop=True)
print(f'痰湿体质患者数: {len(ts)}')

# =================== 2. 约束函数 ===================
GRADE_COST   = {1: 30, 2: 80, 3: 130}
SESSION_COST = {1: 3, 2: 5, 3: 8}
BUDGET       = 2000
F_MIN, F_MAX = 1, 10

def grade_from_tanshi(P0):
    if P0 <= 58: return 1
    elif P0 <= 61: return 2
    else: return 3

def allowed_strength_by_age(age_grp):
    if age_grp in (1, 2): return {1, 2, 3}
    if age_grp in (3, 4): return {1, 2}
    if age_grp == 5: return {1}
    return {1}

def allowed_strength_by_act(act):
    if pd.isna(act): return {1}
    if act < 40: return {1}
    if act < 60: return {1, 2}
    return {1, 2, 3}

def tolerance_f_upper(age_grp, act):
    if pd.isna(act): act = 50
    if age_grp == 5: return 5
    if age_grp in (3, 4): return 7
    if act < 40: return 7
    if act < 60: return 8
    return 10

# =================== 3. 核心仿真函数 ===================
def monthly_drop(s, f, pert_s=0.03, pert_f=0.01):
    if f < 5:
        return 0.0
    return pert_s * (s - 1) + pert_f * (f - 5)

def simulate(P0, g, s, f, pert_s=0.03, pert_f=0.01):
    delta = monthly_drop(s, f, pert_s, pert_f)
    traj = [P0]
    P = P0
    for _ in range(6):
        P *= (1 - delta)
        traj.append(P)
    cost = 6 * GRADE_COST[g] + 24 * f * SESSION_COST[s]
    return traj[-1], cost, traj

# =================== 4. Pareto 前沿 + TOPSIS ===================
def pareto_front(P0, age_grp, act):
    g = grade_from_tanshi(P0)
    S_allowed = allowed_strength_by_age(age_grp) & allowed_strength_by_act(act)
    if not S_allowed:
        S_allowed = {1}
    f_tol = tolerance_f_upper(age_grp, act)
    f_ub = min(F_MAX, f_tol)

    candidates = []
    for s, f in product(sorted(S_allowed), range(F_MIN, f_ub + 1)):
        P6, cost, _ = simulate(P0, g, s, f)
        if cost > BUDGET:
            continue
        obj1 = P6 / P0
        obj2 = cost / BUDGET
        obj3 = f / f_tol
        candidates.append({
            'g': g, 's': s, 'f': f, 'P6': P6, 'cost': cost,
            'obj1': obj1, 'obj2': obj2, 'obj3': obj3
        })
    if not candidates:
        return []
    cdf = pd.DataFrame(candidates)
    n = len(cdf)
    mask = np.ones(n, dtype=bool)
    arr = cdf[['obj1', 'obj2', 'obj3']].values
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by = (
            (arr <= arr[i]).all(axis=1) & (arr < arr[i]).any(axis=1)
        )
        dominated_by[i] = False
        if dominated_by.any():
            mask[i] = False
    return cdf[mask].reset_index(drop=True).to_dict('records')

def topsis_select(pareto_solutions, weights=(0.7, 0.2, 0.1)):
    """
    TOPSIS 多目标决策选解
    先筛选"有实效"的方案 (f≥5, 即月下降率>0), 在这些中选最佳折中;
    若无有效解, 才退回到 f<5 方案
    """
    if not pareto_solutions:
        return None
    # 优先选 f≥5 的有效干预方案
    effective = [s for s in pareto_solutions if s['f'] >= 5]
    target_pool = effective if effective else pareto_solutions

    df_p = pd.DataFrame(target_pool)
    objs = df_p[['obj1', 'obj2', 'obj3']].values
    norm = np.sqrt((objs ** 2).sum(axis=0)) + 1e-12
    obj_norm = objs / norm
    obj_weighted = obj_norm * np.array(weights)
    ideal = obj_weighted.min(axis=0)
    worst = obj_weighted.max(axis=0)
    d_ideal = np.sqrt(((obj_weighted - ideal) ** 2).sum(axis=1))
    d_worst = np.sqrt(((obj_weighted - worst) ** 2).sum(axis=1))
    C = d_worst / (d_ideal + d_worst + 1e-12)
    return target_pool[int(np.argmax(C))]

# =================== 5. 蒙特卡罗鲁棒性 ===================
def monte_carlo_robust(P0, g, s, f, n_sim=500, seed=42):
    if f < 5:
        return {'P6_mean': P0, 'P6_std': 0.0, 'P6_q05': P0, 'P6_q95': P0}
    rng = np.random.default_rng(seed)
    P6_samples = []
    for _ in range(n_sim):
        ps = 0.03 + rng.uniform(-0.005, 0.005)
        pf = 0.01 + rng.uniform(-0.003, 0.003)
        P6, _, _ = simulate(P0, g, s, f, pert_s=ps, pert_f=pf)
        P6_samples.append(P6)
    arr = np.array(P6_samples)
    return {
        'P6_mean': arr.mean(), 'P6_std': arr.std(),
        'P6_q05':  np.quantile(arr, 0.05),
        'P6_q95':  np.quantile(arr, 0.95)
    }

# =================== 6. 全体 278 例求解 ===================
print('\n' + '='*60)
print('Pareto 前沿 + TOPSIS 折中 + 蒙特卡罗鲁棒')
print('='*60)

all_records = []
all_pareto_sols = []

for i in range(len(ts)):
    P0  = ts.loc[i, COL_TANSHI]
    age = int(ts.loc[i, COL_AGEGRP]) if pd.notna(ts.loc[i, COL_AGEGRP]) else 1
    act = ts.loc[i, COL_ACT]

    pareto = pareto_front(P0, age, act)
    if not pareto:
        all_pareto_sols.append([])
        continue
    # 突出健康改善优先性
    best = topsis_select(pareto, weights=(0.7, 0.2, 0.1))
    robust = monte_carlo_robust(P0, best['g'], best['s'], best['f'])
    drop_rate = ((P0 - best['P6']) / P0) * 100 if P0 > 0 else 0

    all_records.append({
        'ID': ts.loc[i, COL_ID],
        '原痰湿P0': P0,
        '年龄组': age,
        '活动总分': act,
        '调理等级g': best['g'],
        '活动强度s': best['s'],
        '每周频次f': best['f'],
        'Pareto解数': len(pareto),
        '6月P6': round(best['P6'], 2),
        '6月P6_MC均值': round(robust['P6_mean'], 2),
        '6月P6_MC标准差': round(robust['P6_std'], 3),
        'P6_95%CI下': round(robust['P6_q05'], 2),
        'P6_95%CI上': round(robust['P6_q95'], 2),
        '累积改善%': round(drop_rate, 2),
        '6月总成本': best['cost'],
        '耐受度上限': tolerance_f_upper(age, act),
        'obj1_改善': round(best['obj1'], 3),
        'obj2_成本': round(best['obj2'], 3),
        'obj3_耐受': round(best['obj3'], 3)
    })
    all_pareto_sols.append(pareto)

plans = pd.DataFrame(all_records)
plans.to_csv(f'{TABLE_DIR}/p3_all_plans.csv',
             index=False, encoding='utf-8-sig')

print(f'\n完成, 共 {len(plans)} 例有解')
print('统计摘要:')
print(plans[['原痰湿P0', '调理等级g', '活动强度s', '每周频次f',
             '6月P6', '累积改善%', '6月总成本', 'Pareto解数']].describe().round(2))

pattern = plans.groupby(['调理等级g', '活动强度s', '每周频次f'])\
               .size().sort_values(ascending=False)\
               .reset_index(name='案例数')
pattern.to_csv(f'{TABLE_DIR}/p3_pattern.csv',
               index=False, encoding='utf-8-sig')
print('\nTOP 方案模式:')
print(pattern.head(10).to_string(index=False))

# 样本 1/2/3
samples_best = plans[plans['ID'].isin([1, 2, 3])].sort_values('ID').reset_index(drop=True)
samples_best.to_csv(f'{TABLE_DIR}/p3_samples_best.csv',
                    index=False, encoding='utf-8-sig')
print('\n样本 1, 2, 3 方案:')
print(samples_best[['ID', '原痰湿P0', '调理等级g', '活动强度s', '每周频次f',
                    '6月P6', '6月P6_MC均值', 'P6_95%CI下', 'P6_95%CI上',
                    '累积改善%', '6月总成本']].to_string(index=False))

# =================== 7. K-means 聚类 ===================
print('\n' + '='*60)
print('K-means 聚类 → 模板方案')
print('='*60)
cluster_feat = plans[['原痰湿P0', '年龄组', '活动总分']].fillna(
    plans[['原痰湿P0', '年龄组', '活动总分']].median())
scaler_c = StandardScaler().fit(cluster_feat)
cluster_feat_s = scaler_c.transform(cluster_feat)

inertias = []
for k in range(2, 8):
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(cluster_feat_s)
    inertias.append(km_k.inertia_)

K_OPT = 5
kmeans = KMeans(n_clusters=K_OPT, n_init=10, random_state=42)
plans['聚类ID'] = kmeans.fit_predict(cluster_feat_s)

templates = []
for k in range(K_OPT):
    sub = plans[plans['聚类ID'] == k]
    if len(sub) == 0: continue
    mode_plan = sub.groupby(['调理等级g', '活动强度s', '每周频次f'])\
                   .size().idxmax()
    g, s, f = mode_plan
    templates.append({
        '聚类ID': k,
        '患者数': len(sub),
        '平均P0': round(sub['原痰湿P0'].mean(), 1),
        '平均年龄组': round(sub['年龄组'].mean(), 1),
        '平均活动': round(sub['活动总分'].mean(), 1),
        '模板g': g, '模板s': s, '模板f': f,
        '平均6月P6': round(sub['6月P6'].mean(), 1),
        '平均成本': int(sub['6月总成本'].mean())
    })
tpl_df = pd.DataFrame(templates).sort_values('聚类ID')
tpl_df.to_csv(f'{TABLE_DIR}/p3_cluster_templates.csv',
              index=False, encoding='utf-8-sig')
print(tpl_df.to_string(index=False))

# =================== 8. 可视化：Pareto 与多目标权衡 ===================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (a) 样本1 Pareto 前沿
ax = axes[0, 0]
if len(samples_best) > 0:
    row1 = samples_best.iloc[0]
    sid1 = int(row1['ID'])
    ts_idx = ts[ts[COL_ID] == sid1].index
    if len(ts_idx) > 0:
        pareto1 = all_pareto_sols[ts_idx[0]]
        if pareto1:
            pdf = pd.DataFrame(pareto1).sort_values(['obj1', 'obj2'])
            sc = ax.scatter(pdf['obj1'], pdf['obj2'],
                            c=pdf['obj3'], cmap='Blues',
                            s=55, edgecolor='black', linewidth=0.4)
            sel = topsis_select(pareto1)
            ax.scatter(sel['obj1'], sel['obj2'], marker='*',
                       s=220, color='#C44E52', edgecolor='black',
                       linewidth=0.8, label='TOPSIS选解', zorder=10)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('耐受目标 obj3 = f / f_tol')
            ax.set_xlabel('改善目标 obj1 = P6 / P0')
            ax.set_ylabel('成本目标 obj2 = C / 2000')
            ax.set_title(f'（a）样本 {sid1} 的 Pareto 前沿', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.35)
            ax.legend(fontsize=9)

# (b) 样本1/2/3 Pareto前沿对比
ax = axes[0, 1]
for idx in range(len(samples_best)):
    row = samples_best.iloc[idx]
    sid = int(row['ID'])
    ts_idx = ts[ts[COL_ID] == sid].index
    if len(ts_idx) == 0:
        continue
    pareto_i = all_pareto_sols[ts_idx[0]]
    if not pareto_i:
        continue
    pdf = pd.DataFrame(pareto_i).sort_values(['obj1', 'obj2'])
    ax.plot(pdf['obj1'], pdf['obj2'], '-o',
            color=colors_s[idx], lw=1.5, markersize=4.5,
            markerfacecolor='white', markeredgewidth=0.9,
            alpha=0.95, label=f'样本{sid}')
    sel = topsis_select(pareto_i)
    ax.scatter(sel['obj1'], sel['obj2'], marker='*',
               s=180, color=colors_s[idx], edgecolor='black',
               linewidth=0.8, zorder=10)
ax.set_xlabel('改善目标 obj1 = P6 / P0')
ax.set_ylabel('成本目标 obj2 = C / 2000')
ax.set_title('（b）样本 1/2/3 的 Pareto 前沿对比', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

# (c) 全体患者改善-成本分布
ax = axes[1, 0]
sc = ax.scatter(plans['6月总成本'], plans['累积改善%'],
                c=plans['每周频次f'], cmap='Blues',
                s=24, alpha=0.75, edgecolors='none')
ax.axvline(BUDGET, ls='--', color='#C44E52', lw=1.2, label='预算上限')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('最优周频次 f')
ax.set_xlabel('6个月总成本（元）')
ax.set_ylabel('累积改善率（%）')
ax.set_title('（c）全体患者改善率与成本分布', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

x = plans['6月总成本'].values
y = plans['累积改善%'].values
if len(x) >= 3:
    coef = np.polyfit(x, y, 2)
    xx = np.linspace(x.min(), x.max(), 300)
    yy = np.polyval(coef, xx)
    ax.plot(xx, yy, color='#DD8452', lw=1.8, label='二次趋势')
    ax.legend(fontsize=9)

# (d) 最优频次与耐受度上限分布
ax = axes[1, 1]
f_cnt = plans['每周频次f'].value_counts().sort_index()
f_tol_cnt = plans['耐受度上限'].value_counts().sort_index()
x_all = np.arange(1, 11)

f_vals = [f_cnt.get(i, 0) for i in x_all]
tol_vals = [f_tol_cnt.get(i, 0) for i in x_all]

ax.bar(x_all - 0.18, f_vals, width=0.36,
       label='最优周频次', color='#4C72B0',
       edgecolor='black', linewidth=0.4)
ax.bar(x_all + 0.18, tol_vals, width=0.36,
       label='耐受上限', color='#DD8452',
       edgecolor='black', linewidth=0.4)

ax.set_xlabel('每周训练频次')
ax.set_ylabel('病例数')
ax.set_xticks(x_all)
ax.set_title('（d）最优频次与耐受上限比较', fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

plt.suptitle('问题三（A）：Pareto 前沿与多目标权衡结果', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{FIG_DIR}/p3_01_pareto.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/p3_01_pareto.pdf', bbox_inches='tight')
plt.close()
print('\n✓ 保存图: p3_01_pareto.png')

# =================== 9. 可视化：蒙特卡罗鲁棒性分析 ===================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (a) 样本 1/2/3 扰动轨迹
ax = axes[0, 0]
rng = np.random.default_rng(42)

for idx in range(len(samples_best)):
    row = samples_best.iloc[idx]
    P0 = row['原痰湿P0']
    g = int(row['调理等级g'])
    s = int(row['活动强度s'])
    f = int(row['每周频次f'])
    sid = int(row['ID'])

    traj_mc = []
    for _ in range(200):
        ps = 0.03 + rng.uniform(-0.005, 0.005)
        pf = 0.01 + rng.uniform(-0.003, 0.003)
        _, _, traj = simulate(P0, g, s, f, pert_s=ps, pert_f=pf)
        traj_mc.append(traj)
    traj_mc = np.array(traj_mc)

    _, _, traj_det = simulate(P0, g, s, f)
    q05 = np.quantile(traj_mc, 0.05, axis=0)
    q95 = np.quantile(traj_mc, 0.95, axis=0)
    month = np.arange(7)

    ax.fill_between(month, q05, q95, color=colors_s[idx], alpha=0.15)
    ax.plot(month, traj_det, '-o', color=colors_s[idx],
            lw=2.0, markersize=5, markerfacecolor='white',
            markeredgewidth=0.9,
            label=f'样本{sid}: g={g}, s={s}, f={f}')

ax.set_xlabel('月份')
ax.set_ylabel('痰湿积分')
ax.set_xticks(range(7))
ax.set_title('（a）样本 1/2/3 干预轨迹及扰动区间', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=8.8)

# (b) 样本1/2/3 的 P6 区间估计
ax = axes[0, 1]
if len(samples_best) > 0:
    x_pos = np.arange(len(samples_best))
    means = samples_best['6月P6_MC均值'].values
    q05s = samples_best['P6_95%CI下'].values
    q95s = samples_best['P6_95%CI上'].values
    p0s = samples_best['原痰湿P0'].values

    err_low = means - q05s
    err_high = q95s - means

    ax.bar(x_pos, means,
           color=colors_s[:len(samples_best)],
           edgecolor='black', linewidth=0.5,
           yerr=[err_low, err_high], capsize=6,
           error_kw={'lw': 1.2, 'capthick': 1.2})

    for i in range(len(samples_best)):
        ax.text(i, means[i] + 1.2, f'{means[i]:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'样本{int(s)}\nP0={int(p)}'
                        for s, p in zip(samples_best['ID'], p0s)],
                       fontsize=10)
    ax.set_ylabel('6个月后痰湿积分')
    ax.set_title('（b）样本 1/2/3 的 P6 区间估计', fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)

# (c) 全体方案鲁棒性分布
ax = axes[1, 0]
ci_width = plans['P6_95%CI上'] - plans['P6_95%CI下']
ax.hist(ci_width, bins=25, color='#4C72B0',
        edgecolor='black', linewidth=0.4, alpha=0.85)
ax.axvline(ci_width.mean(), ls='--', color='#C44E52',
           lw=1.3, label=f'均值 = {ci_width.mean():.2f}')
ax.set_xlabel('区间宽度（P6 不确定性）')
ax.set_ylabel('病例数')
ax.set_title('（c）全体方案鲁棒性分布', fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

# (d) 改善率-鲁棒性关系
ax = axes[1, 1]
sc = ax.scatter(plans['累积改善%'], ci_width,
                c=plans['6月总成本'], cmap='Blues',
                s=24, alpha=0.75, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('6个月总成本（元）')
ax.set_xlabel('累积改善率（%）')
ax.set_ylabel('区间宽度')
ax.set_title('（d）改善率与鲁棒性的关系', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)

plt.suptitle('问题三（B）：蒙特卡罗鲁棒性分析结果', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{FIG_DIR}/p3_02_robust.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/p3_02_robust.pdf', bbox_inches='tight')
plt.close()
print('✓ 保存图: p3_02_robust.png')

# =================== 10. 可视化：聚类分型与模板方案 ===================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (a) 聚类散点图
ax = axes[0, 0]
for k in range(K_OPT):
    sub = plans[plans['聚类ID'] == k]
    ax.scatter(sub['原痰湿P0'], sub['活动总分'],
               color=COLORS_CLUST[k], s=30, alpha=0.78,
               edgecolors='white', linewidths=0.3,
               label=f'类{k}（n={len(sub)}）')
centers_denorm = scaler_c.inverse_transform(kmeans.cluster_centers_)
for k in range(K_OPT):
    x0, _, y0 = centers_denorm[k]
    ax.scatter(x0, y0, marker='X', s=150,
               color=COLORS_CLUST[k],
               edgecolor='black', linewidth=0.8, zorder=10)
ax.set_xlabel('初始痰湿积分 P0')
ax.set_ylabel('活动总分')
ax.set_title(f'（a）K-means 聚类结果（K={K_OPT}）', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=8.2, loc='best')

# (b) 肘部法
ax = axes[0, 1]
ks = list(range(2, 8))
ax.plot(ks, inertias, '-o',
        color='#4C72B0', lw=1.8, markersize=5.5,
        markerfacecolor='white', markeredgewidth=1.0)
ax.axvline(K_OPT, ls='--', color='#C44E52', lw=1.2, label=f'选取 K={K_OPT}')
ax.set_xlabel('聚类数 K')
ax.set_ylabel('组内平方和 Inertia')
ax.set_title('（b）肘部法确定聚类数', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

# (c) 样本 1/2/3 确定性轨迹
ax = axes[1, 0]
for idx in range(len(samples_best)):
    row = samples_best.iloc[idx]
    P0 = row['原痰湿P0']
    g = int(row['调理等级g'])
    s = int(row['活动强度s'])
    f = int(row['每周频次f'])
    sid = int(row['ID'])

    _, _, traj = simulate(P0, g, s, f)
    ax.plot(range(7), traj, '-o',
            color=colors_s[idx], lw=2.0, markersize=5,
            markerfacecolor='white', markeredgewidth=0.9,
            label=f'样本{sid}')
    ax.text(6.08, traj[-1], f'{traj[-1]:.1f}',
            fontsize=9, color=colors_s[idx], va='center')

ax.set_xlabel('月份')
ax.set_ylabel('痰湿积分')
ax.set_xticks(range(7))
ax.set_title('（c）样本 1/2/3 的确定性干预轨迹', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=9)

# (d) 模板方案表嵌入图
ax = axes[1, 1]
ax.axis('off')
tpl_show = tpl_df[['聚类ID', '患者数', '平均P0', '平均活动',
                   '模板g', '模板s', '模板f', '平均6月P6', '平均成本']].copy()
tpl_show.columns = ['类', 'n', '平均P0', '平均活动', 'g', 's', 'f', '平均P6', '平均成本']

table = ax.table(cellText=tpl_show.values.tolist(),
                 colLabels=tpl_show.columns.tolist(),
                 cellLoc='center', colLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.08, 1.55)

for (r, c), cell in table.get_celld().items():
    cell.set_linewidth(0.4)
    cell.set_edgecolor('#666666')
    if r == 0:
        cell.set_facecolor('#D9E2F3')
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('white')

ax.set_title('（d）各聚类对应的模板方案', fontsize=11, pad=10)

plt.suptitle('问题三（C）：患者分型与模板方案结果', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{FIG_DIR}/p3_03_cluster.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/p3_03_cluster.pdf', bbox_inches='tight')
plt.close()
print('✓ 保存图: p3_03_cluster.png')
