# -*- coding: utf-8 -*-
"""
MathorCup 2026 C题 - 问题2 最终版
功能：
1. 构建高血脂三级风险预警模型（低/中/高）
2. 使用 OOF 概率进行严格评估，避免训练集乐观偏差
3. 融合 R 综合风险分 + OOF概率 + 规则阈值
4. 自动保证三级风险均有样本，提升结果稳定性与可解释性
5. 输出图表与结果表，可直接用于论文

注意：
- 机器学习主模型不使用核心血脂项，避免直接数据泄漏
- 最终三级分层允许结合题目示例规则（血脂异常 + 痰湿 + 活动）
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# =========================
# 可选高级库
# =========================
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from mord import LogisticIT
    HAS_MORD = True
except Exception:
    HAS_MORD = False

# =========================
# 路径设置
# =========================
DATA_PATH = r'./data.xlsx'
OUT_DIR   = r'./task2'
FIG_DIR   = os.path.join(OUT_DIR, 'figures')
TAB_DIR   = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# =========================
# 中文字体 + 论文风格绘图参数
# =========================
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

def set_chinese_font():
    """
    更稳健地解决中文乱码：
    1. 优先按字体名称查找，再取对应字体文件路径
    2. 全局注册中文字体，避免标题/坐标轴/图例/树图等局部回退
    3. 关闭 unicode 负号乱码
    """
    candidate_fonts = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
        'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]

    font_path = None
    font_name = None

    for f in font_manager.fontManager.ttflist:
        if f.name in candidate_fonts:
            font_path = f.fname
            font_name = f.name
            break

    if font_path is None:
        # 最后兜底
        for f in font_manager.fontManager.ttflist:
            if 'DejaVu Sans' in f.name:
                font_path = f.fname
                font_name = f.name
                break

    if font_path is None:
        # 极端情况下仍给默认值
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return FontProperties()

    cn_font = FontProperties(fname=font_path)

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    matplotlib.rcParams['font.serif'] = [font_name, 'DejaVu Serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    return cn_font

CN_FONT = set_chinese_font()

# 统一论文风格
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.8,
    'patch.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

# 更偏数模论文风格的低饱和配色
COLORS_3 = ['#4C72B0', '#DD8452', '#55A868']
C_NEG = '#4C72B0'
C_ACC = '#8172B2'
C_POS = '#C44E52'

def apply_chinese_to_ax(ax):
    """统一对子图中的中文文本应用字体"""
    if ax.title:
        ax.title.set_fontproperties(CN_FONT)

    ax.xaxis.label.set_fontproperties(CN_FONT)
    ax.yaxis.label.set_fontproperties(CN_FONT)

    for label in ax.get_xticklabels():
        label.set_fontproperties(CN_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(CN_FONT)
        legend.get_title().set_fontproperties(CN_FONT)

    for txt in ax.texts:
        txt.set_fontproperties(CN_FONT)

def beautify_ax(ax):
    """统一子图风格"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, linestyle='--', alpha=0.22, linewidth=0.6)
    apply_chinese_to_ax(ax)

def save_fig(fig, path):
    # 对整张图中的所有子图再次统一应用中文字体
    for ax in fig.axes:
        apply_chinese_to_ax(ax)

    # 总标题
    if hasattr(fig, '_suptitle') and fig._suptitle is not None:
        fig._suptitle.set_fontproperties(CN_FONT)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========================
# 工具函数
# =========================
def find_col(df, kw):
    for c in df.columns:
        if kw in c:
            return c
    return None

def sex_specific_ua_risk(ua, sex):
    """
    血尿酸上限：
    男 428, 女 357
    仅对超上限部分做风险归一化
    """
    if pd.isna(ua):
        return 0.0
    upper = 428 if sex == 1 else 357
    return max(0.0, (ua - upper) / 200.0)

def lipid_abnormal_count(row, COL):
    """
    仅用于最终规则分层，不用于机器学习主模型训练。
    参考题目给定范围：
    TG > 1.7, LDL > 3.1, HDL < 1.04, TC > 6.2
    """
    abn = 0
    sev = []

    if pd.notna(row[COL['TG']]) and row[COL['TG']] > 1.7:
        abn += 1
        sev.append((row[COL['TG']] - 1.7) / 1.7)

    if pd.notna(row[COL['TC']]) and row[COL['TC']] > 6.2:
        abn += 1
        sev.append((row[COL['TC']] - 6.2) / 6.2)

    if pd.notna(row[COL['LDL']]) and row[COL['LDL']] > 3.1:
        abn += 1
        sev.append((row[COL['LDL']] - 3.1) / 3.1)

    if pd.notna(row[COL['HDL']]) and row[COL['HDL']] < 1.04:
        abn += 1
        sev.append((1.04 - row[COL['HDL']]) / 1.04)

    severity = float(np.clip(np.mean(sev) if sev else 0.0, 0, 1))
    return abn, severity

def safe_fill_median(train_df, test_df, cols):
    med = train_df[cols].median()
    return train_df[cols].fillna(med), test_df[cols].fillna(med), med

def dca_curve(y_true, y_prob, thresholds):
    """
    决策曲线分析
    NB = TP/N - FP/N * p/(1-p)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    N = len(y_true)
    prevalence = y_true.mean()

    nb_model, nb_all, nb_none = [], [], []
    for p in thresholds:
        if p >= 0.999:
            nb_model.append(0.0)
            nb_all.append(0.0)
            nb_none.append(0.0)
            continue

        pred_pos = (y_prob >= p)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        odds_ratio = p / (1 - p)

        nb_m = tp / N - fp / N * odds_ratio
        nb_a = prevalence - (1 - prevalence) * odds_ratio

        nb_model.append(nb_m)
        nb_all.append(nb_a)
        nb_none.append(0.0)

    return np.array(nb_model), np.array(nb_all), np.array(nb_none)

def select_thresholds_stable(df_eval, target_col='y', r_col='R'):
    """
    稳健阈值选择：
    - 低风险：发病率 <= 15%，占比在 [8%, 35%]
    - 高风险：发病率 >= 80%，占比在 [10%, 35%]
    若找不到满足条件的候选，则回退到分位数
    """
    n = len(df_eval)
    q_candidates = np.linspace(0.10, 0.90, 81)

    low_candidates = []
    high_candidates = []

    r_values = df_eval[r_col].values

    for q in q_candidates:
        thr = np.quantile(r_values, q)
        sub = df_eval[df_eval[r_col] <= thr]
        prop = len(sub) / n
        if len(sub) > 0:
            rate = sub[target_col].mean()
            low_candidates.append((thr, prop, rate))

    for q in q_candidates:
        thr = np.quantile(r_values, q)
        sub = df_eval[df_eval[r_col] >= thr]
        prop = len(sub) / n
        if len(sub) > 0:
            rate = sub[target_col].mean()
            high_candidates.append((thr, prop, rate))

    low_ok = [x for x in low_candidates if x[2] <= 0.15 and 0.08 <= x[1] <= 0.35]
    if len(low_ok) > 0:
        R_LO = max(low_ok, key=lambda x: x[0])[0]
    else:
        R_LO = np.quantile(r_values, 0.25)

    high_ok = [x for x in high_candidates if x[2] >= 0.80 and 0.10 <= x[1] <= 0.35]
    if len(high_ok) > 0:
        R_HI = min(high_ok, key=lambda x: x[0])[0]
    else:
        R_HI = np.quantile(r_values, 0.75)

    return round(float(R_LO), 2), round(float(R_HI), 2)

def stratify_joint(row, R_LO, R_HI, p_low, p_high, COL):
    """
    联合分层：
    高风险：R高且概率高，或触发硬规则
    低风险：满足足够多低风险信号，且概率不能太高
    中风险：其余
    """
    R   = row['R']
    p   = row['p_oof']
    ts  = row[COL['TANSHI']]
    act = row[COL['ACT']]
    abn = row['异常项数']

    # 高风险硬规则
    hard_high = (
        (abn >= 2 and ts >= 60) or
        (abn == 0 and ts >= 80 and act < 40) or
        (abn >= 3)
    )

    # 低风险信号计数
    low_signals = 0
    if R <= R_LO:
        low_signals += 1
    if p <= p_low:
        low_signals += 1
    if abn == 0:
        low_signals += 1
    if ts < 40:
        low_signals += 1
    if act >= 60:
        low_signals += 1

    if (R >= R_HI and p >= p_high) or hard_high:
        return '高风险'

    if ((low_signals >= 4 and p <= 0.5) or
        (R <= R_LO and p <= p_low and abn == 0)):
        return '低风险'

    return '中风险'

def apply_ratio_constraint(df, low_ratio=0.10, high_ratio_floor=0.20):
    """
    分层比例约束：
    1. 保证低风险至少占 low_ratio
    2. 不降低已有高风险，仅从中风险中补低风险
    """
    df = df.copy()
    n = len(df)
    n_low_target = int(n * low_ratio)

    current_low = df[df['风险分层'] == '低风险']
    if len(current_low) < n_low_target:
        need = n_low_target - len(current_low)
        candidates = df[df['风险分层'] == '中风险'].copy()
        candidates = candidates.sort_values(['R', 'p_oof', '异常项数', '异常严重度'])
        add_ids = candidates.head(need).index
        df.loc[add_ids, '风险分层'] = '低风险'

    # 防止高风险过低（一般不会触发）
    current_high = df[df['风险分层'] == '高风险']
    n_high_floor = int(n * high_ratio_floor)
    if len(current_high) < n_high_floor:
        need = n_high_floor - len(current_high)
        candidates = df[df['风险分层'] == '中风险'].copy()
        candidates = candidates.sort_values(['R', 'p_oof', '异常项数'], ascending=False)
        add_ids = candidates.head(need).index
        df.loc[add_ids, '风险分层'] = '高风险'

    return df

# =========================
# 读入数据
# =========================
df = pd.read_excel(DATA_PATH)
print(f'数据形状: {df.shape}')

COL = {
    'HDL'   : find_col(df, 'HDL'),
    'LDL'   : find_col(df, 'LDL'),
    'TG'    : find_col(df, 'TG'),
    'TC'    : find_col(df, 'TC'),
    'BMI'   : find_col(df, 'BMI'),
    'GLU'   : find_col(df, '空腹血糖'),
    'UA'    : find_col(df, '血尿酸'),
    'ACT'   : find_col(df, '活动量表总分'),
    'TANSHI': find_col(df, '痰湿质'),
    'TARGET': find_col(df, '高血脂'),
    'TYPE'  : find_col(df, '体质标签'),
    'AGEGRP': find_col(df, '年龄组'),
    'SEX'   : find_col(df, '性别'),
    'SMK'   : find_col(df, '吸烟史'),
    'DRK'   : find_col(df, '饮酒史'),
    'ID'    : find_col(df, '样本ID')
}

# =========================
# 1. 构造子风险
# =========================
abn_tmp = df.apply(lambda r: pd.Series(lipid_abnormal_count(r, COL)), axis=1)
df[['异常项数', '异常严重度']] = abn_tmp

df['LipidR'] = 0.6 * (df['异常项数'] / 4.0) + 0.4 * df['异常严重度']
df['TanShiR'] = df[COL['TANSHI']].fillna(0) / 100.0
df['ActR'] = np.clip((60.0 - df[COL['ACT']].fillna(60)) / 60.0, 0, 1)

def build_met_r(row):
    vals = []

    if pd.notna(row[COL['BMI']]):
        vals.append(max(0.0, row[COL['BMI']] - 24.0) / 10.0)

    if pd.notna(row[COL['GLU']]):
        vals.append(max(0.0, row[COL['GLU']] - 6.1) / 4.0)

    vals.append(sex_specific_ua_risk(
        row[COL['UA']],
        row[COL['SEX']] if pd.notna(row[COL['SEX']]) else 0
    ))

    return float(np.clip(max(vals) if len(vals) > 0 else 0.0, 0, 1))

df['MetR'] = df.apply(build_met_r, axis=1)

# 熵权 + 专家先验
sub_risks = df[['LipidR', 'TanShiR', 'ActR', 'MetR']].values
n_samp = len(sub_risks)
eps = 1e-12

col_sum = sub_risks.sum(axis=0) + eps
P_mat = sub_risks / col_sum

entropy = np.zeros(4)
for j in range(4):
    p = P_mat[:, j]
    mask = p > eps
    entropy[j] = -np.sum(p[mask] * np.log(p[mask])) / np.log(n_samp)

redundancy = 1 - entropy
w_obj = redundancy / redundancy.sum()

# 专家权重：代谢和血脂略高
w_exp = np.array([0.35, 0.22, 0.18, 0.25])
w_final = 0.5 * w_obj + 0.5 * w_exp
w_final = w_final / w_final.sum()

print('\n子风险权重 (熵权法+专家先验 融合):')
for name, w in zip(['LipidR', 'TanShiR', 'ActR', 'MetR'], w_final):
    print(f'  {name:10s}: {w:.4f}')

df['R'] = 100 * (
    w_final[0] * df['LipidR'] +
    w_final[1] * df['TanShiR'] +
    w_final[2] * df['ActR'] +
    w_final[3] * df['MetR']
)

# =========================
# 2. 主模型特征工程（不含核心血脂）
# =========================
safe_feats = [
    COL['TANSHI'], COL['BMI'], COL['GLU'], COL['UA'], COL['ACT'],
    COL['AGEGRP'], COL['SEX'], COL['SMK'], COL['DRK']
]
safe_feats = [c for c in safe_feats if c is not None]

df['痰湿×活动'] = df[COL['TANSHI']] * (100 - df[COL['ACT']].fillna(60)) / 100
df['痰湿×BMI']  = df[COL['TANSHI']] * df[COL['BMI']].fillna(24) / 24
df['痰湿×尿酸'] = df[COL['TANSHI']] * df[COL['UA']].fillna(df[COL['UA']].median()) / max(df[COL['UA']].median(), 1)

feats_main = safe_feats + ['痰湿×活动', '痰湿×BMI', '痰湿×尿酸']
X_all = df[feats_main].copy()
y_all = df[COL['TARGET']].astype(int).values

print(f'\n特征总数: {len(feats_main)} (机器学习主模型不含核心血脂项)')
print(f'  基础特征: {safe_feats}')
print(f'  交互特征: 痰湿×活动, 痰湿×BMI, 痰湿×尿酸')

# =========================
# 3. 主模型
# =========================
if HAS_XGB:
    base_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    core_name = 'XGBoost'
else:
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    core_name = 'GradientBoosting'

print('\n' + '='*60)
print('训练核心模型（OOF 概率评估）')
print('='*60)

# =========================
# 4. OOF 概率评估
# =========================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_prob = np.zeros(len(df))
fold_auc = []
fold_brier = []

for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all), 1):
    X_tr = X_all.iloc[tr_idx].copy()
    X_te = X_all.iloc[te_idx].copy()
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    X_tr_fill, X_te_fill, med = safe_fill_median(X_tr, X_te, feats_main)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_fill)
    X_te_s = scaler.transform(X_te_fill)

    clf = CalibratedClassifierCV(
        estimator=clone(base_model),
        method='isotonic',
        cv=3
    )
    clf.fit(X_tr_s, y_tr)
    p_te = clf.predict_proba(X_te_s)[:, 1]

    oof_prob[te_idx] = p_te
    fold_auc.append(roc_auc_score(y_te, p_te))
    fold_brier.append(brier_score_loss(y_te, p_te))

df['p_oof'] = oof_prob

auc_mean = float(np.mean(fold_auc))
auc_std = float(np.std(fold_auc))
brier_oof = float(brier_score_loss(y_all, oof_prob))

print(f'  {core_name} + Isotonic 5折OOF AUC: {auc_mean:.4f} ± {auc_std:.4f}')
print(f'  OOF Brier Score: {brier_oof:.4f}')

# =========================
# 5. 全样本最终模型（仅解释用）
# =========================
X_full = X_all.fillna(X_all.median())
scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X_full)

final_cal_model = CalibratedClassifierCV(
    estimator=clone(base_model),
    method='isotonic',
    cv=5
)
final_cal_model.fit(X_full_s, y_all)

final_uncal_model = clone(base_model)
final_uncal_model.fit(X_full_s, y_all)

# =========================
# 6. DCA 与阈值
# =========================
print('\n' + '='*60)
print('DCA 决策曲线 + 分层阈值选择')
print('='*60)

thresholds = np.linspace(0.01, 0.99, 99)
nb_model, nb_all, nb_none = dca_curve(y_all, df['p_oof'].values, thresholds)

p_low = 0.30
p_high = 0.75

eval_df = pd.DataFrame({
    'R': df['R'],
    'y': y_all
})
R_LO, R_HI = select_thresholds_stable(eval_df, target_col='y', r_col='R')

print(f'  R 分双阈值: R_LO = {R_LO}, R_HI = {R_HI}')
print(f'  OOF概率辅助阈值: p_low = {p_low}, p_high = {p_high}')

# =========================
# 7. 初始三级分层
# =========================
df['风险分层'] = df.apply(
    lambda row: stratify_joint(row, R_LO, R_HI, p_low, p_high, COL),
    axis=1
)

# =========================
# 8. 分层比例约束（关键）
# =========================
df = apply_ratio_constraint(df, low_ratio=0.10, high_ratio_floor=0.20)

rates = df.groupby('风险分层').agg(
    样本数=('风险分层', 'size'),
    发病率=(COL['TARGET'], 'mean'),
    平均概率=('p_oof', 'mean'),
    平均R分=('R', 'mean'),
    平均痰湿=(COL['TANSHI'], 'mean')
).reindex(['低风险', '中风险', '高风险']).round(3)

print('\n========= 三级分层结果 =========')
print(rates)

rates.to_csv(os.path.join(TAB_DIR, 'p2_risk_rates_final.csv'),
             encoding='utf-8-sig')

# =========================
# 9. 有序Logistic 对比
# =========================
print('\n' + '='*60)
print('有序 Logistic 回归（对比）')
print('='*60)

level_map = {'低风险': 0, '中风险': 1, '高风险': 2}
y_ordinal = df['风险分层'].map(level_map).values

if HAS_MORD:
    olr = LogisticIT(alpha=1.0)
    olr.fit(X_full_s, y_ordinal)
    y_ord_pred = olr.predict(X_full_s)
    olr_acc = (y_ord_pred == y_ordinal).mean()
    print(f'  有序Logistic (mord.LogisticIT) 训练精度: {olr_acc:.4f}')
else:
    from sklearn.linear_model import LogisticRegression
    olr = LogisticRegression(max_iter=2000, random_state=42)
    olr.fit(X_full_s, y_ordinal)
    y_ord_pred = olr.predict(X_full_s)
    olr_acc = (y_ord_pred == y_ordinal).mean()
    print(f'  多分类 Logistic (降级) 训练精度: {olr_acc:.4f}')

# =========================
# 10. 特征重要性
# =========================
print('\n' + '='*60)
print('特征重要性 + 交互效应')
print('='*60)

if HAS_SHAP and HAS_XGB:
    explainer = shap.TreeExplainer(final_uncal_model)
    shap_vals = explainer.shap_values(X_full_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    feat_imp = np.abs(shap_vals).mean(axis=0)
    imp_name = 'SHAP均值'
else:
    perm = permutation_importance(
        final_uncal_model, X_full_s, y_all,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    feat_imp = perm.importances_mean
    imp_name = 'Permutation重要性'

imp_df = pd.DataFrame({
    '特征': feats_main,
    imp_name: np.round(feat_imp, 4)
}).sort_values(imp_name, ascending=False).reset_index(drop=True)

print(imp_df.to_string(index=False))
imp_df.to_csv(os.path.join(TAB_DIR, 'p2_feature_importance_final.csv'),
              index=False, encoding='utf-8-sig')

# =========================
# 11. 阈值总结表
# =========================
thr_summary = pd.DataFrame({
    '规则类型': [
        '低风险R阈值', '高风险R阈值',
        '低风险概率阈值', '高风险概率阈值',
        '高风险硬规则1', '高风险硬规则2', '高风险硬规则3',
        '低风险硬规则', '比例约束'
    ],
    '判据': [
        f'R ≤ {R_LO}',
        f'R ≥ {R_HI}',
        f'p_oof ≤ {p_low}',
        f'p_oof ≥ {p_high}',
        '异常项数 ≥ 2 且 痰湿 ≥ 60',
        '血脂正常 且 痰湿 ≥ 80 且 活动 < 40',
        '异常项数 ≥ 3',
        '血脂正常 且 痰湿 < 40 且 活动 ≥ 60',
        '低风险至少占10%'
    ],
    '依据': [
        '基于样本发病率的稳健低风险边界',
        '基于样本发病率的稳健高风险边界',
        '低风险辅助证据',
        '高风险辅助证据',
        '血脂-体质协同',
        '体质-行为叠加',
        '多指标血脂异常',
        '最严格低风险认定',
        '保证三级分层应用意义'
    ]
})
thr_summary.to_csv(os.path.join(TAB_DIR, 'p2_threshold_summary_final.csv'),
                   index=False, encoding='utf-8-sig')

# 保存患者级结果
save_cols = []
for c in [COL['ID'], COL['TARGET'], COL['TANSHI'], COL['ACT']]:
    if c is not None:
        save_cols.append(c)
save_cols += ['异常项数', '异常严重度', 'LipidR', 'TanShiR', 'ActR', 'MetR',
              'R', 'p_oof', '风险分层']
save_cols = list(dict.fromkeys(save_cols))

df[save_cols].to_csv(os.path.join(TAB_DIR, 'p2_final_risk_final.csv'),
                     index=False, encoding='utf-8-sig')

# =========================
# 12. 图1：总览
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(wspace=0.24, hspace=0.28)

# (a) 特征重要性
ax = axes[0, 0]
imp_show = imp_df.head(10).iloc[::-1]
y_pos = np.arange(len(imp_show))
bars = ax.barh(
    y_pos,
    imp_show[imp_name],
    color=C_NEG,
    alpha=0.88,
    edgecolor='black'
)
ax.set_yticks(y_pos)
ax.set_yticklabels(imp_show['特征'])
for bar, val in zip(bars, imp_show[imp_name]):
    ax.text(val + 0.0015, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5, fontproperties=CN_FONT)
ax.set_xlabel(imp_name, fontproperties=CN_FONT)
ax.set_title('(a) 主要特征重要性', fontproperties=CN_FONT)
beautify_ax(ax)

# (b) OOF概率分布
ax = axes[0, 1]
for lv, c in zip(['低风险', '中风险', '高风险'], COLORS_3):
    data = df[df['风险分层'] == lv]['p_oof']
    if len(data) > 0:
        ax.hist(
            data, bins=22, alpha=0.62, density=False,
            label=f'{lv}（n={len(data)}）',
            color=c, edgecolor='black'
        )
ax.axvline(p_low, ls='--', color='#2F7E2F', lw=1.4, label=f'低阈值={p_low}')
ax.axvline(p_high, ls='--', color='#A12A2A', lw=1.4, label=f'高阈值={p_high}')
ax.set_xlabel('OOF发病概率', fontproperties=CN_FONT)
ax.set_ylabel('样本数', fontproperties=CN_FONT)
ax.set_title('(b) OOF概率分布', fontproperties=CN_FONT)
ax.legend(frameon=True, loc='best', prop=CN_FONT)
beautify_ax(ax)

# (c) R vs OOF概率
ax = axes[1, 0]
for lv, c in zip(['低风险', '中风险', '高风险'], COLORS_3):
    sub = df[df['风险分层'] == lv]
    ax.scatter(
        sub['R'], sub['p_oof'],
        color=c, s=24, alpha=0.68,
        edgecolor='white', linewidth=0.3,
        label=f'{lv}（n={len(sub)}）'
    )
ax.axvline(R_LO, ls='--', color='#2F7E2F', alpha=0.9, lw=1.4, label=f'R_LO={R_LO}')
ax.axvline(R_HI, ls='--', color='#A12A2A', alpha=0.9, lw=1.4, label=f'R_HI={R_HI}')
ax.set_xlabel('R综合风险分', fontproperties=CN_FONT)
ax.set_ylabel('OOF发病概率', fontproperties=CN_FONT)
ax.set_title('(c) 风险分与概率联合分布', fontproperties=CN_FONT)
ax.legend(frameon=True, loc='best', prop=CN_FONT)
beautify_ax(ax)

# (d) 分层发病率
ax = axes[1, 1]
rate_vals = rates['发病率'].fillna(0).values
count_vals = rates['样本数'].fillna(0).values
x_pos = np.arange(3)
bars = ax.bar(
    x_pos, rate_vals,
    color=COLORS_3,
    edgecolor='black',
    width=0.58
)
for b, r, cnt in zip(bars, rate_vals, count_vals):
    cnt = int(cnt) if not pd.isna(cnt) else 0
    ax.text(
        b.get_x() + b.get_width()/2,
        r + 0.02,
        f'{r*100:.1f}%\n(n={cnt})',
        ha='center', va='bottom',
        fontsize=9.5, fontweight='bold',
        fontproperties=CN_FONT
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(['低风险', '中风险', '高风险'], fontproperties=CN_FONT)
ax.set_ylabel('实际发病率', fontproperties=CN_FONT)
ax.set_ylim(0, min(1.05, max(rate_vals) + 0.18))
ax.set_title('(d) 三级分层发病率验证', fontproperties=CN_FONT)
beautify_ax(ax)

fig.suptitle(f'问题2 三级风险预警模型总览（{core_name}）', fontsize=14, fontweight='bold', y=0.98, fontproperties=CN_FONT)
save_fig(fig, os.path.join(FIG_DIR, 'p2_final_01_overview.png'))
print('\n✓ 保存图: p2_final_01_overview.png')

# =========================
# 13. 图2：校准 / DCA / ROC / 混淆矩阵
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(wspace=0.26, hspace=0.30)

# (a) 校准曲线
ax = axes[0, 0]
frac_pos, mean_pred = calibration_curve(y_all, df['p_oof'], n_bins=10)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1.2, label='理想校准线')
ax.plot(
    mean_pred, frac_pos, 'o-',
    color=C_NEG, lw=2, markersize=6,
    label=f'OOF校准曲线（Brier={brier_oof:.3f}）'
)
ax.set_xlabel('预测概率（分箱均值）', fontproperties=CN_FONT)
ax.set_ylabel('实际发病频率', fontproperties=CN_FONT)
ax.set_title('(a) 校准曲线', fontproperties=CN_FONT)
ax.legend(frameon=True, loc='best', prop=CN_FONT)
beautify_ax(ax)

# (b) DCA
ax = axes[0, 1]
ax.plot(thresholds, nb_model, '-', color=C_NEG, lw=2.1, label=f'模型（{core_name}）')
ax.plot(thresholds, nb_all, '--', color=C_POS, lw=1.5, label='Treat-All')
ax.plot(thresholds, nb_none, ':', color='gray', lw=1.5, label='Treat-None')
ax.axvline(p_low, ls='-.', color='#2F7E2F', alpha=0.85, lw=1.2, label=f'低阈值={p_low}')
ax.axvline(p_high, ls='-.', color='#A12A2A', alpha=0.85, lw=1.2, label=f'高阈值={p_high}')
ax.set_xlabel('阈值概率', fontproperties=CN_FONT)
ax.set_ylabel('净收益', fontproperties=CN_FONT)
ax.set_title('(b) 决策曲线分析', fontproperties=CN_FONT)
ax.set_xlim(0, 1)
ax.legend(frameon=True, loc='best', prop=CN_FONT)
beautify_ax(ax)

# (c) ROC
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_all, df['p_oof'])
auc_oof = roc_auc_score(y_all, df['p_oof'])
ax.plot(fpr, tpr, '-', color=C_NEG, lw=2.2, label=f'ROC曲线（AUC={auc_oof:.3f}）')
ax.plot([0, 1], [0, 1], ':', color='gray', lw=1.2)
ax.set_xlabel('假正例率 FPR', fontproperties=CN_FONT)
ax.set_ylabel('真正例率 TPR', fontproperties=CN_FONT)
ax.set_title('(c) ROC曲线', fontproperties=CN_FONT)
ax.legend(loc='lower right', frameon=True, prop=CN_FONT)
beautify_ax(ax)

# (d) 高风险混淆矩阵
ax = axes[1, 1]
y_pred_high = (df['风险分层'] == '高风险').astype(int).values
cm = confusion_matrix(y_all, y_pred_high)
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

im = ax.imshow(cm, cmap='Blues', aspect='equal')
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                color=color, fontsize=13, fontweight='bold',
                fontproperties=CN_FONT)
ax.set_xticks([0, 1])
ax.set_xticklabels(['预测非高风险', '预测高风险'], fontproperties=CN_FONT)
ax.set_yticks([0, 1])
ax.set_yticklabels(['实际未发病', '实际发病'], fontproperties=CN_FONT)
ax.set_title(
    f'(d) 高风险识别混淆矩阵\n敏感性={sens*100:.1f}%  特异性={spec*100:.1f}%',
    fontsize=11, fontweight='bold', fontproperties=CN_FONT
)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
for t in cbar.ax.get_yticklabels():
    t.set_fontproperties(CN_FONT)

fig.suptitle('问题2 模型校准性与识别性能分析', fontsize=14, fontweight='bold', y=0.98, fontproperties=CN_FONT)
save_fig(fig, os.path.join(FIG_DIR, 'p2_final_02_calibration_dca.png'))
print('✓ 保存图: p2_final_02_calibration_dca.png')

# =========================
# 14. 图3：痰湿亚组分析
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(wspace=0.26, hspace=0.30)

# (a) 痰湿 × 血脂异常 热力图
ax = axes[0, 0]
df['_痰湿分段'] = pd.cut(
    df[COL['TANSHI']],
    bins=[-1, 20, 40, 60, 80, 101],
    labels=['0–20', '20–40', '40–60', '60–80', '80–100']
)
piv = df.pivot_table(
    index='异常项数',
    columns='_痰湿分段',
    values=COL['TARGET'],
    aggfunc='mean',
    observed=True
)

im = ax.imshow(piv.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if pd.notna(v):
            color = 'white' if v > 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=9,
                    fontproperties=CN_FONT)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, fontproperties=CN_FONT)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels([int(x) for x in piv.index], fontproperties=CN_FONT)
ax.set_xlabel('痰湿积分分段', fontproperties=CN_FONT)
ax.set_ylabel('血脂异常项数', fontproperties=CN_FONT)
ax.set_title('(a) 痰湿积分与血脂异常的联合热力图', fontproperties=CN_FONT)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label='发病率')
cbar.ax.yaxis.label.set_fontproperties(CN_FONT)
for t in cbar.ax.get_yticklabels():
    t.set_fontproperties(CN_FONT)
apply_chinese_to_ax(ax)

# (b) 痰湿体质患者风险占比
ax = axes[0, 1]
if COL['TYPE'] is not None:
    ts_sub = df[df[COL['TYPE']] == 5]
    prop = ts_sub['风险分层'].value_counts(normalize=True).reindex(['低风险', '中风险', '高风险']).fillna(0)
    wedges, texts, autotexts = ax.pie(
        prop.values,
        labels=prop.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=COLORS_3,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.0},
        textprops={'fontsize': 10}
    )
    for txt in texts:
        txt.set_fontproperties(CN_FONT)
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontproperties(CN_FONT)
    ax.set_title(f'(b) 痰湿体质患者风险结构（n={len(ts_sub)}）', fontproperties=CN_FONT)
else:
    ax.axis('off')

# (c) 痰湿体质子样本决策树
ax = axes[1, 0]
if COL['TYPE'] is not None:
    ts_sub = df[df[COL['TYPE']] == 5].copy()
    tree_feats = [COL['TANSHI'], COL['BMI'], COL['GLU'], COL['UA'], COL['ACT'], COL['AGEGRP']]
    tree_feats = [c for c in tree_feats if c is not None]
    Xt = ts_sub[tree_feats].fillna(ts_sub[tree_feats].median())
    yt = (ts_sub['风险分层'] == '高风险').astype(int)

    if yt.sum() >= 10 and (1 - yt).sum() >= 10:
        dt = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=42
        ).fit(Xt, yt)
        plot_tree(
            dt,
            feature_names=[str(f)[:8] for f in tree_feats],
            class_names=['非高风险', '高风险'],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax,
            precision=2
        )
        # 强制树图中的所有文本使用中文字体
        for txt in ax.texts:
            txt.set_fontproperties(CN_FONT)
        ax.set_title('(c) 痰湿体质高风险判别树', fontproperties=CN_FONT)
    else:
        ax.axis('off')
else:
    ax.axis('off')

# (d) 痰湿相关特征重要性
ax = axes[1, 1]
inter_feats = imp_df[imp_df['特征'].isin(
    ['痰湿×活动', '痰湿×BMI', '痰湿×尿酸', COL['TANSHI'], COL['ACT'], COL['BMI'], COL['UA']]
)]
if len(inter_feats) > 0:
    inter_feats = inter_feats.sort_values(imp_name, ascending=True)
    bars = ax.barh(
        inter_feats['特征'],
        inter_feats[imp_name],
        color=C_ACC,
        alpha=0.85,
        edgecolor='black'
    )
    for i, v in enumerate(inter_feats[imp_name]):
        ax.text(v + 0.0015, i, f'{v:.3f}', va='center', fontsize=8.5,
                fontproperties=CN_FONT)
    ax.set_xlabel(imp_name, fontproperties=CN_FONT)
    ax.set_title('(d) 痰湿相关变量及交互项重要性', fontproperties=CN_FONT)
    beautify_ax(ax)
else:
    ax.axis('off')

fig.suptitle('问题2 痰湿体质高风险特征组合分析', fontsize=14, fontweight='bold', y=0.98, fontproperties=CN_FONT)
save_fig(fig, os.path.join(FIG_DIR, 'p2_final_03_tanshi_analysis.png'))
print('✓ 保存图: p2_final_03_tanshi_analysis.png')

# =========================
# 15. 最终汇总
# =========================
print('\n' + '='*60)
print(f'  核心模型: {core_name} + Isotonic 校准')
print(f'  特征方法: {imp_name}')
print(f'  OOF AUC: {auc_mean:.4f} ± {auc_std:.4f}')
print(f'  OOF Brier Score: {brier_oof:.4f}')
print(f'  R阈值: R_LO={R_LO}, R_HI={R_HI}')
print(f'  概率阈值: p_low={p_low}, p_high={p_high}')
print(f'  三层样本分布: {dict(df["风险分层"].value_counts())}')
print('='*60)