"""
mlr_core.py — 多元線性回歸 (Multiple Linear Regression) 核心函式
=================================================================
HW2：以 CRISP-DM 流程完成多元線性回歸分析。
這個模組只依賴 numpy / pandas / scipy / scikit-learn / matplotlib，
Streamlit 網頁 (app.py) 與離線分析腳本 (analysis.py) 共用同一套程式。

主要功能
--------
1. 讀取資料：本機 CSV、任意 URL（自動判斷分隔符號）
2. 資料準備：缺值補齊、類別變數 One-Hot、移除常數欄位、train/test 切分
3. 特徵選擇：Correlation、SelectKBest(F-test)、RFE、Lasso、Sequential Forward
4. 建模：以 OLS 正規方程式自行計算係數、標準誤、t、p 值（與 statsmodels 結果相同）
5. 評估：MAE / MSE / RMSE / R² / Adjusted R² / 5-fold CV、VIF、殘差診斷
6. 區間：平均反應的「信賴區間 (CI)」與新觀測值的「預測區間 (PI)」
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def lin_model():
    """標準化 + LinearRegression。
    特徵尺度差異極大時（例如 Population ~1e9），未標準化的最小平方解會出現數值不穩定，
    所以所有 sklearn 線性模型都先做 StandardScaler（不影響 OLS 的預測結果）。"""
    return make_pipeline(StandardScaler(), LinearRegression())

# ----------------------------------------------------------------------------
# 內建的兩個 Problem（下拉選單用）
# ----------------------------------------------------------------------------
PROBLEMS = {
    "life": {
        "label": "Problem 1：國家平均壽命預測 (WHO Life Expectancy, 21 features)",
        "path": "data/life_expectancy.csv",
        "target": "Life expectancy",
        "default_k": 15,
        "kaggle": "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who",
        "business": (
            "衛生政策單位想知道：在成人死亡率、疫苗接種率、HIV/AIDS、教育年數、GDP、醫療支出等 "
            "21 個因素中，哪些最能解釋一個國家的平均壽命 (Life expectancy, 歲)，"
            "並用模型估計某國在給定條件下的平均壽命與其不確定範圍。"
        ),
        "drop": [],
    },
    "student": {
        "label": "Problem 2：學生數學期末成績預測 (Student Performance, 32 features)",
        "path": "data/student-mat.csv",
        "target": "G3",
        "default_k": 2,
        "kaggle": "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption",
        "business": (
            "學校希望及早找出期末數學成績 G3（0–20 分）可能不理想的學生。資料包含家庭背景、"
            "讀書時間、缺課次數、飲酒習慣以及第一、二次段考成績 G1、G2 共 32 個特徵。"
        ),
        "drop": [],
    },
}

FS_METHODS = [
    "All features (不做選擇)",
    "Correlation |r| top-k",
    "SelectKBest (F-test)",
    "RFE (Recursive Feature Elimination)",
    "Lasso (L1, LassoCV)",
    "Sequential Forward Selection",
    "Backward Elimination (p < 0.05)",
]


# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def load_csv(source) -> pd.DataFrame:
    """讀取 CSV。source 可以是路徑、URL 或上傳的檔案物件；自動判斷 , ; \\t 分隔。"""
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        source = io.StringIO(raw)
    df = pd.read_csv(source, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    # 把看起來是數字的文字欄轉成數字（例如 "5" → 5）
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.95:
                df[c] = conv
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Data Understanding 用的欄位摘要表。"""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "mean": round(float(s.mean()), 4) if pd.api.types.is_numeric_dtype(s) else None,
            "std": round(float(s.std()), 4) if pd.api.types.is_numeric_dtype(s) else None,
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Data preparation
# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target: str
    original_features: list[str]
    encoded_features: list[str]
    notes: list[str] = field(default_factory=list)


def prepare_data(df: pd.DataFrame, target: str, features: list[str] | None = None,
                 test_size: float = 0.2, random_state: int = RANDOM_STATE) -> Prepared:
    notes = []
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = pd.to_numeric(df[target], errors="coerce")
    n0 = len(df)
    df = df[df[target].notna()]
    if len(df) < n0:
        notes.append(f"移除 {n0 - len(df)} 筆 target 缺值的資料")
    if features is None:
        features = [c for c in df.columns if c != target]
    # 移除看起來像 ID 的欄位（每列唯一的整數或字串）
    id_like = [c for c in features if df[c].nunique() == len(df) and
               (not pd.api.types.is_float_dtype(df[c]))]
    if id_like:
        notes.append(f"移除疑似 ID 欄位：{', '.join(id_like)}")
        features = [c for c in features if c not in id_like]
    X = df[features].copy()
    y = df[target].astype(float)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # 類別欄位太多種類（>30）就捨棄，避免 One-Hot 爆炸
    high_card = [c for c in cat_cols if X[c].nunique() > 30]
    if high_card:
        notes.append(f"類別數 >30 而捨棄：{', '.join(high_card)}")
        cat_cols = [c for c in cat_cols if c not in high_card]
        X = X.drop(columns=high_card)

    # 先切分再補值，避免 data leakage
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_tr, X_te = X_tr.copy(), X_te.copy()
    miss = int(X_tr.isna().sum().sum() + X_te.isna().sum().sum())
    for c in num_cols:
        med = X_tr[c].median()
        X_tr[c] = X_tr[c].fillna(med)
        X_te[c] = X_te[c].fillna(med)
    for c in cat_cols:
        mode = X_tr[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "missing"
        X_tr[c] = X_tr[c].fillna(fill).astype(str)
        X_te[c] = X_te[c].fillna(fill).astype(str)
    if miss:
        notes.append(f"以訓練集中位數/眾數補齊 {miss} 個缺值")

    if cat_cols:
        X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True, dtype=float)
        X_te = pd.get_dummies(X_te, columns=cat_cols, drop_first=True, dtype=float)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
        notes.append(f"One-Hot (drop_first) 編碼 {len(cat_cols)} 個類別欄位")
    X_tr = X_tr.astype(float)
    X_te = X_te.astype(float)
    const = [c for c in X_tr.columns if X_tr[c].std() == 0]
    if const:
        X_tr, X_te = X_tr.drop(columns=const), X_te.drop(columns=const)
        notes.append(f"移除常數欄位：{', '.join(const)}")
    return Prepared(X_tr, X_te, y_tr, y_te, target, list(X.columns), X_tr.columns.tolist(), notes)


# ----------------------------------------------------------------------------
# 3. Feature selection（只用訓練集，避免 leakage）
# ----------------------------------------------------------------------------
def select_features(X: pd.DataFrame, y: pd.Series, method: str, k: int) -> tuple[list[str], pd.Series]:
    """回傳 (被選中的特徵, 每個特徵的分數)。分數越大越重要。"""
    k = int(max(1, min(k, X.shape[1])))
    if X.shape[1] == 1:
        return X.columns.tolist(), pd.Series([1.0], index=X.columns)
    Xs =pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    if method.startswith("All"):
        score = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]))
        return X.columns.tolist(), score.sort_values(ascending=False)
    if method.startswith("Correlation"):
        score = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1])).sort_values(ascending=False)
        return score.index[:k].tolist(), score
    if method.startswith("SelectKBest"):
        sel = SelectKBest(f_regression, k=k).fit(X, y)
        score = pd.Series(sel.scores_, index=X.columns).sort_values(ascending=False)
        return X.columns[sel.get_support()].tolist(), score
    if method.startswith("RFE"):
        rfe = RFE(LinearRegression(), n_features_to_select=k).fit(Xs, y)
        score = pd.Series(-rfe.ranking_.astype(float), index=X.columns).sort_values(ascending=False)
        return X.columns[rfe.support_].tolist(), score
    if method.startswith("Lasso"):
        las = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000).fit(Xs, y)
        score = pd.Series(np.abs(las.coef_), index=X.columns).sort_values(ascending=False)
        nz = score[score > 1e-10]
        chosen = nz.index[:k].tolist() if len(nz) else score.index[:k].tolist()
        return chosen, score
    if method.startswith("Sequential"):
        Xsub, ysub = Xs, y
        if len(Xs) > 5000:  # 大資料抽樣加速
            idx = Xs.sample(5000, random_state=RANDOM_STATE).index
            Xsub, ysub = Xs.loc[idx], y.loc[idx]
        if k >= X.shape[1]:
            return X.columns.tolist(), X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]))
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k,
                                        direction="forward", cv=5, scoring="r2").fit(Xsub, ysub)
        chosen = X.columns[sfs.get_support()].tolist()
        score = pd.Series([1.0 if c in chosen else 0.0 for c in X.columns], index=X.columns)
        return chosen, score.sort_values(ascending=False)
    if method.startswith("Backward"):
        # 不使用 k：從全部特徵開始，逐一移除最不顯著者（p > 0.05）
        chosen, _ = backward_elimination(X, y, X.columns.tolist())
        res = fit_ols(X, y, chosen)
        score = pd.Series(np.abs(res.t[1:]), index=chosen).reindex(X.columns).fillna(0)
        return chosen, score.sort_values(ascending=False)
    raise ValueError(f"unknown method: {method}")


# ----------------------------------------------------------------------------
# 4. OLS（正規方程式，含統計推論與 CI / PI）
# ----------------------------------------------------------------------------
@dataclass
class OLSResult:
    features: list[str]
    beta: np.ndarray            # [intercept, b1, ..., bp]
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    sigma2: float               # 殘差變異數估計 s²
    XtX_inv: np.ndarray        # (ZᵀZ)⁻¹，Z 為標準化後含截距的設計矩陣
    df_resid: int
    n: int
    r2: float
    adj_r2: float
    f_stat: float
    f_p: float
    resid: np.ndarray
    fitted: np.ndarray
    mu: np.ndarray = None       # 標準化用的平均
    sd: np.ndarray = None       # 標準化用的標準差
    gamma: np.ndarray = None    # 標準化尺度下的係數

    def coef_table(self, alpha: float = 0.05) -> pd.DataFrame:
        tcrit = stats.t.ppf(1 - alpha / 2, self.df_resid)
        names = ["Intercept"] + self.features
        return pd.DataFrame({
            "coef": self.beta, "std_err": self.se, "t": self.t, "p_value": self.p,
            f"CI_low({1-alpha:.0%})": self.beta - tcrit * self.se,
            f"CI_high({1-alpha:.0%})": self.beta + tcrit * self.se,
        }, index=names)

    def predict(self, X: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """回傳預測值、平均反應信賴區間 (CI)、個別觀測預測區間 (PI)。"""
        Xo = X[self.features].to_numpy(dtype=float)
        Xm = np.column_stack([np.ones(len(Xo)), (Xo - self.mu) / self.sd])
        yhat = Xm @ self.gamma
        h = np.einsum("ij,jk,ik->i", Xm, self.XtX_inv, Xm)  # x0ᵀ(XᵀX)⁻¹x0
        tcrit = stats.t.ppf(1 - alpha / 2, self.df_resid)
        se_mean = np.sqrt(self.sigma2 * h)
        se_pred = np.sqrt(self.sigma2 * (1 + h))
        return pd.DataFrame({
            "pred": yhat,
            "ci_low": yhat - tcrit * se_mean, "ci_high": yhat + tcrit * se_mean,
            "pi_low": yhat - tcrit * se_pred, "pi_high": yhat + tcrit * se_pred,
        }, index=X.index)


def fit_ols(X: pd.DataFrame, y: pd.Series, features: list[str]) -> OLSResult:
    """以「標準化後的設計矩陣」做最小平方 (lstsq)，再轉換回原始單位的係數與標準誤。
    這樣可避免 XᵀX 條件數過大造成的數值誤差，結果與教科書 OLS 公式一致。"""
    Xo = X[features].to_numpy(dtype=float)
    mu = Xo.mean(axis=0)
    sd = Xo.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(Xo)), (Xo - mu) / sd])
    yv = y.to_numpy(dtype=float)
    n, p1 = Z.shape
    gamma, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    fitted = Z @ gamma
    resid = yv - fitted
    df_resid = n - p1
    sse = float(resid @ resid)
    sst = float(((yv - yv.mean()) ** 2).sum())
    sigma2 = sse / df_resid
    # 轉回原始單位：beta = A @ gamma，Cov(beta) = A Cov(gamma) Aᵀ
    A = np.zeros((p1, p1))
    A[0, 0] = 1.0
    A[0, 1:] = -mu / sd
    A[1:, 1:] = np.diag(1 / sd)
    beta = A @ gamma
    cov_beta = A @ (sigma2 * ZtZ_inv) @ A.T
    se = np.sqrt(np.clip(np.diag(cov_beta), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df_resid)
    r2 = 1 - sse / sst
    k = p1 - 1
    adj = 1 - (1 - r2) * (n - 1) / df_resid
    f_stat = ((sst - sse) / k) / sigma2 if k > 0 else np.nan
    f_p = stats.f.sf(f_stat, k, df_resid) if k > 0 else np.nan
    return OLSResult(list(features), beta, se, t, p, sigma2, ZtZ_inv, df_resid, n,
                     r2, adj, f_stat, f_p, resid, fitted, mu, sd, gamma)


# ----------------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, n_features: int) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2, "Adj_R2": adj}


def cv_r2(X: pd.DataFrame, y: pd.Series, features: list[str], folds: int = 5) -> tuple[float, float]:
    kf = KFold(folds, shuffle=True, random_state=RANDOM_STATE)
    s = cross_val_score(lin_model(), X[features], y, cv=kf, scoring="r2")
    return float(s.mean()), float(s.std())


def interval_coverage(y_true, pred_df: pd.DataFrame) -> dict:
    y = np.asarray(y_true, float)
    return {
        "PI_coverage": float(np.mean((y >= pred_df.pi_low) & (y <= pred_df.pi_high))),
        "PI_avg_width": float((pred_df.pi_high - pred_df.pi_low).mean()),
        "CI_avg_width": float((pred_df.ci_high - pred_df.ci_low).mean()),
    }


def vif(X: pd.DataFrame, features: list[str]) -> pd.Series:
    """Variance Inflation Factor：VIF_j = 1 / (1 - R²_j)。"""
    out = {}
    if len(features) < 2:
        return pd.Series({f: 1.0 for f in features})
    for f in features:
        others = [c for c in features if c != f]
        r2 = lin_model().fit(X[others], X[f]).score(X[others], X[f])
        out[f] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out).sort_values(ascending=False)


def vif_prune(X: pd.DataFrame, features: list[str], threshold: float = 10.0) -> tuple[list[str], list[str]]:
    """反覆移除 VIF 最高且 > threshold 的特徵，處理多重共線性。回傳 (保留, 移除)。"""
    feats, dropped = list(features), []
    while len(feats) > 1:
        v = vif(X, feats)
        if v.iloc[0] <= threshold:
            break
        dropped.append(v.index[0])
        feats.remove(v.index[0])
    return feats, dropped


def backward_elimination(X: pd.DataFrame, y: pd.Series, features: list[str],
                         alpha: float = 0.05) -> tuple[list[str], list[str]]:
    """Backward elimination：每次移除 p-value 最大且 > alpha 的特徵，直到全部顯著。"""
    feats, dropped = list(features), []
    while len(feats) > 1:
        res = fit_ols(X, y, feats)
        pv = pd.Series(res.p[1:], index=feats)
        worst = pv.idxmax()
        if pv[worst] <= alpha:
            break
        dropped.append(worst)
        feats.remove(worst)
    return feats, dropped


def residual_diagnostics(res: OLSResult, X: pd.DataFrame) -> dict:
    """線性回歸假設檢驗：常態性 (Jarque-Bera)、同質變異 (Breusch-Pagan)、獨立性 (Durbin-Watson)。"""
    e = res.resid
    jb_stat, jb_p = stats.jarque_bera(e)
    # Breusch-Pagan：e² 對 X 回歸，LM = n·R²
    Xm = X[res.features].to_numpy(float)
    e2 = e ** 2
    lr = lin_model().fit(Xm, e2)
    r2_aux = lr.score(Xm, e2)
    bp_lm = len(e) * r2_aux
    bp_p = stats.chi2.sf(bp_lm, len(res.features))
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
    return {"Jarque_Bera": float(jb_stat), "JB_p": float(jb_p), "skew": float(stats.skew(e)),
            "kurtosis": float(stats.kurtosis(e, fisher=False)),
            "Breusch_Pagan_LM": float(bp_lm), "BP_p": float(bp_p), "Durbin_Watson": dw}


def k_curve(prep: Prepared, method: str, ks: list[int]) -> pd.DataFrame:
    """不同特徵數 k 下的測試集表現，用來決定 k。"""
    rows = []
    for k in ks:
        feats, _ = select_features(prep.X_train, prep.y_train, method, k)
        m = lin_model().fit(prep.X_train[feats], prep.y_train)
        met = regression_metrics(prep.y_test, m.predict(prep.X_test[feats]), len(feats))
        cvm, _ = cv_r2(prep.X_train, prep.y_train, feats)
        rows.append({"k": len(feats), "test_RMSE": met["RMSE"], "test_R2": met["R2"],
                     "test_Adj_R2": met["Adj_R2"], "cv_R2": cvm})
    return pd.DataFrame(rows).drop_duplicates("k")


# ----------------------------------------------------------------------------
# 6. Plots（matplotlib，網頁與報告共用）
# ----------------------------------------------------------------------------
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_prediction_intervals(y_true, pred_df: pd.DataFrame, n_show: int = 200, title: str = "",
                              ylabel: str = "target"):
    """主預測圖：依預測值排序，畫出實際值、預測線、95% CI 與 95% PI。"""
    import matplotlib.pyplot as plt
    d = pred_df.copy()
    d["actual"] = np.asarray(y_true, float)
    if len(d) > n_show:
        d = d.sample(n_show, random_state=RANDOM_STATE)
    d = d.sort_values("pred").reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(x, d.pi_low, d.pi_high, color="#9ec5e8", alpha=0.45, label="95% Prediction Interval")
    ax.fill_between(x, d.ci_low, d.ci_high, color="#2b6cb0", alpha=0.55, label="95% Confidence Interval")
    ax.plot(x, d.pred, color="#1a365d", lw=1.6, label="Predicted")
    inside = (d.actual >= d.pi_low) & (d.actual <= d.pi_high)
    ax.scatter(x[inside], d.actual[inside], s=12, color="#dd6b20", label="Actual (inside PI)", zorder=3)
    ax.scatter(x[~inside], d.actual[~inside], s=22, color="#c53030", marker="x", label="Actual (outside PI)", zorder=3)
    _style(ax, title or "Test-set predictions with 95% CI / PI", "test samples (sorted by prediction)", ylabel)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_actual_vs_pred(y_true, pred_df: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt
    y = np.asarray(y_true, float)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y, pred_df.pred, s=8, alpha=0.4, color="#2b6cb0")
    lo, hi = min(y.min(), pred_df.pred.min()), max(y.max(), pred_df.pred.max())
    ax.plot([lo, hi], [lo, hi], color="#c53030", lw=1.2, ls="--", label="y = ŷ")
    _style(ax, title or "Actual vs Predicted (test)", "actual", "predicted")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_simple_regression(x, y, feature: str, alpha: float = 0.05, title: str = "", ylabel: str = "target"):
    """單一特徵的簡單線性回歸：回歸線 + CI 帶 + PI 帶。"""
    import matplotlib.pyplot as plt
    x, y = np.asarray(x, float), np.asarray(y, float)
    df_ = pd.DataFrame({feature: x})
    res = fit_ols(df_, pd.Series(y), [feature])
    grid = pd.DataFrame({feature: np.linspace(x.min(), x.max(), 100)})
    p = res.predict(grid, alpha)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    jitter = np.random.default_rng(0).normal(0, 0.08, len(x)) if len(np.unique(x)) < 30 else 0
    idx = np.random.default_rng(1).choice(len(x), min(len(x), 3000), replace=False)
    ax.scatter(x[idx] + (jitter[idx] if np.ndim(jitter) else 0), y[idx], s=6, alpha=0.25, color="#718096")
    ax.fill_between(grid[feature], p.pi_low, p.pi_high, color="#9ec5e8", alpha=0.45, label=f"{1-alpha:.0%} PI")
    ax.fill_between(grid[feature], p.ci_low, p.ci_high, color="#2b6cb0", alpha=0.6, label=f"{1-alpha:.0%} CI")
    ax.plot(grid[feature], p.pred, color="#1a365d", lw=2,
            label=f"ŷ = {res.beta[0]:.4f} + {res.beta[1]:.4f}·x  (R²={res.r2:.3f})")
    _style(ax, title or f"Simple linear regression on {feature}", feature, ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, res


def plot_residuals(res: OLSResult, title_prefix: str = ""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    idx = np.random.default_rng(0).choice(len(res.resid), min(len(res.resid), 4000), replace=False)
    axes[0].scatter(res.fitted[idx], res.resid[idx], s=6, alpha=0.35, color="#2b6cb0")
    axes[0].axhline(0, color="#c53030", lw=1)
    _style(axes[0], f"{title_prefix}Residuals vs Fitted", "fitted", "residual")
    axes[1].hist(res.resid, bins=50, color="#2b6cb0", alpha=0.8)
    _style(axes[1], "Residual distribution", "residual", "count")
    (osm, osr), (slope, inter, _) = stats.probplot(res.resid, dist="norm")
    sub = np.linspace(0, len(osm) - 1, min(len(osm), 2000)).astype(int)
    axes[2].scatter(osm[sub], osr[sub], s=6, color="#2b6cb0")
    axes[2].plot(osm, slope * osm + inter, color="#c53030", lw=1)
    _style(axes[2], "Normal Q-Q", "theoretical quantiles", "sample quantiles")
    fig.tight_layout()
    return fig


def plot_k_curve(kdf: pd.DataFrame, title: str = "", chosen_k: int | None = None):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(kdf.k, kdf.test_Adj_R2, marker="o", color="#2b6cb0", label="test Adjusted R²")
    ax1.plot(kdf.k, kdf.cv_R2, marker="s", color="#38a169", label="5-fold CV R² (train)")
    _style(ax1, title or "Feature count k vs performance", "number of features k", "R²")
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if chosen_k is not None:
        ax1.axvline(chosen_k, color="#718096", ls=":", lw=1.2)
        ax1.annotate(f"chosen k={chosen_k}", (chosen_k, ax1.get_ylim()[0]), xytext=(4, 8),
                     textcoords="offset points", fontsize=8, color="#4a5568")
    ax2 = ax1.twinx()
    ax2.plot(kdf.k, kdf.test_RMSE, marker="^", color="#dd6b20", label="test RMSE")
    ax2.set_ylabel("RMSE")
    ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


def plot_correlation(X: pd.DataFrame, y: pd.Series, top: int = 20, title: str = ""):
    import matplotlib.pyplot as plt
    r = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).sort_values()
    r = r.reindex(r.abs().sort_values().index[-top:])
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(r) + 1)))
    ax.barh(r.index, r.values, color=["#c53030" if v < 0 else "#2b6cb0" for v in r.values])
    ax.axvline(0, color="#4a5568", lw=0.8)
    _style(ax, title or "Pearson correlation with target", "r", "")
    fig.tight_layout()
    return fig


def plot_coefficients(res: OLSResult, title: str = ""):
    """標準化係數（特徵增加 1 個標準差時 target 的變化）與 95% CI，方便跨特徵比較影響力。"""
    import matplotlib.pyplot as plt
    se_g = np.sqrt(res.sigma2 * np.diag(res.XtX_inv))[1:]
    tcrit = stats.t.ppf(0.975, res.df_resid)
    tab = pd.DataFrame({"coef": res.gamma[1:], "err": tcrit * se_g}, index=res.features)
    tab = tab.reindex(tab.coef.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(tab) + 1)))
    ax.errorbar(tab.coef, tab.index, xerr=tab.err, fmt="o", color="#2b6cb0", ecolor="#2b6cb0",
                elinewidth=2, capsize=4)
    ax.axvline(0, color="#c53030", lw=0.8)
    _style(ax, title or "Standardized coefficients (per 1 SD) with 95% CI", "change in target per +1 SD", "")
    fig.tight_layout()
    return fig
