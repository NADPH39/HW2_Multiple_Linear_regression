"""
HW2 — Multiple Linear Regression × CRISP-DM（Streamlit 網頁）
執行：streamlit run app.py

加分項目
  (+1) 側邊欄下拉選單切換兩種 Problem（WHO 平均壽命 / 學生成績）
  (+2) 可輸入任意 Dataset 網址或上傳 CSV（例如 50_Startups.csv）
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import mlr_core as mc

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HW2 Multiple Linear Regression", page_icon="📈", layout="wide")

DEFAULT_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
CUSTOM = "自訂 Dataset（輸入網址 / 上傳 CSV）"


@st.cache_data(show_spinner="讀取資料中…")
def load_source(src: str) -> pd.DataFrame:
    return mc.load_csv(src)


@st.cache_data(show_spinner="讀取上傳檔案…")
def load_upload(content: bytes) -> pd.DataFrame:
    import io
    return mc.load_csv(io.BytesIO(content))


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# Sidebar：資料與參數
# ============================================================================
st.sidebar.title("⚙️ 設定")
labels = [cfg["label"] for cfg in mc.PROBLEMS.values()] + [CUSTOM]
choice = st.sidebar.selectbox("① 選擇 Problem", labels, index=0)

cfg = None
df = None
if choice == CUSTOM:
    url = st.sidebar.text_input("Dataset 網址（CSV）", DEFAULT_URL)
    up = st.sidebar.file_uploader("或上傳 CSV", type=["csv", "txt"])
    try:
        df = load_upload(up.getvalue()) if up is not None else load_source(url.strip())
    except Exception as e:  # noqa: BLE001
        st.error(f"無法讀取資料：{e}")
        st.stop()
    num_cols = mc.numeric_columns(df)
    if not num_cols:
        st.error("這份資料沒有數值欄位可以當作 target。")
        st.stop()
    target = st.sidebar.selectbox("Target（預測目標）", num_cols, index=len(num_cols) - 1)
    source_link = up.name if up is not None else url
    business = (f"使用者自訂資料集，目標是以其他欄位預測 **{target}**。"
                "請依資料的實際情境補充商業目標。")
else:
    key = [k for k, v in mc.PROBLEMS.items() if v["label"] == choice][0]
    cfg = mc.PROBLEMS[key]
    df = load_source(cfg["path"])
    target = cfg["target"]
    source_link = cfg["kaggle"]
    business = cfg["business"]

candidates = [c for c in df.columns if c != target]
features_in = st.sidebar.multiselect("② 使用的原始特徵（可移除）", candidates, default=candidates)
model_type = st.sidebar.radio("③ 模型類型", ["多元線性回歸 (Multiple LR)", "簡單線性回歸 (Simple LR)"])
fs_method = st.sidebar.selectbox("④ 特徵選擇方法", mc.FS_METHODS, index=2,
                                 disabled=model_type.startswith("簡單"))
use_vif = st.sidebar.checkbox("移除 VIF > 10 的共線性特徵", value=True,
                              disabled=model_type.startswith("簡單"))
conf = st.sidebar.slider("⑤ 信賴水準", 0.80, 0.99, 0.95, 0.01)
test_size = st.sidebar.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
alpha = 1 - conf

if not features_in:
    st.warning("請至少選擇一個特徵。")
    st.stop()

prep = mc.prepare_data(df, target, features_in, test_size=test_size)
if len(prep.X_train) < 10 or prep.X_train.shape[1] == 0:
    st.error("可用資料太少（訓練集 < 10 筆或沒有可用特徵）。")
    st.stop()
p_enc = prep.X_train.shape[1]
default_k = min(cfg.get("default_k", 10) if cfg else 10, p_enc)
if p_enc > 1:
    k = st.sidebar.slider("特徵數 k", 1, p_enc, default_k,
                          disabled=model_type.startswith("簡單") or fs_method.startswith(("All", "Backward")))
else:
    k = 1

# ---- 特徵選擇 + 建模 ----
if model_type.startswith("簡單"):
    corr = prep.X_train.apply(lambda c: abs(np.corrcoef(c, prep.y_train)[0, 1])).sort_values(ascending=False)
    one = st.sidebar.selectbox("簡單回歸使用的特徵", corr.index.tolist(), index=0)
    feats, scores, dropped_vif = [one], corr, []
else:
    feats, scores = mc.select_features(prep.X_train, prep.y_train, fs_method, k)
    dropped_vif = []
    if use_vif and len(feats) > 1:
        feats, dropped_vif = mc.vif_prune(prep.X_train, feats, 10)

res = mc.fit_ols(prep.X_train, prep.y_train, feats)
pred = res.predict(prep.X_test, alpha)
met = mc.regression_metrics(prep.y_test, pred.pred, len(feats))
cov = mc.interval_coverage(prep.y_test, pred)

# ============================================================================
# Header
# ============================================================================
st.title("📈 HW2：多元線性回歸 × CRISP-DM")
st.caption(f"目前資料：**{choice}**　｜　Target：**{target}**　｜　資料來源：{source_link}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("使用特徵數", len(feats))
c2.metric("Test R²", f"{met['R2']:.3f}")
c3.metric("Test Adj. R²", f"{met['Adj_R2']:.3f}")
c4.metric("Test RMSE", f"{met['RMSE']:.4g}")
c5.metric(f"{conf:.0%} PI 覆蓋率", f"{cov['PI_coverage']:.1%}")

tabs = st.tabs(["1 Business Understanding", "2 Data Understanding", "3 Data Preparation",
                "4 Modeling", "5 Evaluation", "6 Deployment"])

# ---- 1. Business Understanding ----
with tabs[0]:
    st.subheader("商業理解")
    st.markdown(business)
    st.markdown(f"""
- **問題類型**：連續數值預測（Regression），預測目標 `{target}`
- **成功標準**：測試集 R² 越高越好、RMSE 越小越好；預測區間覆蓋率應接近設定的 {conf:.0%}
- **為何選線性回歸**：係數可直接解釋「某因素增加 1 單位，目標變化多少」，並可計算信賴區間與預測區間
- **資料來源**：{source_link}
""")

# ---- 2. Data Understanding ----
with tabs[1]:
    st.subheader("資料理解")
    a, b, c = st.columns(3)
    a.metric("資料筆數", f"{len(df):,}")
    b.metric("原始特徵數", df.shape[1] - 1)
    c.metric("缺值總數", int(df.isna().sum().sum()))
    st.markdown("**前 10 筆資料**")
    st.dataframe(df.head(10))
    st.markdown("**欄位摘要**")
    st.dataframe(mc.data_summary(df))
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(df[target].dropna(), bins=40, color="#2b6cb0", alpha=0.85)
        mc._style(ax, f"Distribution of {target}", target, "count")
        fig.tight_layout()
        show(fig)
    with col2:
        show(mc.plot_correlation(prep.X_train, prep.y_train, top=20))

# ---- 3. Data Preparation ----
with tabs[2]:
    st.subheader("資料準備")
    st.markdown(f"""
1. 移除 target 缺值的資料列，自動移除疑似 ID 欄位
2. 依 {1-test_size:.0%} / {test_size:.0%} 切分 **訓練集 {len(prep.X_train):,} 筆 / 測試集 {len(prep.X_test):,} 筆**（random_state=42）
3. **先切分再補值**：數值欄位用訓練集中位數、類別欄位用眾數補齊，避免 data leakage
4. 類別欄位做 One-Hot 編碼（drop_first，避免虛擬變數陷阱）
5. 編碼後共有 **{p_enc}** 個候選特徵
""")
    if prep.notes:
        st.info("　\n".join(f"• {n}" for n in prep.notes))
    st.markdown("**編碼後的訓練資料（前 5 筆）**")
    st.dataframe(prep.X_train.head())

# ---- 4. Modeling ----
with tabs[3]:
    st.subheader("建模：特徵選擇 + OLS")
    if model_type.startswith("簡單"):
        st.markdown(f"簡單線性回歸：`{target} = b0 + b1 × {feats[0]}`")
        fig, sres = mc.plot_simple_regression(prep.X_train[feats[0]], prep.y_train, feats[0], alpha,
                                              ylabel=target)
        show(fig)
    else:
        st.markdown(f"特徵選擇方法：**{fs_method}**（只使用訓練集計算，避免偷看測試集）")
        if dropped_vif:
            st.warning(f"因多重共線性 (VIF > 10) 移除：{', '.join(dropped_vif)}")
        st.markdown(f"**最終使用 {len(feats)} 個特徵**：" + "、".join(f"`{f}`" for f in feats))
        sc = scores.head(30).rename("score").to_frame()
        sc["selected"] = sc.index.isin(feats)
        with st.expander("每個特徵的選擇分數"):
            st.dataframe(sc)
        with st.expander("比較所有特徵選擇方法（同一個 k）"):
            if st.button("執行比較"):
                rows = []
                for m in mc.FS_METHODS:
                    fs_, _ = mc.select_features(prep.X_train, prep.y_train, m, k)
                    mm = mc.lin_model().fit(prep.X_train[fs_], prep.y_train)
                    r_ = mc.regression_metrics(prep.y_test, mm.predict(prep.X_test[fs_]), len(fs_))
                    rows.append({"method": m, "k": len(fs_), "R2": r_["R2"], "Adj_R2": r_["Adj_R2"],
                                 "RMSE": r_["RMSE"], "CV_R2": mc.cv_r2(prep.X_train, prep.y_train, fs_)[0]})
                st.dataframe(pd.DataFrame(rows).round(4))
        with st.expander("k 曲線：特徵數與表現"):
            if st.button("畫 k 曲線"):
                ks = list(range(1, p_enc + 1)) if p_enc <= 25 else sorted(set(np.linspace(1, p_enc, 20).astype(int)))
                method_k = fs_method if not fs_method.startswith(("All", "Backward")) else "SelectKBest (F-test)"
                show(mc.plot_k_curve(mc.k_curve(prep, method_k, ks), title=f"{method_k}: choosing k",
                                     chosen_k=len(feats)))
    st.markdown(f"**OLS 係數表（{conf:.0%} 信賴區間）**　訓練集 R² = {res.r2:.4f}，Adj. R² = {res.adj_r2:.4f}，"
                f"F = {res.f_stat:.2f} (p = {res.f_p:.2e})")
    st.dataframe(res.coef_table(alpha).style.format("{:.5g}"))
    if len(feats) > 1:
        show(mc.plot_coefficients(res))

# ---- 5. Evaluation ----
with tabs[4]:
    st.subheader("評估")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("MAE", f"{met['MAE']:.4g}")
    e2.metric("MSE", f"{met['MSE']:.4g}")
    e3.metric("RMSE", f"{met['RMSE']:.4g}")
    e4.metric("R²", f"{met['R2']:.4f}")
    cvm, cvs = mc.cv_r2(prep.X_train, prep.y_train, feats)
    e5.metric("5-fold CV R²", f"{cvm:.4f}", f"± {cvs:.4f}", delta_color="off")

    st.markdown(f"#### 預測圖（{conf:.0%} 信賴區間 CI 與 預測區間 PI）")
    if len(pred) > 30:
        n_show = st.slider("圖上顯示的測試樣本數", 30, min(500, len(pred)), min(150, len(pred)))
    else:
        n_show = len(pred)
    show(mc.plot_prediction_intervals(prep.y_test, pred, n_show=n_show, ylabel=target,
                                      title=f"{target}: test predictions with {conf:.0%} CI & PI"))
    st.caption(f"深藍帶 = 平均反應的 {conf:.0%} 信賴區間（平均寬度 {cov['CI_avg_width']:.4g}）；"
               f"淺藍帶 = 單一新觀測的 {conf:.0%} 預測區間（平均寬度 {cov['PI_avg_width']:.4g}）。"
               f"測試集實際值落在 PI 內的比例為 {cov['PI_coverage']:.1%}。")
    col1, col2 = st.columns([1, 1.6])
    with col1:
        show(mc.plot_actual_vs_pred(prep.y_test, pred))
    with col2:
        diag = mc.residual_diagnostics(res, prep.X_train)
        v = mc.vif(prep.X_train, feats)
        st.markdown("**線性回歸假設檢驗（訓練集殘差）**")
        st.dataframe(pd.DataFrame([
            ["線性 / 獨立", "Durbin-Watson", f"{diag['Durbin_Watson']:.3f}", "≈2 表示殘差無自相關"],
            ["常態性", "Jarque-Bera p", f"{diag['JB_p']:.3g}", "p<0.05 → 殘差非常態（大樣本下常見）"],
            ["同質變異", "Breusch-Pagan p", f"{diag['BP_p']:.3g}", "p<0.05 → 有異質變異"],
            ["多重共線性", "最大 VIF", f"{v.max():.2f}", "<10 可接受"],
        ], columns=["假設", "檢定", "值", "判讀"]))
    show(mc.plot_residuals(res))

# ---- 6. Deployment ----
with tabs[5]:
    st.subheader("部署：輸入新資料即時預測")
    st.markdown("預設值為訓練集中位數，修改後即可得到預測值與區間。")
    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(feats):
        col = prep.X_train[f]
        is_dummy = set(col.unique()) <= {0.0, 1.0}
        with cols[i % 3]:
            if is_dummy:
                inputs[f] = float(st.selectbox(f, [0, 1], index=int(col.median()), key=f"in_{f}"))
            else:
                inputs[f] = st.number_input(f, value=float(col.median()), key=f"in_{f}",
                                            format="%.4g")
    new = pd.DataFrame([inputs])
    out = res.predict(new, alpha).iloc[0]
    d1, d2, d3 = st.columns(3)
    d1.metric(f"預測 {target}", f"{out.pred:.4g}")
    d2.metric(f"{conf:.0%} 信賴區間 (平均)", f"{out.ci_low:.4g} ~ {out.ci_high:.4g}")
    d3.metric(f"{conf:.0%} 預測區間 (個體)", f"{out.pi_low:.4g} ~ {out.pi_high:.4g}")

    terms = " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.4g}·[{f}]" for f, b in zip(feats, res.beta[1:]))
    st.markdown("**模型方程式**")
    st.code(f"{target} = {res.beta[0]:.4g} {terms}", language=None)

    out_df = prep.X_test[feats].copy()
    out_df[f"actual_{target}"] = prep.y_test
    out_df = pd.concat([out_df, pred], axis=1)
    st.download_button("下載測試集預測結果 (CSV)", out_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="predictions.csv", mime="text/csv")
    st.download_button("下載係數表 (CSV)", res.coef_table(alpha).to_csv().encode("utf-8-sig"),
                       file_name="coefficients.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("HW2 · Multiple Linear Regression · CRISP-DM\n\n本網頁程式由 Claude 輔助撰寫，詳見 README。")
