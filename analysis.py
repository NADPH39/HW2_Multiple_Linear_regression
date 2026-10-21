"""
analysis.py — 離線產生報告用的所有數據與圖表
執行：python analysis.py
輸出：figures/*.png、results/*.json、results/*.csv
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import mlr_core as mc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)


def choose_k(kdf: pd.DataFrame, tol: float = 0.995) -> int:
    """選 k 的規則：CV R² 達到最佳值 99.5% 的最小 k（兼顧精簡與準確）。"""
    best = kdf.cv_R2.max()
    return int(kdf[kdf.cv_R2 >= tol * best].k.min())


def run(key: str):
    cfg = mc.PROBLEMS[key]
    print(f"\n==== {cfg['label']} ====")
    df = mc.load_csv(ROOT / cfg["path"])
    out = {"problem": cfg["label"], "kaggle": cfg["kaggle"], "shape": list(df.shape),
           "target": cfg["target"], "n_features_raw": df.shape[1] - 1,
           "missing_total": int(df.isna().sum().sum())}
    out["target_desc"] = df[cfg["target"]].describe().round(4).to_dict()
    mc.data_summary(df).to_csv(RES / f"{key}_data_summary.csv", index=False)

    prep = mc.prepare_data(df, cfg["target"])
    out["n_features_encoded"] = len(prep.encoded_features)
    out["n_train"], out["n_test"] = len(prep.X_train), len(prep.X_test)
    out["prep_notes"] = prep.notes
    save(mc.plot_correlation(prep.X_train, prep.y_train, top=20,
                             title=f"Correlation with {cfg['target']} (train)"), f"{key}_correlation.png")

    # ---- 特徵選擇方法比較（同一個 k）----
    p = prep.X_train.shape[1]
    k_cmp = 10 if p > 10 else max(1, p // 2)
    rows = []
    for m in mc.FS_METHODS:
        feats, _ = mc.select_features(prep.X_train, prep.y_train, m, k_cmp)
        lr = mc.lin_model().fit(prep.X_train[feats], prep.y_train)
        met = mc.regression_metrics(prep.y_test, lr.predict(prep.X_test[feats]), len(feats))
        cvm, cvs = mc.cv_r2(prep.X_train, prep.y_train, feats)
        rows.append({"method": m, "k": len(feats), **{k_: round(v, 5) for k_, v in met.items()},
                     "cv_R2": round(cvm, 5), "features": ", ".join(feats)})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(RES / f"{key}_fs_comparison.csv", index=False)
    out["fs_comparison_k"] = k_cmp
    print(cmp_df.drop(columns="features").to_string())

    # ---- k 曲線（SelectKBest）----
    method = "SelectKBest (F-test)"
    ks = list(range(1, p + 1)) if p <= 25 else list(range(1, 11)) + list(range(12, p + 1, 3)) + [p]
    kdf = mc.k_curve(prep, method, ks)
    kdf.to_csv(RES / f"{key}_k_curve.csv", index=False)
    k_best = choose_k(kdf)
    save(mc.plot_k_curve(kdf, title=f"{method}: choosing k", chosen_k=k_best), f"{key}_k_curve.png")
    feats_k, scores = mc.select_features(prep.X_train, prep.y_train, method, k_best)
    # Step 2：VIF 篩除多重共線性；Step 3：Backward elimination 移除不顯著特徵
    feats_v, drop_v = mc.vif_prune(prep.X_train, feats_k, 10)
    feats, drop_p = mc.backward_elimination(prep.X_train, prep.y_train, feats_v, 0.05)
    stages = []
    for name, fs in [("All features", prep.encoded_features), (f"SelectKBest k={k_best}", feats_k),
                     ("+ VIF<10 pruning", feats_v), ("+ Backward elimination p<0.05", feats)]:
        m = mc.lin_model().fit(prep.X_train[fs], prep.y_train)
        met = mc.regression_metrics(prep.y_test, m.predict(prep.X_test[fs]), len(fs))
        stages.append({"stage": name, "k": len(fs), **{k_: round(v, 4) for k_, v in met.items()},
                       "cv_R2": round(mc.cv_r2(prep.X_train, prep.y_train, fs)[0], 4),
                       "max_VIF": round(float(mc.vif(prep.X_train, fs).max()), 2)})
    pd.DataFrame(stages).to_csv(RES / f"{key}_selection_stages.csv", index=False)
    print(pd.DataFrame(stages).to_string())
    out["final_method"] = f"{method} → VIF<10 → Backward elimination"
    out["final_k"], out["final_features"] = len(feats), feats
    out["k_selectkbest"], out["vif_dropped"], out["pvalue_dropped"] = k_best, drop_v, drop_p

    # ---- 最終 OLS 模型 ----
    res = mc.fit_ols(prep.X_train, prep.y_train, feats)
    sk = mc.lin_model().fit(prep.X_train[feats], prep.y_train)
    assert np.allclose(sk.predict(prep.X_test[feats]), res.predict(prep.X_test).pred, atol=1e-6), "OLS 與 sklearn 預測不一致"
    pred = res.predict(prep.X_test)
    out["train"] = {"R2": res.r2, "Adj_R2": res.adj_r2, "F": res.f_stat, "F_p": res.f_p}
    out["test"] = mc.regression_metrics(prep.y_test, pred.pred, len(feats))
    out["cv_R2"] = mc.cv_r2(prep.X_train, prep.y_train, feats)
    out["interval"] = mc.interval_coverage(prep.y_test, pred)
    out["diagnostics"] = mc.residual_diagnostics(res, prep.X_train)
    out["vif_max"] = float(mc.vif(prep.X_train, feats).max())
    coef = res.coef_table()
    coef.round(6).to_csv(RES / f"{key}_coefficients.csv")
    mc.vif(prep.X_train, feats).round(3).to_csv(RES / f"{key}_vif.csv", header=["VIF"])

    # 全特徵基準 & Ridge/Lasso 對照
    allf = prep.encoded_features
    base = mc.regression_metrics(prep.y_test, mc.lin_model().fit(prep.X_train, prep.y_train)
                                 .predict(prep.X_test), len(allf))
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(prep.X_train, prep.y_train)
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42)).fit(prep.X_train, prep.y_train)
    out["baseline_all_features"] = base
    out["ridge_all"] = mc.regression_metrics(prep.y_test, ridge.predict(prep.X_test), len(allf))
    out["lasso_all"] = mc.regression_metrics(prep.y_test, lasso.predict(prep.X_test), len(allf))

    # ---- 圖 ----
    save(mc.plot_prediction_intervals(prep.y_test, pred, n_show=150,
                                      title=f"{cfg['target']}: test predictions with 95% CI & PI ({len(feats)} features)",
                                      ylabel=cfg["target"]),
         f"{key}_prediction_interval.png")
    save(mc.plot_actual_vs_pred(prep.y_test, pred), f"{key}_actual_vs_pred.png")
    save(mc.plot_residuals(res), f"{key}_residuals.png")
    save(mc.plot_coefficients(res), f"{key}_coefficients.png")
    top = scores.index[0]
    fig, sres = mc.plot_simple_regression(prep.X_train[top], prep.y_train, top,
                                          title=f"Simple LR: {cfg['target']} ~ {top} (95% CI & PI)",
                                          ylabel=cfg["target"])
    save(fig, f"{key}_simple_lr.png")
    out["simple_lr"] = {"feature": top, "R2_train": sres.r2,
                        "test": mc.regression_metrics(prep.y_test, sres.predict(prep.X_test[[top]]).pred, 1)}

    with open(RES / f"{key}_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k_: out[k_] for k_ in ["final_k", "test", "cv_R2", "interval", "diagnostics", "vif_max",
                                             "baseline_all_features", "simple_lr"]},
                     ensure_ascii=False, indent=1, default=float))
    return out


if __name__ == "__main__":
    for key in mc.PROBLEMS:
        run(key)
