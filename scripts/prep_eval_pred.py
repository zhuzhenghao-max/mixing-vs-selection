"""把上传格式预测补上真值里的对照细胞, 变成 cell-eval2 打分格式

模拟平台行为: 上传文件不含 non-targeting, 打分时平台从 reference 里
取留出的对照拼到预测上 (control_source: real)。
"""

import argparse

import anndata as ad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--real", required=True, help="含 non-targeting 的真值 h5ad")
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--control", default="non-targeting")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pred = ad.read_h5ad(args.pred)
    real = ad.read_h5ad(args.real, backed="r")
    ctrl = real[real.obs[args.pert_col].astype(str) == args.control].to_memory()
    ctrl.obs = ctrl.obs[[c for c in pred.obs.columns if c in ctrl.obs.columns]]

    out = ad.concat([pred, ctrl], join="outer")
    out.write_h5ad(args.out)
    print(f"pred {pred.n_obs} + controls {ctrl.n_obs} -> {args.out}")


if __name__ == "__main__":
    main()
