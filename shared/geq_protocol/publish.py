#!/usr/bin/env python3
"""Synchronize reviewable results and code to the explicitly requested branch."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

from study_paths import BACKBONES, FIGURES, KIT, OUTPUT, PROJECT, dump

BRANCH = "gmr-mr-detr-multiroute-20261030"
REMOTE = "https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval.git"
WORKTREE = Path(os.environ.get("GMR_PUBLISH_WORKTREE", "/tmp/gmr-multiroute-upload-20261031"))
TOKEN_FILE = Path(os.environ.get("GMR_GIT_TOKEN_FILE", "/home/guoxiangyu/token2git.txt"))


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def copy_tree(source, destination, extensions):
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix not in extensions:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def root_readme():
    path = WORKTREE/"README.md"
    old = path.read_text()
    guide = old[old.index("## 从哪里开始"):]
    guide = guide.replace("## 当前训练配置", "## 原六路线训练配置")
    summary = read(OUTPUT/"summary.json") or []
    requested = [s for s in summary if s["variant"] in ("shared","multi")]
    completed = sum(s["status"]=="COMPLETED" for s in requested)
    controls = [s for s in summary if s["variant"]=="baseline"]
    finished_controls = sum(s["status"]=="COMPLETED" for s in controls)
    lines = ["# GMR × Mr. DETR 多骨干实验", "",
        "四骨干的Temporal O2M、独立FFN、Instruction路线，以及Global Event Query存在性预测。", "",
        "## 最新：统一验证集选模协议", "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。Multi / Shared全量与冻结 **{completed}/16** 组完成Test；同源Baseline控制 **{finished_controls}/8** 组。", "",
        "所有新微调均使用seed2023；按完整Val的G-mIoU@1在0.4/0.5/0.6三点均值选best，连续超过5轮（6轮）不提升早停，上限20轮。Test只评估选定best。两种方案并行排队，每卡最多两个CUDA任务，包含已有作业。", "",
        "full保留原路线GMR/O2M损失并加GEQ BCE；frozen冻结原模型，仅训练GEQ。原GMR与Multi/Shared既有训练预算仍有差异，完整方法比较不能完全归因于单一模块。", "",
        "[全部24组状态、选模和最新Test](results/geq_val_selection/README.md) · [复现代码](shared/geq_protocol/README.md)", ""]
    for b in BACKBONES:
        lines += [f"### {b}", ""]
        available=False
        for m in ("full","frozen"):
            p=f"results/geq_val_selection/{b}/{m}_comparison.png"
            if (WORKTREE/p).is_file():
                lines += [f"**{m}：GMR / GMR+GEQ / Multi+GEQ / Shared+GEQ**", "",f"![{b} {m}]({p})", ""]
                available=True
        if not available:
            for v in ("shared","multi"):
                p=f"results/geq_val_selection/{b}/{v}_full_frozen.png"
                if (WORKTREE/p).is_file():
                    lines += [f"![{b} {v} full/frozen]({p})", ""]
                    available=True
        if not available:
            lines += ["新协议训练进行中；完成相应Test后自动展示曲线。", ""]
    lines += ["## 已完成：main E1同源初始化的八组GEQ对照", "",
        "这组使用main E1的历史GMR检查点，与上面的当前多路线起点不同，因此单列。八组均已完成完整Val/Test；下图直接展示G-mIoU@1、@3的101点阈值曲线、拒绝/保留关系、Rej-F1与固定工作点，图例报告mAP。", "",
        "| Backbone | GMR mAP (%) | full + GEQ | frozen + GEQ |",
        "|---|---:|---:|---:|"]
    main_root=PROJECT/"geq_epoch_study/20261031"
    for b in BACKBONES:
        values=[]
        for mode, filename in (("full","baseline_test_metrics.json"),("full","test_metrics.json"),("frozen","test_metrics.json")):
            values.append(read(main_root/b/mode/filename).get("0.4",{}).get("brief",{}).get("mAP"))
        lines.append("| "+b+" | "+" | ".join("—" if v is None else f"{v:.2f}" for v in values)+" |")
    lines += ["", "[完整指标、Val曲线、PDF/SVG和CSV](results/geq_epoch_study/README.md)", ""]
    for b in BACKBONES:
        lines += [f"### {b} · main E1对照", "", f"![{b} full/frozen GEQ](results/geq_epoch_study/{b}/full_frozen_comparison.png)", ""]
    lines += ["## 历史结果", "",
        "- [固定四轮GEQ的12组结果与四曲线](results/geq_finetune/README.md)：旧协议，保留供追溯。",
        "- [原六路线24组汇总快照](results/README.md)：按原Val mAP选模，与本次GEQ选模分开。",
        "- G-mIoU及存在性图使用0–1；mAP、AUROC使用百分数。所有图表按骨干分文件夹。", "", guide]
    path.write_text("\n".join(lines))


def synchronize():
    copy_tree(KIT, WORKTREE/"shared/geq_protocol", {".py",".sh",".md"})
    copy_tree(PROJECT/"gmr_geq_epoch_study_kit", WORKTREE/"shared/geq_epoch_study", {".py",".sh",".md"})
    # Published files use server defaults or explicit environment overrides.
    for directory in ("geq_protocol", "geq_epoch_study"):
        p=WORKTREE/"shared"/directory/"study_paths.py"
        s=p.read_text()
        s=s.replace('PROJECT = KIT.parent', 'PROJECT = Path(os.environ.get("GMR_PROJECT_ROOT", "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval"))')
        s=s.replace('str(KIT.parent)', '"/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval"')
        if "import os" not in s:
            s="import os\n"+s
        p.write_text(s)
    copy_tree(FIGURES,WORKTREE/"results/geq_val_selection",{".png",".pdf",".svg",".csv",".json",".md"})
    source=PROJECT/"newidea/figure/gmr_mr_detr_test_20261030/geq_finetune/epoch_study"
    copy_tree(source,WORKTREE/"results/geq_epoch_study",{".png",".pdf",".svg",".csv",".json",".md"})
    metadata={"CONFIG.json","PREFLIGHT.json","STATUS.json","COMPLETE.json","FAILURE.json","history.jsonl",
        "best_val_metrics.json","baseline_val_metrics.json","baseline_test_metrics.json","test_metrics.json","QUEUE.json","summary.csv","summary.json"}
    for root,target in ((OUTPUT,WORKTREE/"results/geq_val_selection/runs"),
                       (PROJECT/"geq_epoch_study/20261031",WORKTREE/"results/geq_epoch_study/runs")):
        for p in root.rglob("*"):
            if p.is_file() and p.name in metadata:
                dest=target/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    root_readme()


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/"publish.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not WORKTREE.is_dir():
            raise RuntimeError("Dedicated publishing worktree is missing")
        raw=TOKEN_FILE.read_text().strip()
        match=re.search(r"(?:github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+)",raw)
        secret=match.group(0) if match else raw
        askpass=Path(tempfile.gettempdir())/"gmr_geq_git_askpass.py"
        askpass.write_text("#!/usr/bin/env python3\nimport os,pathlib,re,sys\ns=pathlib.Path(os.environ['GMR_GIT_TOKEN_FILE']).read_text().strip()\nm=re.search(r'(?:github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+)',s)\nprint('x-access-token' if 'username' in sys.argv[1].lower() else (m.group(0) if m else s))\n")
        askpass.chmod(0o700)
        env=dict(os.environ,GIT_ASKPASS=str(askpass),GIT_TERMINAL_PROMPT="0",GMR_GIT_TOKEN_FILE=str(TOKEN_FILE))
        def git(*args, check=True):
            result=subprocess.run(["git","-C",str(WORKTREE),*args],text=True,env=env,capture_output=True)
            if check and result.returncode:
                raise RuntimeError((result.stdout+result.stderr).replace(secret,"<redacted>")[-4000:])
            return result
        branch=git("branch","--show-current").stdout.strip()
        if branch!=BRANCH:
            raise RuntimeError("Publishing worktree is on the wrong branch")
        git("fetch","--quiet",REMOTE,BRANCH)
        ahead=git("rev-list","--count","HEAD..FETCH_HEAD").stdout.strip()
        if ahead!="0":
            if git("status","--porcelain").stdout.strip():
                raise RuntimeError("Remote advanced while publishing worktree has changes")
            git("merge","--ff-only","FETCH_HEAD")
        with (OUTPUT/"report.lock").open("a") as report_lock:
            fcntl.flock(report_lock,fcntl.LOCK_EX)
            synchronize()
        git("add","README.md","shared/geq_protocol","shared/geq_epoch_study","results/geq_val_selection","results/geq_epoch_study")
        if git("diff","--cached","--quiet",check=False).returncode:
            git("commit","-m","Update Val-selected GEQ studies, curves and parallel training queue")
        git("push","--quiet",REMOTE,"HEAD:refs/heads/"+BRANCH)
        commit=git("rev-parse","HEAD").stdout.strip()
        dump(OUTPUT/"PUBLISH_STATUS.json",dict(status="PUBLISHED",commit=commit,branch=BRANCH,updated_at=datetime.now().astimezone().isoformat()))
        print("GITHUB_PUBLISHED",commit,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        dump(OUTPUT/"PUBLISH_STATUS.json",dict(status="FAILED",error=str(error),updated_at=datetime.now().astimezone().isoformat()))
        raise
