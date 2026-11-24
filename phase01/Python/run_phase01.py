"""在 Project1124 建立獨立執行目錄，依序完成第一階段。"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from image_pipeline import CODE, CONFIG, sha, json_file

PROJECT_FOLDERS = ("01_研究規格","02_參考文獻","03_程式碼","04_原始影像",
                   "05_實驗一_JPEG矩陣擾動","06_實驗二_奇異值嵌入",
                   "07_實驗三_奇異向量嵌入","08_書面報告","09_研究紀錄")
RUN_FOLDERS = ("01_JPEG影像","02_原始矩陣","03_SVD分解結果","04_逐區塊擾動數據",
               "05_統計分析","06_MATLAB圖表","07_Python驗證","08_交叉驗證",
               "09_實驗報告","10_執行紀錄")


def run_logged(argv, log_path):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps([str(x) for x in argv], ensure_ascii=False)+"\n")
        with subprocess.Popen([str(x) for x in argv], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                try:
                    text = line.decode("utf-8")
                except UnicodeDecodeError:
                    text = line.decode("mbcs" if os.name == "nt" else "utf-8", errors="replace")
                log.write(text)
                log.flush()
                print(text.rstrip(), flush=True)
            status = process.wait()
        log.write(f"Exit code: {status}\n")
    if status:
        raise subprocess.CalledProcessError(status, argv)


def quote(path):
    return "'" + str(path).replace("'","''") + "'"


def launch(base, upstream, matlab, pilot):
    base = base.resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(base)
    for name in PROJECT_FOLDERS:
        (base/name).mkdir(exist_ok=True)
    exe = Path(matlab).resolve(strict=True)
    commit = subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    assert commit == CONFIG["upstream_sha"]
    assert not subprocess.check_output(["git","-C",str(upstream),"status","--porcelain",
                                        "--untracked-files=no"],text=True).strip()
    stamp = datetime.now().astimezone()
    identifier = stamp.strftime("phase01_%Y%m%d_%H%M%S_%f") + ("_pilot" if pilot else "")
    run = base/PROJECT_FOLDERS[4]/identifier
    run.mkdir()
    for part in RUN_FOLDERS:
        (run/part).mkdir()
    log = run/RUN_FOLDERS[-1]
    print(f"RUN={run}", flush=True)
    input_source = upstream/CONFIG["upstream_image"]
    assert sha(input_source) == CONFIG["image_sha256"]
    image_dir = base/PROJECT_FOLDERS[3]/CONFIG["image_sha256"][:16]
    image_dir.mkdir(exist_ok=True)
    source_copy = image_dir/input_source.name
    if not source_copy.exists():
        with source_copy.open("xb") as out, input_source.open("rb") as image:
            shutil.copyfileobj(image,out)
    assert sha(source_copy) == CONFIG["image_sha256"]
    json_file(log/"run.json",dict(run_id=identifier, mode="pilot" if pilot else "formal",
              timestamp=stamp.isoformat(), actual_path=str(run), original_path=str(source_copy),
              code_path=str(CODE), matlab_executable=str(exe), upstream_path=str(upstream)))
    fixture_args = ["--pilot"] if pilot else []
    run_logged([sys.executable,"-m","unittest","discover","-s",CODE/"Tests","-v"],log/"python_unit_tests.log")
    run_logged([sys.executable,CODE/"Python"/"image_pipeline.py",run,upstream,*fixture_args],log/"python_generation.log")
    matlab_dir = CODE/"MATLAB"
    m_call = f"addpath({quote(matlab_dir)}); main_phase01({quote(run)},{str(pilot).lower()})"
    command = [exe]
    if os.name == "nt":
        command.append("-wait")
    run_logged([*command,"-sd",matlab_dir,"-logfile",log/"matlab_analysis.log","-batch",m_call],
               log/"matlab_analysis_launcher.log")
    run_logged([sys.executable,CODE/"Python"/"validate_run.py",run,*fixture_args],log/"python_validation.log")
    if not pilot:
        run_logged([sys.executable,CODE/"Python"/"validate_associations.py",run],
                   log/"association_reaggregation.log")
        fig_call = (f"addpath({quote(matlab_dir)}); "
                    f"paired_angle_summary({quote(run)}); generate_figures({quote(run)}); "
                    f"audit_figures({quote(run)})")
        run_logged([*command,"-sd",matlab_dir,"-logfile",log/"matlab_figures.log","-batch",fig_call],
                   log/"matlab_figures_launcher.log")
        run_logged([sys.executable,CODE/"Python"/"validation_plot.py",run],log/"python_figure.log")
        run_logged([sys.executable,CODE/"Python"/"verify_exports.py",run],log/"figure_checks.log")
        run_logged([sys.executable,CODE/"Python"/"verify_archive.py",run],log/"archive_checks.log")
        run_logged([sys.executable,CODE/"Python"/"report_evidence.py",run],log/"report_evidence.log")
        run_logged([sys.executable,CODE/"Python"/"figure_index.py",run],log/"figure_index.log")
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--matlab",type=Path,required=True)
    p.add_argument("--upstream",type=Path)
    p.add_argument("--pilot",action="store_true")
    args = p.parse_args()
    if args.upstream:
        launch(args.project,args.upstream.resolve(strict=True),args.matlab,args.pilot)
    else:
        with tempfile.TemporaryDirectory(prefix="phase01-upstream-") as d:
            folder = Path(d)
            subprocess.run(["git","init",str(folder)],check=True)
            subprocess.run(["git","-C",str(folder),"fetch","--depth=1",CONFIG["upstream_url"],
                            CONFIG["upstream_sha"]],check=True)
            subprocess.run(["git","-C",str(folder),"checkout","--detach","FETCH_HEAD"],check=True)
            launch(args.project,folder,args.matlab,args.pilot)


if __name__ == "__main__":
    main()
