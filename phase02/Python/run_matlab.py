"""用同一 MATLAB 可執行檔分方法運算，正式結果每組獨立保存。"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from allocation import METHODS
from calibrate import FOLDERS
from watermark import CODE


def matlab_quote(value):
    return "'" + str(value).replace("'","''") + "'"


def run_method(args):
    run,exe,method,wait=args
    out=run/FOLDERS[-1]/f"matlab_{method}.log"
    path1=CODE.parent/"phase01"/"MATLAB"
    path2=CODE/"MATLAB"
    cmd=f"addpath({matlab_quote(path1)}); addpath({matlab_quote(path2)}); analyze_method({matlab_quote(run)},'{method}')"
    if wait:
        while not (run/FOLDERS[5]/f"{method}_jpeg.h5").is_file():
            time.sleep(4)
    subprocess.run([str(exe),"-wait","-singleCompThread","-sd",str(path2),"-logfile",str(out),"-batch",cmd],check=True)
    print(method,"MATLAB completed",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("matlab",type=Path)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--wait-for-jpeg",action="store_true")
    p.add_argument("--methods",choices=METHODS,nargs="+",default=METHODS)
    args=p.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_method,[(args.run,args.matlab,m,args.wait_for_jpeg) for m in args.methods]))
