"""在每組 JPEG 與 HDF5 就緒後獨立啟動 MATLAB 分析。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import subprocess
import time

from image_experiment import DATA, LABELS, ROOT, STATS


def quote(p):
    return "'" + str(p).replace("'","''") + "'"


def batch(exe,command,log):
    code=f"addpath({quote(ROOT/'phase03'/'MATLAB')}); {command}"
    subprocess.run([str(exe),"-wait","-batch",code,"-logfile",str(log)],check=True)


def analyze(exe,run,label):
    root=run/DATA/label
    batch(exe,f"phase03_analyze({quote(run)},{quote(label)})",root/"matlab_analysis.log")
    outputs=[run/STATS/(label+"_"+name+"_matlab.csv") for name in ("matrix","checks","decision","relation")]
    assert all(p.is_file() for p in outputs), label
    print(label,"MATLAB statistics complete",flush=True)


def figures(exe,run,phase02,ids):
    label=str(ids[0])+":"+str(ids[-1])
    batch(exe,f"phase03_figures({quote(run)},{quote(phase02)},{label})",
          run/"02_研究規格"/f"matlab_figures_{ids[0]:02d}_{ids[-1]:02d}.log")


def main(args):
    labels={l for l in args.labels if not (args.run/STATS/(l+"_relation_matlab.csv")).is_file()}
    inflight={}
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while labels or inflight:
            ready=[l for l in LABELS if l in labels and (args.run/STATS/(l+"_ber_psnr.csv")).is_file()]
            jpeg_finished=all((args.run/STATS/(l+"_ber_psnr.csv")).is_file() for l in args.labels)
            limit=args.workers if jpeg_finished else min(args.workers,args.overlap_workers)
            for label in ready[:max(0,limit-len(inflight))]:
                labels.remove(label)
                inflight[pool.submit(analyze,args.matlab,args.run,label)]=label
            if inflight:
                completed,_=wait(inflight,timeout=5,return_when=FIRST_COMPLETED)
                for task in completed:
                    label=inflight.pop(task)
                    try:
                        task.result()
                    except Exception as error:
                        failures.append(label)
                        print(label,repr(error),flush=True)
            elif labels:
                if not args.wait_for_jpeg:
                    raise FileNotFoundError(f"Missing completed JPEG results for {sorted(labels)}")
                time.sleep(5)
    if failures:
        raise RuntimeError(f"MATLAB measurement failed for {', '.join(failures)}")
    if args.phase02:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(figures,args.matlab,args.run,args.phase02,r)
                   for r in (range(1,13),range(13,25))]
            for task in tasks:
                task.result()
        fig=args.run/"06_MATLAB圖表"
        assert all((fig/f"Fig{i:02d}.{ext}").is_file() for i in range(1,25)
                   for ext in ("fig","png","pdf","svg","csv"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","matlab"):
        p.add_argument(name,type=Path)
    p.add_argument("--phase02",type=Path)
    p.add_argument("--labels",nargs="+",choices=LABELS,default=LABELS)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--overlap-workers",type=int,default=3)
    p.add_argument("--wait-for-jpeg",action="store_true")
    main(p.parse_args())
