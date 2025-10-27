import os
import csv
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data/files")
    p.add_argument("--output", default="./data/labels.csv")
    p.add_argument("--pos", nargs="*", default=["malware", "malicious", "positive"])
    p.add_argument("--neg", nargs="*", default=["benign", "negative", "clean"])
    args = p.parse_args()
    pos = set([s.lower() for s in args.pos])
    neg = set([s.lower() for s in args.neg])
    rows = []
    for dirpath, _, files in os.walk(args.root):
        base = os.path.basename(dirpath).lower()
        label = None
        if base in pos:
            label = 1
        elif base in neg:
            label = 0
        for f in files:
            fl = f.lower()
            if fl.startswith("."):
                continue
            if fl.endswith((".pdf", ".png", ".jpg", ".jpeg")):
                y = 0 if label is None else label
                rows.append([f, y])
    rows.sort(key=lambda r: r[0])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    main()

