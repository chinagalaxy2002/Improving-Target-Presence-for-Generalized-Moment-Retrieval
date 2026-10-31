#!/usr/bin/env python3
"""Re-render a published comparison snapshot using only NumPy and Matplotlib."""
import argparse
import json
from pathlib import Path

import plotting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    snapshot=json.loads(args.snapshot.read_text())
    plotting.setup(snapshot["methods"],snapshot["labels"])
    plotting.configure_style()
    plotting.render(snapshot["backbone"],snapshot["data"],args.output)


if __name__=="__main__":
    main()
