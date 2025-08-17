#!/usr/bin/env python3
"""
Extract (copy) a specific set of files from a dataset tree into a separate folder,
preserving their relative directory structure.

Usage:
  python extract_selected_files.py --root /path/to/vuldeepecker_root --out /path/to/output

If you prefer to maintain a list in a text file, add:
  --list /path/to/relpaths.txt   # one relative path per line (relative to --root)
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_RELATIVE_PATHS = [
    # --- CWE-119 / 148881 ---
    "CWE-119/source_files/148881/capture_wpcap_packet.c",
    "CWE-119/source_files/148881/diam_dict.c",
    "CWE-119/source_files/148881/emem.c",
    "CWE-119/source_files/148881/epan.c",
    "CWE-119/source_files/148881/erf.c",
    "CWE-119/source_files/148881/packet-afs.c",
    "CWE-119/source_files/148881/packet-bpkmreq.c",
    "CWE-119/source_files/148881/packet-bpkmrsp.c",

    # --- CWE-119 / 148966 ---
    "CWE-119/source_files/148966/emem.c",
    "CWE-119/source_files/148966/packet-cip.c",
    "CWE-119/source_files/148966/packet-dcp-etsi.c",
    "CWE-119/source_files/148966/packet-http.c",
    "CWE-119/source_files/148966/packet-mpeg-dsmcc.c",
    "CWE-119/source_files/148966/packet-sflow.c",
    "CWE-119/source_files/148966/reassemble.c",
    "CWE-119/source_files/148966/tvbuff.c",

    # --- CWE-399 / 148818 ---
    "CWE-399/source_files/148818/app_milliwatt.c",
    "CWE-399/source_files/148818/channel.c",
    "CWE-399/source_files/148818/chan_sip.c",
    "CWE-399/source_files/148818/devicestate.c",
    "CWE-399/source_files/148818/event.c",
    "CWE-399/source_files/148818/http.c",
    "CWE-399/source_files/148818/res_jabber.c",
    "CWE-399/source_files/148818/strings.h",

    # --- Top-level ---
    "README.md",
    "split_cache_preview.pkl",
    "vectorized_gadgets.pkl",

    # --- CWE-119 / 1310 ---
    "CWE-119/source_files/1310/txt-dns-file-ok.c",
    "CWE-119/source_files/1310/txt-dns.h",

    # --- CWE-119 / CVE-2014-8713 ---
    "CWE-119/source_files/CVE-2014-8713/CVE-2014-8713.txt",
    "CWE-119/source_files/CVE-2014-8713/Wireshark_1.12.1_CVE_2014_8713_epan_dissectors_packet-ncp2222.inc",

    # --- CWE-399 / 148849 ---
    "CWE-399/source_files/148849/epan.c",
    "CWE-399/source_files/148849/tvbuff.c",

    # --- CWE-399 / CVE-2012-3988 ---
    "CWE-399/source_files/CVE-2012-3988/CVE-2012-3988.txt",
    "CWE-399/source_files/CVE-2012-3988/firefox_15.0b6_CVE_2012_3988_docshell_base_Makefile.in",

    # --- CWE-119 / misc ---
    "CWE-119/.gitattributes",
    "CWE-119/source_files/13/Write-what-where_condition.c",
    "CWE-119/source_files/1300/recipient-ok.c",
    "CWE-119/source_files/1303/mime2.h",
    "CWE-119/source_files/1304/mime2.h",
    "CWE-119/source_files/1309/txt-dns.h",
    "CWE-119/source_files/1335/threaded_memccpy_bad1.c",
    "CWE-119/source_files/14/Stack_overflow.c",
    "CWE-119/source_files/148804/strings.h",
    "CWE-119/source_files/148923/tvbuff.c",
    "CWE-119/source_files/149042/s3_red_both.c",
    "CWE-119/source_files/1495/Figure2-29-windows.cpp",
]

def read_paths_from_file(pathfile: Path):
    rels = []
    for line in pathfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rels.append(line)
    return rels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory of the dataset (where CWE-119, CWE-399 live)")
    ap.add_argument("--out", required=True, help="Output directory to copy files into (structure preserved)")
    ap.add_argument("--list", help="Optional path to a text file with relative paths (one per line). If omitted, uses the built-in list above.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out  = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.list:
        rel_paths = read_paths_from_file(Path(args.list))
    else:
        rel_paths = DEFAULT_RELATIVE_PATHS

    found, missing, copied = 0, 0, 0
    for rel in rel_paths:
        src = root / rel
        if src.exists() and src.is_file():
            found += 1
            dst = out / rel  # preserve tree
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"[MISS] {src}")

    print(f"\nDone. Found: {found}, Copied: {copied}, Missing: {missing}")
    print(f"Output root: {out}")

if __name__ == "__main__":
    main()
