import os
import re
import json
from typing import List, Tuple, Optional

from preprocessing.extract_slices import extract_slices_from_directory
from representation.symbolic_transform import normalize_tokens  # if you want to normalize here
from utils.config import CONFIG

# Optional manifest: CONFIG may provide a path mapping relative file paths -> 0/1
MANIFEST_PATH: Optional[str] = CONFIG.get("labels_manifest")

# Regexes for quick, common patterns
WORD = re.compile(r"\b[_a-zA-Z][_a-zA-Z0-9]*\b")
VUL_PAT   = re.compile(r'(^|[_\-])(bad|vul|vuln|vulnerable|bug|unsafe)([_\-]|$)', re.IGNORECASE)
SAFE_PAT  = re.compile(r'(^|[_\-])(good|safe|benign|clean)([_\-]|$)', re.IGNORECASE)

RE_SLICE_BADCMT  = re.compile(r'/\*\s*ERROR\s*:', re.I)
RE_SLICE_GOODCMT = re.compile(r'/\*\s*(?:NO\s+ERROR|Tool should not detect this line as error)', re.I)

RE_SL_BAD_CALL  = re.compile(r'\b(gets|strcpy|strcat|sprintf|vsprintf|memcpy|memmove|tvb_memcpy|tvb_get_ptr)\b')
RE_SL_SCANF_STR = re.compile(r'\bscanf\s*\(\s*"[^\"]*%s')
RE_SL_SAFE_CALL = re.compile(r'\b(v?)snprintf|g_snprintf|strl(?:cpy|cat)|g_strl(?:cpy|cat)\b')
RE_SL_GUARD     = re.compile(r'\bif\s*\([^)]*(?:<|<=|sizeof|ARRAY_SIZE|NS_ARRAY_LENGTH)\b[^)]*\)')

RE_SL_ALLOC  = re.compile(r'\b(?:malloc|calloc|realloc|new|g_(?:malloc0?|try_malloc|realloc)|PR_Malloc|PR_Realloc|ast_malloc)\b')
RE_SL_FREE   = re.compile(r'\b(?:free|delete\s*\[\]?\s*|g_free|PR_Free|ast_free)\b')
RE_SL_OPEN   = re.compile(r'\b(?:fopen|open|socket|PR_Open|PR_OpenFile)\s*\(')
RE_SL_CLOSE  = re.compile(r'\b(?:fclose|close|closesocket|PR_Close)\s*\(')

# Guards and macros
RE_SL_ASSERT_GUARDS = re.compile(
    r'\b(?:assert|g_assert|DISSECTOR_ASSERT|WS_ASSERT|g_return_if_fail|NS_ENSURE_TRUE|NS_ASSERTION)\s*\(',
    re.I
)
RE_SL_OFFSET_LEN_GUARD = re.compile(
    r'\b(?:offset\s*\+\s*(?:len|length|needed|n)\s*(?:<=|<)\s*tvb_(?:reported|captured)?_?length(?:_remaining)?\s*\(\s*tvb\s*(?:,\s*offset)?\s*\))',
    re.I
)
RE_SL_TVB_REMAIN_GUARD = re.compile(
    r'\btvb_(?:reported|captured)?_?length(?:_remaining)?\s*\(\s*tvb\s*,\s*offset\s*\)\s*(?:>=|>)\s*(?:len|length|needed|n)\b',
    re.I
)

# wmem / memcpy size correlation
RE_SL_WMEM_ALLOC_ASSIGN = re.compile(
    r'\b([A-Za-z_]\w*)\s*=\s*wmem_\w+\s*\([^,]+,\s*([A-Za-z_]\w*)\s*\)\s*;',
    re.I
)
RE_SL_MEMCPY_WITH_VARLEN = re.compile(
    r'\bmem(?:cpy|move|set)\s*\(\s*([A-Za-z_]\w*)\s*,\s*[^,]+,\s*([A-Za-z_]\w*)\s*\)',
    re.I
)

# More “safe” string FNs
RE_SL_SAFE_STRING_FNS = re.compile(
    r'\b(?:g_snprintf|strl(?:cpy|cat)|g_strl(?:cpy|cat)|ast_copy_string|ast_strncpy)\b',
    re.I
)

_RE_ARR_DECL = re.compile(r'\b(?:char|unsigned\s+char|signed\s+char|wchar_t|uint8_t|int|short|long|unsigned|size_t)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;')
_RE_CALL     = re.compile(r'\b(memcpy|memmove|memset)\s*\(\s*([A-Za-z_]\w*)[^,]*,\s*[^,]+,\s*(\d+)\s*\)')
_RE_SIZEOFN  = re.compile(r'\b(memcpy|memmove|memset)\s*\(\s*([A-Za-z_]\w*)[^,]*,\s*[^,]+,\s*sizeof\s*\(\s*\2\s*\)\s*\)')

RE_SL_TVB_LEN = re.compile(r'\btvb_(?:reported|captured)?_?length(?:_remaining)?\b', re.I)
RE_SL_PROTO_ADD = re.compile(r'\bproto_tree_add_item\s*\(', re.I)
RE_SL_EMEM = re.compile(r'\b(?:ep_|se_|emem_)', re.I)  # Wireshark pools
POLICY = CONFIG.get("label_unknown_policy", "manifest")  # "manifest"|"skip"|"safe"
MANIFEST = None

if CONFIG.get("labels_manifest"):
    mp = CONFIG["labels_manifest"]
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            MANIFEST = json.load(f)

# load function-level labels if present
FUNC_LABELS = None
func_labels_path = os.path.join(os.path.dirname(CONFIG["labels_manifest"]), "function_labels.json")
if os.path.exists(func_labels_path):
    with open(func_labels_path, "r", encoding="utf-8") as f:
        FUNC_LABELS = json.load(f)

def _tokens_from_slice(code_slice: List[str]) -> List[str]:
    return WORD.findall(" ".join(code_slice))

def _load_manifest(manifest_path: Optional[str]):
    mapping = None
    if manifest_path and os.path.exists(manifest_path):
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)  # expects {"relative/path/to/file.c": 0 or 1, ...}
        print(f"[Labeler] Loaded manifest with {len(mapping)} entries from {manifest_path}")
    return mapping

_MANIFEST = _load_manifest(MANIFEST_PATH)
def _slice_sizeflow_label(txt: str) -> int | None:
    arrays = {}
    for m in _RE_ARR_DECL.finditer(txt):
        arrays[m.group(1)] = int(m.group(2))
    # sizeof(dest) is safe
    if _RE_SIZEOFN.search(txt):
        return 0
    # concrete numeric length
    for m in _RE_CALL.finditer(txt):
        dest, n = m.group(2), int(m.group(3))
        if dest in arrays:
            if n > arrays[dest]:
                return 1
            else:
                return 0
    return None

def _label_from_slice_text(slice_text: str) -> int | None:
    # CWE-119 cues
    danger = RE_SL_BAD_CALL.search(slice_text) or RE_SL_SCANF_STR.search(slice_text)
    safeish = bool(
        RE_SL_SAFE_CALL.search(slice_text) or
        RE_SL_SAFE_STRING_FNS.search(slice_text) or
        RE_SL_GUARD.search(slice_text) or
        RE_SL_ASSERT_GUARDS.search(slice_text) or
        RE_SL_OFFSET_LEN_GUARD.search(slice_text) or
        RE_SL_TVB_REMAIN_GUARD.search(slice_text)
    )

    # Heuristic: proto_tree_add_item + tvb_len with no guards
    unguarded_tree_add = bool(RE_SL_PROTO_ADD.search(slice_text) and RE_SL_TVB_LEN.search(slice_text) and not safeish)
    # CWE-399 cues
    alloc = RE_SL_ALLOC.search(slice_text)
    free  = RE_SL_FREE .search(slice_text)
    opn   = RE_SL_OPEN .search(slice_text)
    cls   = RE_SL_CLOSE.search(slice_text)

    using_pools = bool(RE_SL_EMEM.search(slice_text) or "wmem_" in slice_text)
    leak_like = ((alloc and not free) or (opn and not cls)) and not using_pools

    # Correlate wmem alloc size to copy length within the slice
    wmem_safe = False
    for dest, nvar in RE_SL_WMEM_ALLOC_ASSIGN.findall(slice_text):
        for md, mv in RE_SL_MEMCPY_WITH_VARLEN.findall(slice_text):
            if md == dest and mv == nvar:
                wmem_safe = True
                break
        if wmem_safe:
            break

    if (danger and not safeish) or unguarded_tree_add or leak_like:
        return 1
    if (safeish or wmem_safe) and not (danger or unguarded_tree_add or leak_like):
        return 0
    return None

def _relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except Exception:
        return os.path.basename(path)

def _label_from_manifest(path: str):
    if _MANIFEST is None:
        return None
    return _MANIFEST.get(path, None)

def _label_from_name(name: str) -> Optional[int]:
    """Infer label from a single path component (filename or directory name)."""
    if SAFE_PAT.search(name):
        return 0
    if VUL_PAT.search(name):
        return 1
    return None

def _label_from_tokens(code_tokens: List[str]) -> Optional[int]:
    """
    Weak heuristic: if a function identifier contains 'bad' or 'good'.
    Works for Juliet-style datasets with foo_bad(), foo_good().
    """
    # collapse to identifiers
    joined = " ".join(code_tokens)
    idents = WORD.findall(joined.lower())
    # Look for exact token 'bad'/'good' or suffix/prefix with underscores
    if any(tok == "good" or tok.endswith("_good") or tok.startswith("good_") for tok in idents):
        return 0
    if any(tok == "bad" or tok.endswith("_bad") or tok.startswith("bad_") for tok in idents):
        return 1
    return None

def label_from_path(path: str, root: str, code_tokens: Optional[List[str]] = None) -> Optional[int]:
    """
    Multi-source labeling with fallbacks:
    1) Manifest (if provided)
    2) Filename basename
    3) Parent directory name
    4) Function-name heuristic from tokens
    Returns 0/1, or None if undecidable.
    """
    # 1) Manifest wins
    m = _label_from_manifest(path, root)
    if m is not None:
        return m

    # 2) Filename
    base = os.path.basename(path).lower()
    lb = _label_from_name(base)
    if lb is not None:
        return lb

    # 3) Parent directory
    parent = os.path.basename(os.path.dirname(path)).lower()
    lb = _label_from_name(parent)
    if lb is not None:
        return lb

    # 4) Function-name heuristic from tokens
    if code_tokens:
        lb = _label_from_tokens(code_tokens)
        if lb is not None:
            return lb

    # Unknown → None
    return None

def generate_gadgets(root_dir: str) -> List[Tuple[List[str], int, str]]:
    slices = extract_slices_from_directory(root_dir)
    gadgets = []
    skipped = 0

    for file_path, line_no, code_slice in slices:
        rel = os.path.relpath(file_path, root_dir).replace("\\", "/")
        raw_tokens = _tokens_from_slice(code_slice)

        lab = _label_from_manifest(rel)

        if lab in (0, 1):
            label = int(lab)

        elif lab == "both":
            # 1) size-flow & real-world heuristics first
            slice_text = " ".join(code_slice) if isinstance(code_slice, (list, tuple)) else str(code_slice)
            label = _slice_sizeflow_label(slice_text)
            if label is None:
                label = _label_from_slice_text(slice_text)  # your real-world slice heuristic

            # 2) fallback to token heuristic
            if label is None:
                label = _label_from_tokens(raw_tokens)

            # 3) policy
            if label is None:
                if POLICY == "safe":
                    label = 0
                elif POLICY in ("skip", "manifest"):
                    skipped += 1
                    continue

        else:
            # lab is None (no manifest entry)
            slice_text = " ".join(code_slice) if isinstance(code_slice, (list, tuple)) else str(code_slice)

            # 1) size-flow first (exact-fit / guarded copies => safe; obvious over-copies => vul)
            label = _slice_sizeflow_label(slice_text)   # <-- add the helper earlier

            # 2) then real-world CWE-119/399 cues (Wireshark/Mozilla/etc.)
            if label is None:
                label = _label_from_slice_text(slice_text)

            # 3) then your filename/parent heuristics
            if label is None:
                base   = os.path.basename(file_path).lower()
                parent = os.path.basename(os.path.dirname(file_path)).lower()
                if SAFE_PAT.search(base) or SAFE_PAT.search(parent):
                    label = 0
                elif VUL_PAT.search(base) or VUL_PAT.search(parent):
                    label = 1
                else:
                    # 4) last resort: token cue
                    label = _label_from_tokens(raw_tokens)

            # 5) still unknown -> respect POLICY
            if label is None:
                if POLICY == "safe":
                    label = 0
                elif POLICY in ("skip", "manifest"):
                    skipped += 1
                    continue

        # now normalize for modeling (unchanged)
        tokens = normalize_tokens(raw_tokens)
        gadgets.append((tokens, label, file_path))

    print(f"[Labeler] Built {len(gadgets)} gadgets (skipped {skipped} undecidable).")
    return gadgets


if __name__ == "__main__":
    data = generate_gadgets(CONFIG["code_dir"])
    # Quick preview & class distribution
    import numpy as np, collections
    labels = np.array([lbl for _, lbl, _ in data], dtype=np.int32)
    dist = collections.Counter(labels.tolist())
    print("[Labeler] class_dist (preview):", dict(dist))
    # Save a human-readable sample
    preview_path = os.path.join(CONFIG["code_dir"], "gadgets_preview.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        for tokens, label, path in data[:50]:
            f.write(f"Label: {label} | {os.path.basename(path)}\n{' '.join(tokens)}\n{'-'*60}\n")
    print(f"[Labeler] Preview saved to {preview_path}")
