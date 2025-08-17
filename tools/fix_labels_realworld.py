# tools/fix_labels_realworld.py
import os, re, sys, json, collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

C_EXTS = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".inc", ".inl"}
ROOT = CONFIG["code_dir"]
IN_PATH = CONFIG["labels_manifest"]
OUT_PATH = (IN_PATH if "--inplace" in sys.argv
            else IN_PATH.replace(".json", "_realworld.json"))

# ---------- CWE-119: buffer/overflow-ish, dangerous by default ----------
DANGEROUS_FUNCS = [
    r"\bgets\s*\(",
    r"\bstrcpy\s*\(",
    r"\bstrcat\s*\(",
    r"\bsprintf\s*\(",
    r"\bvsprintf\s*\(",
    r'\bscanf\s*\(\s*"[^\"]*%s',           # scanf("%s")
    r"\bmemcpy\s*\(",
    r"\bmemmove\s*\(",
    # Wireshark: copying from captured buffer directly
    r"\btvb_memcpy\s*\(",
    r"\btvb_get_ptr\s*\(",
]

RE_DANGER = re.compile("|".join(DANGEROUS_FUNCS), re.I)

# if memcpy/memmove length comes from strlen/wcslen or from tvb length without guard
RE_LEN_FROM_STR = re.compile(r"\b(memcpy|memmove)\s*\([^,]+,\s*[^,]+,\s*(strlen|wcslen)\s*\(", re.I)
RE_LEN_FROM_TVB  = re.compile(r"\b(memcpy|memmove)\s*\([^,]+,\s*[^,]+,\s*(?:tvb_(?:reported|captured)?_?length(?:_remaining)?)\s*\(", re.I)

# "safer" calls / guards
RE_SNPRINTF_SAFE = re.compile(r"\b(v?)snprintf|g_snprintf|strl(?:cpy|cat)|g_strl(?:cpy|cat)\b", re.I)
RE_GUARD_IF      = re.compile(r"\bif\s*\([^)]*(?:<|<=|sizeof|ARRAY_SIZE|NS_ARRAY_LENGTH)\b[^)]*\)")
RE_RANGE_CHECK   = re.compile(r"\b(?:MIN|CLAMP|BOUND|cap|limit)\b", re.I)
# ---- Extra guards & project macros (Wireshark/Mozilla/Asterisk) ----
RE_ASSERT_GUARDS = re.compile(
    r'\b(?:assert|g_assert|DISSECTOR_ASSERT|WS_ASSERT|g_return_if_fail|'
    r'NS_ENSURE_TRUE|NS_ASSERTION|MOZ_ASSERT|MOZ_DIAGNOSTIC_ASSERT|'
    r'NS_WARN_IF|NS_ENSURE_ARG(?:_POINTER)?)\s*\(',
    re.I
)

# Offset/length guards seen in dissectors
RE_OFFSET_LEN_GUARD = re.compile(
    r'\b(?:offset\s*\+\s*(?:len|length|needed|n)\s*(?:<=|<)\s*tvb_(?:reported|captured)?_?length(?:_remaining)?\s*\(\s*tvb\s*(?:,\s*offset)?\s*\))',
    re.I
)
RE_TVB_REMAIN_GUARD = re.compile(
    r'\btvb_(?:reported|captured)?_?length(?:_remaining)?\s*\(\s*tvb\s*,\s*offset\s*\)\s*(?:>=|>)\s*(?:len|length|needed|n)\b',
    re.I
)

# Correlate destination allocation with copy length: dest = wmem_alloc(..., N); memcpy(dest, ..., N)
RE_WMEM_ALLOC_ASSIGN = re.compile(
    r'\b([A-Za-z_]\w*)\s*=\s*wmem_\w+\s*\([^,]+,\s*([A-Za-z_]\w*)\s*\)\s*;',
    re.I
)
RE_MEMCPY_WITH_VARLEN = re.compile(
    r'\bmem(?:cpy|move|set)\s*\(\s*([A-Za-z_]\w*)\s*,\s*[^,]+,\s*([A-Za-z_]\w*)\s*\)',
    re.I
)

# Treat more “safe” APIs as safeish
RE_SAFE_STRING_FNS = re.compile(
    r'\b(?:g_snprintf|strl(?:cpy|cat)|g_strl(?:cpy|cat)|ast_copy_string|ast_strncpy)\b',
    re.I
)

# ---------- CWE-399: resource mgmt ----------
# alloc/free families
RE_ALLOC  = re.compile(r"\b(?:malloc|calloc|realloc|new\b|g_(?:malloc0?|try_malloc|realloc)|PR_Malloc|PR_Realloc|ast_malloc)\b")
RE_FREE   = re.compile(r"\b(?:free|delete\s*\[\]?\s*|g_free|PR_Free|ast_free)\b")

# open/close families (file/socket/NSS)
RE_FOPEN  = re.compile(r"\b(?:fopen|open|socket|PR_Open|PR_OpenFile)\s*\(")
RE_FCLOSE = re.compile(r"\b(?:fclose|close|closesocket|PR_Close)\s*\(")

# classic double-free / UAF hints
RE_DOUBLE_FREE = re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;[^;]*\bfree\s*\(\s*\1\s*\)", re.S)
RE_UAF = re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;[^;]*\b(\1->|\*\1)", re.S)

# Wireshark length/guard symbols
RE_TVB_LEN_SYMS = re.compile(r"\btvb_(?:reported|captured)?_?length(?:_remaining)?\b", re.I)
RE_PROTO_ADD    = re.compile(r"\bproto_tree_add_item\s*\(", re.I)

# Ignore Wireshark emem/ep/se pool allocators (not leaks in design)
RE_EMEM_ALLOC = re.compile(r"\b(?:ep_|se_|emem_)", re.I)  # presence means pool alloc is in use
def read_text(fp):
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(800_000)
    except Exception:
        return ""

def decide_file_label(text: str, rel: str) -> int | str | None:
    if not text:
        return None

    # Skip header-only / mostly decls
    if rel.lower().endswith((".h", ".inl")) and text.count("\n") < 5:
        return None

    # CWE-119 danger present?
    danger = bool(
        RE_DANGER.search(text) or RE_LEN_FROM_STR.search(text) or RE_LEN_FROM_TVB.search(text)
    )
    safeish = bool(RE_SNPRINTF_SAFE.search(text) 
                    or RE_GUARD_IF.search(text) 
                    or RE_RANGE_CHECK.search(text)
                    or RE_ASSERT_GUARDS.search(text)
                    or RE_OFFSET_LEN_GUARD.search(text)
                    or RE_TVB_REMAIN_GUARD.search(text)
                    or RE_SAFE_STRING_FNS.search(text))

    # If we see proto_tree_add_item AND tvb length symbols but no guards, that’s suspect
    tvb_len_present = bool(RE_TVB_LEN_SYMS.search(text))
    tree_add_present = bool(RE_PROTO_ADD.search(text))
    unguarded_tree_add = tree_add_present and tvb_len_present and not safeish

    # CWE-399 resource mgmt
    alloc = bool(RE_ALLOC.search(text))
    free  = bool(RE_FREE.search(text))
    opn   = bool(RE_FOPEN.search(text))
    cls   = bool(RE_FCLOSE.search(text))

    # Ignore Wireshark emem/ep/se pools for leak judgement
    using_pools = bool(RE_EMEM_ALLOC.search(text))
    leak_like = ((alloc and not free) or (opn and not cls)) and not using_pools

    double_free = bool(RE_DOUBLE_FREE.search(text))
    uaf         = bool(RE_UAF.search(text))
        # Correlate pool/dynamic alloc size to copy length (safe if sizes match)
    wmem_pairs = set(RE_WMEM_ALLOC_ASSIGN.findall(text))  # {(dest, Nvar), ...}
    wmem_safe = False
    if wmem_pairs:
        for dest, nvar in wmem_pairs:
            # look for memcpy(dest, ..., nvar)
            for md, mv in RE_MEMCPY_WITH_VARLEN.findall(text):
                if md == dest and mv == nvar:
                    wmem_safe = True
                    break
            if wmem_safe:
                break

    vul_signals = (danger and not safeish) or unguarded_tree_add or leak_like or double_free or uaf
    safe_signals = (safeish or wmem_safe) and not (danger or unguarded_tree_add or leak_like or double_free or uaf)

    if vul_signals and safe_signals:
        return "both"
    if vul_signals:
        return 1
    if safe_signals:
        return 0

    # Optional: conservative CVE fallback — only if nothing else matched
    if "/CVE-" in rel or "\\CVE-" in rel:
        return "both"  # let per-slice logic decide inside

    return None
def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    before = {
        "unknown": sum(1 for v in labels.values() if v is None),
        "both":    sum(1 for v in labels.values() if v == "both"),
        "safe":    sum(1 for v in labels.values() if v == 0),
        "vul":     sum(1 for v in labels.values() if v == 1),
    }

    updates = {}
    for rel, lab in labels.items():
        if lab is not None:
            continue
        ext = os.path.splitext(rel)[1].lower()
        # Only skip obviously non-code (markdown, txt, pickle, etc.)
        if ext in {".md", ".txt", ".pkl", ".json", ".pdf"}:
            continue
        full = os.path.join(ROOT, rel)
        text = read_text(full)
        decide = decide_file_label(text, rel)
        if decide is not None:
            updates[rel] = decide

    labels_out = labels.copy()
    for rel, v in updates.items():
        labels_out[rel] = v

    after = {
        "unknown": sum(1 for v in labels_out.values() if v is None),
        "both":    sum(1 for v in labels_out.values() if v == "both"),
        "safe":    sum(1 for v in labels_out.values() if v == 0),
        "vul":     sum(1 for v in labels_out.values() if v == 1),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_out, f, indent=2)

    print("[realworld] input :", IN_PATH)
    print("[realworld] output:", OUT_PATH)
    print("[realworld] changes:", len(updates))
    print("[realworld] before :", before)
    print("[realworld] after  :", after)
    print("Tip: run fix_labels_family --inplace after this to propagate to lettered siblings.")

if __name__ == "__main__":
    main()
