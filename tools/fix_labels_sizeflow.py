# tools/fix_labels_sizeflow.py
import os, re, sys, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

ROOT = CONFIG["code_dir"]
IN_PATH = CONFIG["labels_manifest"]
OUT_PATH = (IN_PATH if "--inplace" in sys.argv else IN_PATH.replace(".json", "_sizeflow.json"))

C_EXTS = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"}

# Decls like: char buf[10]; unsigned char src[LEN]; int a[4];
RE_ARR_DECL = re.compile(r'\b(?:char|unsigned\s+char|signed\s+char|wchar_t|uint8_t|int|short|long|unsigned|size_t)\s+([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]\s*;')
# Assignments: int n = 10; size = sizeof(buf); n = MIN(x,y); n = std::min(x,y);
RE_ASSIGN   = re.compile(r'\b([A-Za-z_]\w*)\s*=\s*([^;]+);')
RE_SIZEOF   = re.compile(r'\bsizeof\s*\(\s*([A-Za-z_]\w*)\s*\)')
RE_NUM      = re.compile(r'^\s*(\d+)\s*$')
RE_MIN      = re.compile(r'\b(?:MIN|min|std::min)\s*\(\s*([^,]+)\s*,\s*([^)]+)\)')
# memcpy/memmove/memset calls
RE_CALL     = re.compile(r'\b(memcpy|memmove|memset)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)')
# Guards: if (n <= sizeof(buf)) etc.
RE_IF_GUARD = re.compile(r'\bif\s*\([^)]*(?:<=|<)\s*[^)]*\)')
RE_ASSERT   = re.compile(r'\b(?:assert|g_assert|DISSECTOR_ASSERT|WS_ASSERT)\s*\(')

def _num_or_size(expr, arrays, assigns):
    expr = expr.strip()
    # sizeof(var)
    m = RE_SIZEOF.search(expr)
    if m:
        var = m.group(1)
        return arrays.get(var)  # known array length
    # numeric literal
    m = RE_NUM.match(expr)
    if m:
        return int(m.group(1))
    # variable known from assigns
    if expr in assigns and isinstance(assigns[expr], int):
        return assigns[expr]
    return None

def _eval_rhs(rhs, arrays, assigns):
    rhs = rhs.strip()
    # sizeof()
    v = _num_or_size(rhs, arrays, assigns)
    if v is not None:
        return v
    # MIN-like
    m = RE_MIN.search(rhs)
    if m:
        a = _num_or_size(m.group(1), arrays, assigns)
        b = _num_or_size(m.group(2), arrays, assigns)
        if a is not None and b is not None:
            return min(a, b)
    # bare symbol with known size/const
    return _num_or_size(rhs, arrays, assigns)

def decide_sizeflow(text):
    """
    Try to decide 0/1/"both"/None for a file using simple size reasoning.
    - Safe if every memcpy/memmove/memset is bounded by array length (or guarded/asserted).
    - Vul if we find any call with a concrete length > dest length or no guarding against overflow.
    - Both if we see a mix.
    """
    if not text:
        return None

    arrays = {}   # name -> integer length if known
    assigns = {}  # name -> integer if can infer

    lines = text.splitlines()
    safe_hits, vul_hits = 0, 0

    for ln in lines:
        # collect array declarations
        for m in RE_ARR_DECL.finditer(ln):
            name, dim = m.group(1), m.group(2)
            v = None
            if dim.isdigit():
                v = int(dim)
            else:
                v = assigns.get(dim)  # dimension via macro/var we may know
            if v is not None and v > 0:
                arrays[name] = v

        # collect assignments we can evaluate (n=10; n=sizeof(buf); n=min(a,b)...)
        for m in RE_ASSIGN.finditer(ln):
            lhs, rhs = m.group(1), m.group(2)
            v = _eval_rhs(rhs, arrays, assigns)
            if isinstance(v, int) and v >= 0:
                assigns[lhs] = v

        # check calls
        for m in RE_CALL.finditer(ln):
            fn, dst, src, nexpr = m.groups()
            # strip & normalize destination var (allow &buf[0], buf, *buf)
            dst_var = None
            # try pattern like: buf, &buf[0], buf + 0
            m2 = re.search(r'([A-Za-z_]\w*)', dst)
            if m2:
                dst_var = m2.group(1)

            nval = _eval_rhs(nexpr, arrays, assigns)
            guarded = bool(RE_IF_GUARD.search(ln) or RE_ASSERT.search(ln))
            if dst_var and dst_var in arrays and nval is not None:
                if nval > arrays[dst_var]:
                    vul_hits += 1
                else:
                    safe_hits += 1
            else:
                # no concrete proof; if we see guards/asserts on same line, consider safe-ish
                if guarded:
                    safe_hits += 1

    if vul_hits and safe_hits:
        return "both"
    if vul_hits:
        return 1
    if safe_hits:
        return 0
    return None

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    updates = {}
    for rel, lab in labels.items():
        if lab is not None:
            continue
        ext = os.path.splitext(rel)[1].lower()
        if ext not in C_EXTS:
            continue
        full = os.path.join(ROOT, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(800_000)
        except Exception:
            continue
        decide = decide_sizeflow(text)
        if decide is not None:
            updates[rel] = decide

    out = labels.copy()
    out.update(updates)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    before = {"unknown": sum(v is None for v in labels.values())}
    after  = {"unknown": sum(v is None for v in out.values())}
    print("[sizeflow] input :", IN_PATH)
    print("[sizeflow] output:", OUT_PATH)
    print("[sizeflow] changes:", len(updates))
    print("[sizeflow] before :", before)
    print("[sizeflow] after  :", after)

if __name__ == "__main__":
    main()
