import os
import glob
import re

TARGET_EXTENSIONS = [".c", ".cpp", ".cc"]
API_CALLS = [
    "malloc", "free", "strcpy", "memcpy", "strncpy", "strdup", "fopen", "fclose"

    # existing Juliet seeds…
    "bad", "good", "goodG2B", "goodB2G",
    # CWE-119
    "gets", "strcpy", "strcat", "sprintf", "vsprintf", "scanf", "memcpy", "memmove",
    "tvb_memcpy", "tvb_get_ptr", "tvb_reported_length", "tvb_reported_length_remaining",
    "tvb_captured_length", "tvb_captured_length_remaining", "proto_tree_add_item",
    # CWE-399
    "malloc", "calloc", "realloc", "free", "new", "delete",
    "g_malloc", "g_malloc0", "g_try_malloc", "g_realloc", "g_free",
    "PR_Malloc", "PR_Realloc", "PR_Free", "PR_Open", "PR_OpenFile", "PR_Close",
    "ast_malloc", "ast_free",
    "open", "close", "closesocket", "fopen", "fclose", "socket",
    # safe guards (to capture context, even if safe)
    "snprintf", "g_snprintf", "strlcpy", "strlcat", "g_strlcpy", "g_strlcat",
        "tvb_get_guint8","tvb_get_guint16","tvb_get_guint32",
    "tvb_get_ntohs","tvb_get_ntohl","tvb_get_letohs","tvb_get_letohl",
    "tvb_get_bytes","tvb_reported_length","tvb_reported_length_remaining",
    "tvb_captured_length","tvb_captured_length_remaining",
    # Wireshark allocation
    "wmem_alloc","wmem_alloc0","wmem_realloc",
    # Assertions/guards
    "DISSECTOR_ASSERT","WS_ASSERT","g_assert","g_return_if_fail",
    "NS_ENSURE_TRUE","NS_ASSERTION",
    # Asterisk safer string helpers
    "ast_copy_string","ast_strncpy",
]


def extract_slices_from_file(file_path, window=5):
    with open(file_path, "r", errors='ignore') as f:
        lines = f.readlines()

    slices = []
    for i, line in enumerate(lines):
        if any(api in line for api in API_CALLS):
            start = max(0, i - window)
            end = min(len(lines), i + window + 1)
            code_slice = lines[start:end]
            slices.append((file_path, i, code_slice))
    return slices


def extract_slices_from_directory(root_dir):
    all_slices = []
    for ext in TARGET_EXTENSIONS:
        for file_path in glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True):
            slices = extract_slices_from_file(file_path)
            all_slices.extend(slices)
    return all_slices


if __name__ == "__main__":
    from utils.config import CONFIG
    save_path = os.path.join(CONFIG["code_dir"], "slices_preview.txt")
    result = extract_slices_from_directory(CONFIG["code_dir"])
    with open(save_path, "w") as f:
        for file_path, line_no, code_slice in result:
            f.write(f"File: {file_path}, Line: {line_no}\n")
            f.writelines(code_slice)
            f.write("\n" + "-" * 60 + "\n")
    print(f"Extracted {len(result)} code slices. Preview saved to {save_path}.")
