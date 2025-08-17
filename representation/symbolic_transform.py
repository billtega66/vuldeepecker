def normalize_tokens(tokens):
    """
    Replace all variable names and literals with symbolic tokens.
    Example: variables → VAR, numeric constants → NUM, strings → STR
    """
    normalized = []
    for token in tokens:
        if token.isdigit():
            normalized.append("NUM")
        elif token.startswith("\"") or token.startswith("'"):
            normalized.append("STR")
        elif token in {"if", "for", "while", "return", "int", "char", "void", "NULL"}:
            normalized.append(token)
        elif token.isidentifier():
            normalized.append("VAR")
        else:
            normalized.append(token)
    return normalized


if __name__ == "__main__":
    sample = ["int", "main", "(", ")", "{", "int", "x", "=", "42", ";", "char", "*", "s", "=", "\"hello\"", ";", "}"]
    print("Original:", sample)
    print("Normalized:", normalize_tokens(sample))
