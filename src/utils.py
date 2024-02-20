# Code module for the utility functions


def syntax_check(code: str) -> dict:
    try:
        compile(code, "<string>", "exec")
        return {"status": "success"}
    except SyntaxError as e:
        return {"status": "error", "line": e.lineno, "message": e.msg}
