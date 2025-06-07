from typing import Iterable

def summarize_detection(results: Iterable) -> str:
    """결과 객체의 개수를 요약한 문자열을 반환합니다."""
    count = len(results) if results is not None else 0
    return f"감지된 객체 수: {count}"
