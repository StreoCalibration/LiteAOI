"""데이터셋 다운로드를 담당하는 모듈."""
from typing import Optional


def download_dataset(url: str, dest_path: str) -> None:
    """주어진 URL에서 데이터셋을 다운로드합니다.

    실제 다운로드 로직은 구현되어 있지 않고 메시지만 출력합니다.

    Args:
        url: 데이터셋이 호스팅된 Git 또는 HTTP URL.
        dest_path: 데이터를 저장할 경로.
    """
    print(f"데이터셋 다운로드: {url} -> {dest_path}")
    # 실제 다운로드 로직은 필요에 따라 구현

