"""
이미지 다운로더 유틸.

백엔드가 보내준 URL(s3://, http(s)://) 에서 이미지를 다운로드하여
로컬 임시 파일 경로를 반환한다. detector는 이 로컬 경로를 받아 분석한다.

지원 URL 형식:
- s3://bucket-name/path/to/image.png      ← S3/MinIO
- https://example.com/image.png            ← HTTP

"""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError

from app.core.config import settings


logger = logging.getLogger(__name__)


def download_image(image_url: str) -> str:
    """
    이미지 URL에서 다운로드하여 로컬 임시 파일 경로 반환.
    
    Args:
        image_url: 이미지 URL (s3:// 또는 http(s)://)
    
    Returns:
        다운로드된 로컬 파일의 절대 경로
    
    Raises:
        ValueError: 지원되지 않는 URL 스킴
        IOError: 다운로드 실패
    """
    parsed = urlparse(image_url)
    
    if parsed.scheme == "s3":
        # s3://bucket-name/path/to/file.png
        # parsed.netloc = "bucket-name", parsed.path = "/path/to/file.png"
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return _download_from_s3(bucket, key)
    
    elif parsed.scheme in ("http", "https"):
        return _download_from_http(image_url)
    
    else:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r}. "
            f"Use 's3://...' or 'http(s)://...'."
        )


def _download_from_s3(bucket: str, key: str) -> str:
    """
    S3/MinIO에서 이미지 다운로드.
    
    settings의 자격증명(.env)을 사용해 인증한 뒤 다운로드.
    """
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint or None,   # MinIO일 때만 설정, AWS면 None
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    
    # 원본 확장자 보존 (.png, .jpg 등)
    suffix = Path(key).suffix or ".png"
    
    # 임시 파일 생성 (delete=False: 우리가 명시적으로 삭제할 때까지 유지)
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="floorplan_",
        delete=False,
    ) as temp_file:
        local_path = temp_file.name
    
    try:
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{key} -> {local_path}")
        return local_path
    except ClientError as e:
        # 실패 시 임시 파일 삭제
        Path(local_path).unlink(missing_ok=True)
        raise IOError(f"S3 download failed: {bucket}/{key}: {e}")


def _download_from_http(url: str) -> str:
    """HTTP(S) URL에서 이미지 다운로드."""
    # URL 경로에서 확장자 추출
    suffix = Path(urlparse(url).path).suffix or ".png"
    
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="floorplan_",
        delete=False,
    ) as temp_file:
        local_path = temp_file.name
    
    try:
        # stream=True: 대용량 파일 메모리 절약
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {url} -> {local_path}")
        return local_path
    except requests.RequestException as e:
        Path(local_path).unlink(missing_ok=True)
        raise IOError(f"HTTP download failed: {url}: {e}")


def cleanup_image(local_path: str) -> None:
    """
    임시 파일 정리.
    
    detector 사용 후 호출하여 디스크 공간 절약.
    이미 삭제됐거나 경로가 잘못돼도 에러 없이 처리.
    """
    try:
        Path(local_path).unlink(missing_ok=True)
        logger.debug(f"Cleaned up {local_path}")
    except OSError as e:
        logger.warning(f"Failed to cleanup {local_path}: {e}")