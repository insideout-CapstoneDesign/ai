"""
모든 AI Detector의 공통 인터페이스.

이 모듈은 추상화 레이어를 정의한다. 모든 detector(텍스트, POI, 오브젝트, 노드/엣지)는
이 Detector 클래스를 상속받아 detect() 메서드를 구현해야 한다.

"""

from abc import ABC, abstractmethod
from typing import List

from app.schemas.analyze import Detection


class Detector(ABC):
    """
    AI Detector의 추상 베이스 클래스.
    
    모든 구체적인 detector(TextDetector, ObjectDetector 등)는
    이 클래스를 상속하여 detect() 메서드를 구현해야 한다.
    """

    # 모델 이름/버전 정보 (자식 클래스에서 오버라이드)
    name: str = "base"
    version: str = "v1.0"

    @abstractmethod
    def detect(self, image_path: str) -> List[Detection]:
        """
        이미지에서 객체를 감지하여 Detection 리스트를 반환한다.
        
        @abstractmethod 데코레이터가 붙은 메서드는 자식 클래스에서
        반드시 구현해야 한다. 안 하면 인스턴스 생성 시 에러 발생.
        
        Args:
            image_path: 분석할 이미지 파일의 경로 (또는 URL)
        
        Returns:
            감지된 객체들의 Detection 리스트
        """
        ...