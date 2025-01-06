from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def calculate(self, reference_video_path: str, generated_video_path: str) -> float:
        pass
