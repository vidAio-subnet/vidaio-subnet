from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.75)
    host: str = Field(default="localhost")
    port: int = Field(default=8201)
    upscaling_score_port: int = Field(default=8201)
    compression_score_port: int = Field(default=8202)
    upscaling_organics_score_port: int = Field(default=8203)
    compression_organics_score_port: int = Field(default=8204)
    vmaf_threshold: float = Field(default=0.5)
    vmaf_sample_count: int = Field(default=10)
    pieapp_threshold: float = Field(default=1.0)
    pieapp_sample_count: int = Field(default=4)
    max_performance_records: int = Field(default=10)
    synthetics_hours_threshold: int = Field(default=5)
    synthetics_select_probability: float = Field(default=0.7)
