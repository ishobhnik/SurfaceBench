# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a LLMSR experiments
."""
from __future__ import annotations

import dataclasses
from typing import Type, Optional
import os
import warnings

# Local imports from llmsr package
from methods.llmsr import sampler
from methods.llmsr import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2 
    num_islands: int = 10 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000 


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt drawn from LLM for each prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout in seconds. Crucial for complex surfaces.
       use_api (bool): Flag to indicate whether to use an external LLM API (True) or a local server (False).
       api_model (str): The name of the LLM model to use (e.g., "gpt-4o", "gemini-pro").
       api_key (Optional[str]): The API key for the LLM. Can also be set via environment variable.
       api_url (Optional[str]): Base URL for the LLM API (e.g., "https://api.openai.com/v1").
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 60 # Increased default for surface evaluation
    use_api: bool = True
    api_model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_url: str = "https://api.openai.com/v1"

    def __post_init__(self):
        if self.api_key is None and self.use_api and "openai" in self.api_url:
            object.__setattr__(self, 'api_key', os.getenv("OPENAI_API_KEY"))
        elif self.api_key is None and self.use_api and "huggingface" in self.api_url:
            object.__setattr__(self, 'api_key', os.getenv("HF_API_KEY"))
        if self.use_api and not self.api_key:
            warnings.warn("API usage is enabled, but no API key is set. LLM calls will likely fail.")
            
@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]