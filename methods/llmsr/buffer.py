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

"""A multi-island experience buffer that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

import methods.llmsr.code_manipulation as code_manipulation
from methods.llmsr import config as config_lib


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """
    Reduces scores from multiple tests into a single score for a program.
    For SURFACEBENCH, we prioritize Chamfer Distance (lower is better),
    and return its negative mean to make higher scores better for evolution.
    """
    if not scores_per_test:
        return -float('inf')

    chamfer_scores = []
    for test_id, metrics_dict in scores_per_test.items():
        if 'chamfer_distance' in metrics_dict and np.isfinite(metrics_dict['chamfer_distance']):
            chamfer_scores.append(metrics_dict['chamfer_distance'])
        else:
            chamfer_scores.append(float('inf')) 
            
    if not chamfer_scores:
        return -float('inf')
    avg_chamfer = np.mean(chamfer_scores)
    return -avg_chamfer


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature.
    For SURFACEBENCH, this will use Chamfer and Hausdorff distances."""

    signature_elements = []
    for test_id in sorted(scores_per_test.keys()):
        metrics = scores_per_test[test_id]
        signature_elements.append(metrics.get('chamfer_distance', float('inf')))
        signature_elements.append(metrics.get('hausdorff_distance', float('inf')))
    return tuple(signature_elements)


@dataclasses.dataclass(frozen=True)
class Prompt:
    """ A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)

        self._last_reset_time: float = time.time()


    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)


    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs 
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)


    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs 
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        # Check island reset
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()


    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            if keep_islands_ids.size > 0: 
                founder_island_id = np.random.choice(keep_islands_ids)
                founder = self._best_program_per_island[founder_island_id]
                founder_scores = self._best_scores_per_test_per_island[founder_island_id]
                if founder:
                    self._register_program_in_island(founder, island_id, founder_scores)
                else:
                    print(f"Warning: No valid founder program for island {founder_island_id}. Skipping founder transfer.")
            else:
                print(f"Warning: No islands to keep. Starting island {island_id} from scratch.")


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0


    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)

        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1


    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        signatures = list(self._clusters.keys())
        if not signatures: 
            return "", 0

        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities, replace=False)
        chosen_signatures = [signatures[i] for i in idx]
        
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)[::-1]
        sorted_implementations = [implementations[i] for i in indices]
        
        next_version = len(sorted_implementations)
        return self._generate_prompt(sorted_implementations, next_version), next_version


    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function],
            next_version: int) -> str:
        """ Create a prompt containing a sequence of function `implementations`. """
        implementations_copy = copy.deepcopy(list(implementations)) 

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations_copy):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            
            if i > 0:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.'
                    f' Score: {_reduce_score(implementation.scores_per_test):.4f}'
                )
            
            if self._function_to_evolve in implementation.body:
                 implementation.body = code_manipulation.rename_function_calls(
                    implementation.body, self._function_to_evolve, new_function_name)

            versioned_functions.append(implementation)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        
        header_base_func = versioned_functions[-1] if versioned_functions else self._template.get_function(self._function_to_evolve)
        
        # A new Function object for the header to be completed by LLM
        new_function_header = dataclasses.replace(
            header_base_func,
            name=new_function_name,
            body='\n    # Your code here for the body of the function.\n', 
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.') if next_version > 0 else 'Initial version.',
        )
        versioned_functions.append(new_function_header)

        prompt_program = dataclasses.replace(self._template, functions=versioned_functions)
        
        return str(prompt_program)


class Cluster:
    """ A cluster of programs on the same island and with the same Signature. """

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score 
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorter programs."""
        if not self._programs:
            raise ValueError("Cannot sample from an empty cluster.")

        normalized_lengths = (np.array(self._lengths) - np.min(self._lengths)) / (
                np.max(self._lengths) - np.min(self._lengths) + 1e-6) 
        
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        
        if len(self._programs) == 1:
            return self._programs[0]
        else:
            idx = np.random.choice(len(self._programs), p=probabilities)
            return self._programs[idx]