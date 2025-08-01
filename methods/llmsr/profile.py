# profile the experiment with tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict, Any, Optional, Union
import json
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import methods.llmsr.code_manipulation as code_manipulation

class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file (deprecated/unused in current main).
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples') if log_dir else None
        if self._json_dir:
            os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order: Optional[int] = None
        self._cur_best_program_score: float = -np.inf 
        self._cur_best_program_str: Optional[str] = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0.0
        self._tot_evaluate_time = 0.0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        self._writer: Optional[SummaryWriter] = None
        if log_dir:
            try:
                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception as e:
                warnings.warn(f"Failed to initialize TensorBoard SummaryWriter at {log_dir}: {e}")
                self._writer = None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir or not self._writer:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

        if self._cur_best_program_str:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )

    def _write_json(self, programs: code_manipulation.Function):
        if not self._json_dir: return

        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        try:
            with open(path, 'w') as json_file:
                json.dump(content, json_file)
        except Exception as e:
            warnings.warn(f"Failed to write JSON log for sample {sample_order}: {e}")

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str


        if score is not None and np.isfinite(score):
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time
        if evaluate_time is not None:
            self._tot_evaluate_time += evaluate_time