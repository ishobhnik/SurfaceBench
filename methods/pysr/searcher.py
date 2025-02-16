from datetime import datetime
import os
from pathlib import Path
from bench.searchers.base import BaseSearcher
from bench.dataclasses import Equation, SEDTask, SearchResult
from pysr import PySRRegressor
# TensorBoardLoggerSpec

custom_loss = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
end
"""

class PySRSearcher(BaseSearcher):
    def __init__(self, name,
                 log_path, 
                 num_iterations=25, 
                 num_populations=10,
                 early_stopping_condition=None,
                 max_num_samples=2000) -> None:
        super().__init__(name)

        self.num_iterations = num_iterations
        self.early_stopping_condition = early_stopping_condition
        self.num_populations = num_populations
        self.max_num_samples = max_num_samples

        self.log_path = log_path

    def discover(self, task: SEDTask):
        info = task
        datasets = task.samples[:self.max_num_samples]

        var_names = task.symbols
        var_desc = task.symbol_descs
        var_desc = [f"{d} ({n})" for d,n in zip(var_desc, var_names)]
        temp_dir = "logs/temp/lasr_runs"
        
        
        # print(set_llm_options['var_order'])
        # print(set_llm_options)

        # Create a logger that writes to "logs/run*":
        # logger_spec = TensorBoardLoggerSpec(
        #     log_dir=os.path.join(set_llm_options['llm_recorder_dir'], 'run'),
        #     log_interval=1,  # Log every 10 iterations
        # )

        # Refer to https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/blob/lasr-experiments/experiments/model.py
        model = PySRRegressor(
            niterations=self.num_iterations,
            ncyclesperiteration=550,
            # ncycles_per_iteration=100,
            populations=self.num_populations,
            population_size=33,
            maxsize=30,
            binary_operators=["+", "*", "-", "/", "^"],
            unary_operators=[
                "exp",
                "log",
                "sqrt",
                "sin",
                "cos",
            ],
            loss_function=custom_loss,
            early_stop_condition=f"f(loss, complexity) = (loss < {format(float(self.early_stopping_condition), 'f')})"
            if self.early_stopping_condition
            else None,
            verbosity=1,
            temp_equation_file=True,
            tempdir=temp_dir,
            delete_tempfiles=True,
            weight_randomize=0.1,
            should_simplify=True,
            constraints={
                "sin": 10,
                "cos": 10,
                "exp": 20,
                "log": 20,
                "sqrt": 20,
                "pow": (-1, 20),
            },
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
            # logger_spec=logger_spec,
            progress=True,
        )

        X, y = datasets[:self.max_num_samples, 1:], datasets[:self.max_num_samples, 0]
        y = y.reshape(-1, 1)

        now = datetime.now()
        now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
        run_log_file=str(os.path.abspath(os.path.join(self.log_path, "run_logs", f"{task.name}_{now_str}.csv")))
        run_log_file = Path(run_log_file)
        run_log_file.parent.mkdir(exist_ok=True, parents=True)
        print(f"Logging to {run_log_file}")
        model.fit(X, y, run_log_file=str(run_log_file))

        best_equation = Equation(
            symbols=task.symbols,
            symbol_descs=task.symbol_descs,
            symbol_properties=task.symbol_properties,
            expression=str(model.sympy()),
            sympy_format=model.sympy(),
            lambda_format=model.predict
        )

        lasr_score = None
        for i, row in model.equations_.iterrows():
            # print(str(row.equation))
            if row.equation == best_equation.expression:
                lasr_score = row.score
                break

        return [SearchResult(
            equation=best_equation,
            aux={"lasr_score": lasr_score}
        )]