from typing import List

from ..dataclasses import SEDTask, Equation


class BaseSearcher:
    def __init__(self, name) -> None:
        self._name = name

    def discover(self, task: SEDTask) -> List[Equation]:
        '''
        
        Return:
            equations
            aux
        '''
        raise NotImplementedError

    def __str__(self):
        return self._name