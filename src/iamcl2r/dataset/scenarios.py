from typing import Callable, List, Optional, Union

import numpy as np
from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario


class NoContinualLearning(_BaseScenario):
    """No continual learning scenario.

    This scenario is used when no continual learning is used.
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: Optional[int] = None,
        transformations: Union[List[Callable], List[List[Callable]]] = None,
        random_seed: int = 1,
    ):
        self.cl_dataset = cl_dataset
        self._nb_tasks = self._setup(nb_tasks)
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=self._nb_tasks,
            transformations=transformations,
        )
        self._random_state = np.random.RandomState(seed=random_seed)

    def _setup(self, nb_tasks: Optional[int]) -> int:
        x, y, t = self.cl_dataset.get_data()

        if (
            nb_tasks is not None and nb_tasks > 0
        ):  # If the user wants a particular nb of tasks
            x, y, t = _duplicate_dataset(x, y, nb_tasks)
            self.dataset = (x, y, t)
        elif (
            t is not None
        ):  # Otherwise use the default task ids if provided by the dataset
            self.dataset = (x, y, t)
            nb_tasks = len(np.unique(t))
        else:
            raise Exception(
                f"The dataset ({self.cl_dataset}) doesn't provide task ids, "
                f"you must then specify a number of tasks, not ({nb_tasks}."
            )
        return nb_tasks


def _duplicate_dataset(x, y, nb_tasks):
    t = np.hstack([np.full_like(y, i) for i in range(nb_tasks)])
    x = np.hstack([x for _ in range(nb_tasks)])
    y = np.hstack([y for _ in range(nb_tasks)])
    return x, y, t
