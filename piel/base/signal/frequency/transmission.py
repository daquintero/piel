import numpy as np
from typing import Union, List


def get_phasor_length(phasor: Union[int, float, List[float], np.ndarray]) -> int:
    if isinstance(phasor, (list, np.ndarray)):
        return len(phasor)
    elif isinstance(phasor, (int, float)):
        return 1
    else:
        try:
            if isinstance(phasor.magnitude, (list, np.ndarray)):
                return len(phasor.magnitude)
            elif isinstance(phasor.magnitude, (int, float)):
                return 1
            else:
                raise ValueError(f"Unsupported PhasorType: {type(phasor.magnitude)}")
        except AttributeError:
            raise ValueError(f"Unsupported PhasorType: {type(phasor)}")
