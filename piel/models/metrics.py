from piel.types import ScalarMetrics, NumericalTypes


def value(value: NumericalTypes = None, **kwargs) -> ScalarMetrics:
    return ScalarMetrics(value=value, **kwargs)


def min_max(
    min: NumericalTypes = None, max: NumericalTypes = None, **kwargs
) -> ScalarMetrics:
    return ScalarMetrics(min=min, max=max, **kwargs)
