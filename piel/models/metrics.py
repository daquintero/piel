from piel.types import ScalarMetric, NumericalTypes


def value(value: NumericalTypes = None, **kwargs) -> ScalarMetric:
    return ScalarMetric(value=value, **kwargs)


def min_max(
    min: NumericalTypes = None, max: NumericalTypes = None, **kwargs
) -> ScalarMetric:
    return ScalarMetric(min=min, max=max, **kwargs)
