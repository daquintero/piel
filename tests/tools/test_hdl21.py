import piel
import hdl21 as h

value = 1e4
converted_value = piel.convert_numeric_to_prefix(value)
assert 10 * h.prefix.K == converted_value
print(converted_value)
