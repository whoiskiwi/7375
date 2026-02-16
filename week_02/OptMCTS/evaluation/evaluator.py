import re

def extract_answer(output):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None

def evaluate(predicted, actual):
    if predicted is None:
        return False
    if abs(actual) > 1e-8:
        rel_error = abs(predicted - actual) / abs(actual)
        return rel_error < 0.1
    else:
        return abs(predicted - actual) < 1e-4