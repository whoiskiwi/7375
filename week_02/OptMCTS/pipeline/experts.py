from core.llm import call_llm
from core.executor import execute_code, is_valid


def chain_of_experts(question):
    interpretation = call_llm(
        "You are an optimization problem expert. Please understand this problem and extract the key information:" + question
    )
    formulation = call_llm(
        "Based on the following understanding, write the mathematical formula (variables, objective, constraints):" + interpretation
    )
    code = call_llm(
        "Based on the following mathematical formulation, write Python code using scipy to solve this optimization problem. "
        "Output ONLY executable Python code, no explanation, no markdown. "
        "Print the optimal objective value at the end.\n\n" + formulation
    )
    result = execute_code(code)

    for retry in range(3):
        if not is_valid(result):
            code = call_llm(
                "The following Python code failed with this error:\n" + result["error"][:500]
                + "\n\nOriginal code:\n" + code
                + "\n\nPlease fix the code. Output ONLY executable Python code, no explanation, no markdown. "
                "Print the optimal objective value at the end."
            )
            result = execute_code(code)
        else:
            break

    return result
