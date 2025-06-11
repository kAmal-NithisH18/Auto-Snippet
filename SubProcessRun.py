import io
import contextlib

code = """
print('Hello')
print("Kamal Nithish")
import numpy as np
print(np.max([1,3]))
"""
def run_SubProcesscode(code):
    
    # Redirect stdout to capture print output
    output_buffer = io.StringIO()
    local_vars = {}

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {}, local_vars)  # empty globals, locals dict for security
        output = output_buffer.getvalue()
        return output,True

    except Exception as e:
        return e,False

if __name__ == "__main__":
    print(run_SubProcesscode(code)[0])