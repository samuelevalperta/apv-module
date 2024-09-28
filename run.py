#!/usr/bin/python3

import numpy as np # This import used for evaluating numpy code written in parameters.py
import os
import sys
import re
import subprocess

def generate_header(model_path, param_folder):
    header_path = os.path.join('src', 'config.h')
    pattern = re.compile(r'dx\[(\d+)\] = (.*)')

    function_code = """
void f(double *x, double dx[]) {
"""

    with open(model_path, 'r') as file:
        for line in file:
            match = pattern.search(line.strip())
            if match:
                index = match.group(1)
                equation = match.group(2)
                function_code += f"  dx[{index}] = {equation};\n"

    function_code += "}\n"

    print(f"[+] Generated f(t, x) based on {model_path}")
    
    pattern = re.compile(r'([a-z0-9]+) = (.*)')
    costants_code = ""
    
    with open(f"{param_folder}/parameters.py", 'r') as file:
        for line in file:
            match = pattern.search(line.strip())
            if match:
                if match.group(1) == 'x0':
                    x0 = eval(match.group(2))
                else:
                    costants_code += f"#define {match.group(1)} {eval(match.group(2))}\n"

    print(f"[+] Costants read from {os.path.basename(param_folder)}/parameters.py")
    
    if os.path.exists(header_path):
        os.remove(header_path)
        print(f"[-] Existing header removed: {header_path}")

    with open(f"{header_path}", 'w') as f:
        f.write(costants_code)
        f.write(f"#define VAR_NUMBER {len(x0[0])}\n")
        f.write(function_code)

    print(f"[+] New header generated: {header_path}")


def build_extension(param_folder):
    build_dir = 'build'
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    try:
        subprocess.run(['python3', 'setup.py', 'build_ext', f'--build-lib={param_folder}'], check=True)
        print("[+] C extension compiled successfully using setuptools.")
    except subprocess.CalledProcessError as e:
        print(f"[!] An error occurred during the build: {e}")
       

def generate_new_model(model_path, param_folder):
    with open(model_path, 'r') as file:
        lines = file.readlines()[2:] # Skip shebang

    filtered_lines = ["#!/usr/bin/env python3\n\n"]

    # Import compiled module

    import_string = f"""
extension_dir = os.path.abspath("{param_folder}")
sys.path.append(extension_dir)
def import_extension():
    try:
        import euler_solver
        return euler_solver
    except ImportError as e:
        print("Error importing C extension:", e)
        return None
euler_solver = import_extension()\n
"""
    for line in lines:
        if line.strip().startswith("from scipy.integrate import solve_ivp"):
            filtered_lines += import_string
        elif line.strip().startswith("x = solve_"):
            filtered_lines += ["""
    class Bunch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    x = Bunch(t = np.arange(ti, tf, dt), y = euler_solver.getSolutionMatrix(x0[i]))
"""]
        else:
            filtered_lines += line

    parent_dir = os.path.dirname(model_path)
    new_model_path = os.path.join(parent_dir, f"fast-{os.path.basename(model_path)}")
    with open(new_model_path, 'w') as outfile:
        outfile.writelines(filtered_lines)

    print(f"[+] New script saved to {new_model_path}")

def main(model_path, param_folder):
    generate_header(model_path, param_folder)
    build_extension(param_folder)
    generate_new_model(model_path, param_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <model.py> <paramFolder>")
        sys.exit(1)
    
    arg1 = os.path.abspath(sys.argv[1])
    arg2 = os.path.abspath(sys.argv[2])
    
    main(arg1, arg2)
