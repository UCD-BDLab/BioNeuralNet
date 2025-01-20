# #!/usr/bin/env python3

# import os
# import platform
# import subprocess
# import sys
# import shutil
# import re
# import argparse

# def run_command(command, shell=False, check=True, env=None):
#     try:
#         if isinstance(command, list):
#             cmd_str = ' '.join(command)
#         else:
#             cmd_str = command
#         print(f"Running command: {cmd_str}")
#         subprocess.run(command, shell=shell, check=check, env=env)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with exit code {e.returncode}: {cmd_str}")
#         sys.exit(e.returncode)
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         sys.exit(1)

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Fast Install Script for BioNeuralNet")
#     parser.add_argument('-dev', '--dev', action='store_true', help='Install development dependencies')
#     parser.add_argument('-cuda', '--cuda', action='store_true', help='Install CUDA-enabled PyTorch if supported')
#     parser.add_argument('--cuda-version', type=str, choices=['11.8', '12.1', '12.4'], help='Specify CUDA version to install (supported: 11.8, 12.1, 12.4)')
#     return parser.parse_args()

# def is_command_available(command):
#     return shutil.which(command) is not None

# def detect_cuda_version():
#     try:
#         output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT)
#         output = output.decode('utf-8')
#         match = re.search(r'release\s+(\d+\.\d+)', output)
#         if match:
#             return match.group(1)
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         pass
#     return None

# def install_r_system_dependencies(os_type):
#     print("\nInstalling system dependencies for R...")
#     if os_type == "Linux":
#         run_command(["sudo", "apt-get", "update"])
#         run_command([
#             "sudo", "apt-get", "install", "-y",
#             "libxml2-dev",
#             "libcurl4-openssl-dev",
#             "libssl-dev",
#             "libpng-dev",
#             "r-base"
#         ])
#     elif os_type == "Darwin":
#         if not is_command_available("R"):
#             print("R is not installed. Installing R via Homebrew...")
#             if not is_command_available("brew"):
#                 print("Homebrew not found. Please install Homebrew from https://brew.sh/")
#                 sys.exit(1)
#             run_command(["brew", "install", "r"])
#     elif os_type == "Windows":
#         if not is_command_available("Rscript"):
#             print("R is not installed. Please install R from https://cran.r-project.org/bin/windows/base/")
#             sys.exit(1)
#     else:
#         print(f"Unsupported OS: {os_type}. Please install R and its dependencies manually.")
#         sys.exit(1)
#     print("System dependencies for R installed successfully.\n")

# def install_r_dependencies(os_type):
#     print("Installing R dependencies...")
#     r_packages = [
#         'dplyr',
#         'SmCCNet',
#         'jsonlite'
#     ]
#     bioc_packages = [
#         'impute',
#         'preprocessCore',
#         'GO.db',
#         'AnnotationDbi'
#     ]

#     if r_packages:
#         r_cran_packages = ', '.join([f'"{pkg}"' for pkg in r_packages])
#         r_cran_install = f'install.packages(c({r_cran_packages}), repos="https://cran.r-project.org")'
#         run_command(["Rscript", "-e", r_cran_install])

#     if bioc_packages:
#         run_command([
#             "Rscript",
#             "-e",
#             "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager', repos='https://cran.r-project.org')"
#         ])
#         bioc_install_packages = ', '.join([f'"{pkg}"' for pkg in bioc_packages])
#         bioc_install = f'BiocManager::install(c({bioc_install_packages}), update=FALSE, ask=FALSE)'
#         run_command(["Rscript", "-e", bioc_install])

#     run_command([
#         "Rscript",
#         "-e",
#         "install.packages('WGCNA', repos='https://cran.r-project.org')"
#     ])

#     print("R dependencies installed successfully.\n")

# def create_virtual_env(venv_path):
#     print(f"Creating virtual environment at {venv_path}...")
#     run_command([sys.executable, "-m", "venv", venv_path])
#     print("Virtual environment created successfully.\n")

# def install_python_dependencies(requirements, venv_path, os_type):
#     print("Installing Python dependencies...")
#     if os_type == "Windows":
#         pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
#     else:
#         pip_executable = os.path.join(venv_path, "bin", "pip")

#     run_command([pip_executable, "install", "--upgrade", "pip", "setuptools", "wheel", "--no-cache-dir", "-q"])
#     run_command([pip_executable, "install", "-r", requirements, "--no-cache-dir", "-q"])

#     print("Python dependencies installed successfully.\n")

# def get_installed_torch_version(venv_path, os_type):
#     if os_type == "Windows":
#         python_executable = os.path.join(venv_path, "Scripts", "python.exe")
#     else:
#         python_executable = os.path.join(venv_path, "bin", "python")

#     try:
#         version = subprocess.check_output(
#             [python_executable, "-c", "import torch; print(torch.__version__)"],
#             stderr=subprocess.STDOUT
#         ).decode('utf-8').strip()
#         return version
#     except subprocess.CalledProcessError:
#         print("Failed to retrieve PyTorch version.")
#         return None

# def get_pytorch_cuda_version(venv_path, os_type):
#     if os_type == "Windows":
#         python_executable = os.path.join(venv_path, "Scripts", "python.exe")
#     else:
#         python_executable = os.path.join(venv_path, "bin", "python")

#     try:
#         cuda_available = subprocess.check_output(
#             [python_executable, "-c", "import torch; print(torch.version.cuda)"],
#             stderr=subprocess.STDOUT
#         ).decode('utf-8').strip()
#         if cuda_available and cuda_available.lower() != "none":
#             return cuda_available
#         else:
#             return "cpu"
#     except subprocess.CalledProcessError:
#         print("Failed to retrieve PyTorch CUDA version.")
#         return "cpu"

# def get_pyg_os_arch(os_type):
#     if os_type == "Linux":
#         return "linux_x86_64"
#     elif os_type == "Windows":
#         return "cp310-cp310-win_amd64"
#     elif os_type == "Darwin":
#         arch = platform.machine()
#         if arch == "x86_64":
#             return "macosx_10_15_x86_64"
#         elif arch in ["arm64", "aarch64"]:
#             return "macosx_10_15_arm64"
#         else:
#             print(f"Unsupported macOS architecture: {arch}")
#             sys.exit(1)
#     else:
#         print(f"Unsupported OS: {os_type}. Cannot determine PyG wheel architecture.")
#         sys.exit(1)

# def install_torch(install_type, cuda_version, venv_path, os_type):
#     print("Installing PyTorch...")
#     if os_type == "Windows":
#         pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
#     else:
#         pip_executable = os.path.join(venv_path, "bin", "pip")

#     if install_type == "2":
#         if not cuda_version:
#             cuda_version = detect_cuda_version()
#             if not cuda_version:
#                 print("No CUDA installation detected. Proceeding with CPU-only PyTorch installation.")
#                 install_type = "1"

#         if install_type == "2" and cuda_version:
#             supported_cuda_versions = ["11.8", "12.1", "12.4"]
#             if cuda_version not in supported_cuda_versions:
#                 print(f"Unsupported CUDA version detected: {cuda_version}. Supported versions are {supported_cuda_versions}.")
#                 sys.exit(1)

#             torch_index_url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
#             print(f"Installing PyTorch with CUDA {cuda_version} support...")
#             run_command([
#                 pip_executable, "install",
#                 "torch", "torchvision", "torchaudio",
#                 "--index-url", torch_index_url,
#                 "--no-cache-dir",
#                 "-q"
#             ])
#     else:
#         torch_index_url = "https://download.pytorch.org/whl/cpu"
#         print("Installing PyTorch CPU-only...")
#         run_command([
#             pip_executable, "install",
#             "torch", "torchvision", "torchaudio",
#             "--index-url", torch_index_url,
#             "--no-cache-dir",
#             "-q"
#         ])

#     print("PyTorch installed successfully.\n")

# def install_extra_packages(venv_path, os_type):
#     print("Installing torch-scatter, torch-sparse, and torch-geometric...")
#     if os_type == "Windows":
#         pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
#     else:
#         pip_executable = os.path.join(venv_path, "bin", "pip")

#     torch_version = get_installed_torch_version(venv_path, os_type)
#     pytorch_cuda_version = get_pytorch_cuda_version(venv_path, os_type)

#     if not torch_version:
#         print("PyTorch is not installed correctly. Cannot proceed with installing related packages.")
#         sys.exit(1)

#     print(f"Installed PyTorch version: {torch_version}")
#     print(f"Installed CUDA version: {pytorch_cuda_version}\n")

#     if pytorch_cuda_version == "cpu":
#         pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
#     else:
#         cuda_no_dot = pytorch_cuda_version.replace('.', '')
#         pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_no_dot}.html"

#     print(f"Using PyG wheel URL: {pyg_url}\n")

#     run_command([
#         pip_executable, "install",
#         "torch-scatter", "torch-sparse",
#         "-f", pyg_url,
#         "--no-cache-dir",
#         "-q"
#     ])

#     run_command([
#         pip_executable, "install",
#         "torch-geometric",
#         "--no-cache-dir",
#         "-q"
#     ])

#     print("torch-scatter, torch-sparse, and torch-geometric installed successfully.\n")

# def install_dev_dependencies(dev_requirements, venv_path, os_type):
#     print("Installing development dependencies...")
#     install_python_dependencies(dev_requirements, venv_path, os_type)
#     print("Development dependencies installed successfully.\n")

# def main():
#     args = parse_arguments()

#     install_type = "1"
#     if args.cuda:
#         install_type = "2"

#     cuda_version = None
#     if args.cuda_version:
#         cuda_version = args.cuda_version

#     install_dev = args.dev

#     required_python_version = (3, 10)
#     current_python_version = sys.version_info
#     if current_python_version < required_python_version:
#         print(f"Python {required_python_version[0]}.{required_python_version[1]} or higher is required.")
#         print(f"Current Python version: {current_python_version.major}.{current_python_version.minor}")
#         sys.exit(1)
#     else:
#         print(f"Python version {current_python_version.major}.{current_python_version.minor} detected. Proceeding...\n")

#     os_type = platform.system()
#     print(f"Detected Operating System: {os_type}\n")

#     current_dir = os.getcwd()
#     scripts_dir = os.path.join(current_dir, "scripts")
#     requirements_txt = os.path.join(current_dir, "requirements.txt")
#     dev_requirements = os.path.join(scripts_dir, "requirements-dev.txt")
#     venv_path = os.path.join(current_dir, "bioneuralnet-env")

#     install_r_system_dependencies(os_type)

#     if not os.path.exists(venv_path):
#         create_virtual_env(venv_path)
#     else:
#         print(f"Virtual environment already exists at {venv_path}.\n")

#     install_python_dependencies(requirements_txt, venv_path, os_type)

#     if install_dev:
#         install_dev_dependencies(dev_requirements, venv_path, os_type)
#     else:
#         print("Skipping development dependencies installation.\n")

#     install_torch(install_type, cuda_version, venv_path, os_type)
#     install_extra_packages(venv_path, os_type)
#     install_r_dependencies(os_type)

#     print("\n--------------------------")
#     print("BioNeuralNet setup complete!")
#     print("--------------------------\n")
#     print("To activate the virtual environment, run:")
#     if os_type == "Windows":
#         print("bioneuralnet-env\\Scripts\\activate\n")
#     else:
#         print("source ./bioneuralnet-env/bin/activate\n")
#     print("To deactivate the virtual environment, run:")
#     print("deactivate\n")

# if __name__ == "__main__":
#     main()
