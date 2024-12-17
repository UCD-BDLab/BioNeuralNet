import os
import platform
import subprocess

def execute_script(script_path):
    """Execute the provided script."""
    try:
        subprocess.run(script_path, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")
        exit(1)

def main():
    os_type = platform.system()
    current_dir = os.getcwd()
    print(f"Detected OS: {os_type}")
    print(f"Current working directory: {current_dir}")

    # Step 1: Call setup-env.sh
    print("Running environment setup...")
    if os_type in ["Linux", "Darwin"]:
        script_path = os.path.join(current_dir, "scripts", "setup-env.sh")
        if not os.path.exists(script_path):
            print(f"Error: '{script_path}' not found.")
            exit(1)
        execute_script(f"./{script_path}")
    elif os_type == "Windows":
        print("Windows detected. Please use a compatible script to set up the Python environment.")
        exit(1)

    # Step 2: Call the appropriate R setup script
    print("Setting up R dependencies...")
    if os_type == "Windows":
        script_path = os.path.join(current_dir, "scripts", "setup-R.bat")
        if not os.path.exists(script_path):
            print(f"Error: '{script_path}' not found.")
            exit(1)
        execute_script(script_path)
    elif os_type in ["Linux", "Darwin"]:
        script_path = os.path.join(current_dir, "scripts", "setup-R.sh")
        if not os.path.exists(script_path):
            print(f"Error: '{script_path}' not found.")
            exit(1)
        execute_script(f"./{script_path}")
    else:
        print("Unsupported OS. Please install R dependencies manually.")
        exit(1)

    print("------------------------------------------------")
    print("BioNeuralNet quick-start completed successfully!")
    print("------------------------------------------------\n")
    print("To activate the virtual environment, run:")
    print("source .venv/bin/activate\n")
    print("To deactivate the virtual environment, run:")
    print("deactivate\n")

if __name__ == "__main__":
    main()
