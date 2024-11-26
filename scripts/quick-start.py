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
    print(f"Detected OS: {os_type}")

    # Step 1: Call setup-env.sh
    print("Running environment setup...")
    if os_type in ["Linux", "Darwin"]: 
        if not os.path.exists("setup-env.sh"):
            print("Error: 'setup-env.sh' not found in the current directory.")
            exit(1)
        execute_script("./setup-env.sh")
    elif os_type == "Windows":
        print("Windows detected. Please use a compatible script to setup Python environment.")
        exit(1)

    # Step 2: Call the appropriate R setup script
    print("Setting up R dependencies...")
    if os_type == "Windows":
        if not os.path.exists("setup-R.bat"):
            print("Error: 'setup-R.bat' not found in the current directory.")
            exit(1)
        execute_script("setup-R.bat")
    elif os_type in ["Linux", "Darwin"]:
        if not os.path.exists("setup-R.sh"):
            print("Error: 'setup-R.sh' not found in the current directory.")
            exit(1)
        execute_script("./setup-R.sh")
    else:
        print("Unsupported OS. Please install R dependencies manually.")
        exit(1)

    print("Quick-start completed successfully!")

if __name__ == "__main__":
    main()
