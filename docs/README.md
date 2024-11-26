# BioNeuralNet Documentation

Welcome to the BioNeuralNet documentation! This directory contains all the source files and configurations necessary to build and maintain the project's documentation using [Sphinx](https://www.sphinx-doc.org/).

## **Contents**

- **source/**: Contains the reStructuredText (`.rst`) files and Sphinx configuration (`conf.py`) used to generate the documentation.
- **build/**: The output directory where the generated documentation (e.g., HTML files) will be placed. **This directory should not be modified manually.**

## **Building the Documentation**

To build the documentation locally, follow these steps:

1. **Navigate to the `docs/` Directory:**

    ```bash
    cd docs
    ```

2. **Activate the Virtual Environment:**

    Ensure that you have the project's virtual environment activated. If not, activate it:

    ```bash
    # On Unix or MacOS:
    source ../venv/bin/activate

    # On Windows:
    ../venv/Scripts/activate
    ```

3. **Install Documentation Dependencies:**

    If you haven't installed the development dependencies yet, do so:

    ```bash
    pip install -r requirements-dev.txt
    ```

4. **Build the Documentation:**

    Use Sphinx's Makefile to build the HTML documentation:

    ```bash
    make html
    ```

    **Alternatively**, you can use the Sphinx build command directly:

    ```bash
    sphinx-build -b html source/ build/html/
    ```

5. **View the Documentation:**

    Open the generated HTML files in your web browser:

    ```bash
    open build/html/index.html  # On MacOS
    xdg-open build/html/index.html  # On Linux
    start build\html\index.html  # On Windows
    ```

## **Contributing to the Documentation**

Contributions to the documentation are highly encouraged! To contribute:

1. **Fork the Repository:**

    Click the "Fork" button on the repository's GitHub page to create your own copy.

2. **Clone Your Fork:**

    ```bash
    git clone https://github.com/https://github.com/UCD-BDLab/BioNeuralNet.git
    cd bioneuralnet
    ```

3. **Create a New Branch:**

    ```bash
    git checkout -b update-documentation
    ```

4. **Make Your Changes:**

    Edit or add `.rst` files in the `docs/source/` directory as needed.

5. **Build and Preview Locally:**

    Follow the [Building the Documentation](#building-the-documentation) steps to ensure your changes render correctly.

6. **Commit and Push Your Changes:**

    ```bash
    git add docs/source/your_changes.rst
    git commit -m "Update documentation: Added section on XYZ"
    git push origin update-documentation
    ```

7. **Open a Pull Request:**

    Navigate to your fork on GitHub and open a pull request against the main repository's `main` branch.

## **Documentation Structure**

Here's an overview of the key documentation files:

- **`source/index.rst`**: The main entry point of the documentation, containing the table of contents.
- **`source/installation.rst`**: Instructions on how to install the BioNeuralNet package.
- **`source/usage.rst`**: Guides on how to use the package's features.
- **`source/api_reference.rst`**: Detailed API documentation generated from docstrings.
- **`source/tutorials.rst`**: Step-by-step tutorials for various use cases.
- **`source/faq.rst`**: Frequently Asked Questions to assist users.

## **Additional Resources**

- **Sphinx Documentation**: [https://www.sphinx-doc.org/](https://www.sphinx-doc.org/)
- **Sphinx Quickstart**: Useful for initializing Sphinx projects. [https://www.sphinx-doc.org/en/master/usage/quickstart.html](https://www.sphinx-doc.org/en/master/usage/quickstart.html)
- **Read the Docs Theme**: Enhances the visual appearance of your documentation. [https://sphinx-rtd-theme.readthedocs.io/](https://sphinx-rtd-theme.readthedocs.io/)




