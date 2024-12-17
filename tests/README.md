# BioNeuralNet Testing Docs

Welcome to the BioNeuralNet testing docs. This documentation provides an overview of the testing strategy, explains each test module, offers instructions on running the tests, and provides guidelines for writing new tests.

## **Table of Contents**

- [Overview](#overview)
- [Testing Framework](#testing-framework)
- [Pre-Commit Hooks](#pre-commit-hooks)
- [Directory Structure](#directory-structure)
- [Test Modules](#test-modules)
  - [test_gnns.py](#test_gnnspy)
  - [test_node2vec.py](#test_node2vecpy)
  - [test_utils.py](#test_utilspy)
  - [test_smccnet.py](#test_smccnetpy)
  - [test_wgcna.py](#test_wgcnapy)
  - [test_hierarchical.py](#test_hierarchicalpy)
  - [test_pagerank.py](#test_pagerankpy)
- [Running the Tests](#running-the-tests)
  - [Using Pytest](#using-pytest)
  - [Viewing Coverage Reports](#viewing-coverage-reports)
  - [Continuous Integration](#continuous-integration)
- [Writing New Tests](#writing-new-tests)
  - [Best Practices](#best-practices)
  - [Using Fixtures](#using-fixtures)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## **Overview**

The BioNeuralNet testing suite is designed to ensure the reliability and correctness of the package's components. By systematically verifying each module's functionality, we aim to maintain high code quality and facilitate future developments.

**Key Objectives:**

- **Unit Testing:** Verify the functionality of individual components and functions.
- **Integration Testing:** Ensure that different modules interact correctly.
- **Regression Testing:** Prevent new changes from introducing bugs.

---

## **Testing Framework**

We utilize **`pytest`** as our primary testing framework due to its simplicity, scalability, and rich feature set.

**Advantages of `pytest`:**

- **Simple Syntax:** Write tests as simple functions without needing to inherit from classes.
- **Powerful Fixtures:** Manage setup and teardown with reusable fixtures.
- **Rich Assertions:** Use Python's `assert` statements with detailed introspection.
- **Extensibility:** Leverage numerous plugins for additional functionalities like coverage reporting, mocking, etc.

**Installation:**

Ensure that `pytest` and `pytest-cov` are installed:

```bash
pip install pytest pytest-cov
```

---

## **Pre-Commit Hooks**

Pre-commit hooks automate the execution of scripts before a commit is finalized. In BioNeuralNet, we've configured pre-commit hooks to run tests, enforce code formatting, and perform linting. This ensures that only high-quality and passing code is committed to the repository.

### **a. Purpose of Pre-Commit Hooks**

- **Automated Testing:** Runs the test suite to catch issues early.
- **Code Formatting:** Ensures code adheres to style guidelines (e.g., using Black).
- **Linting:** Detects potential errors and enforces coding standards (e.g., using Flake8).

### **b. Setting Up Pre-Commit Hooks**

1. **Install Pre-Commit:**

   Ensure that `pre-commit` is installed in your development environment. It's included in `requirements-dev.txt`, so you can install it via pip:

   ```bash
   pip install -r requirements-dev.txt
   ```

   Alternatively, install it directly:

   ```bash
   pip install pre-commit
   ```

2. **Create Configuration File:**

   A `.pre-commit-config.yaml` file should be present in the root directory of your project. Here's an example configuration tailored for BioNeuralNet:

   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.3.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-added-large-files

     - repo: https://github.com/psf/black
       rev: 23.3.0
       hooks:
         - id: black
           language_version: python3

     - repo: https://github.com/PyCQA/flake8
       rev: 6.0.0
       hooks:
         - id: flake8
           additional_dependencies: [flake8>=6.0.0]

     - repo: local
       hooks:
         - id: run-tests
           name: Run Tests
           entry: pytest
           language: system
           types: [python]
   ```

3. **Install the Pre-Commit Hooks:**

   Run the following command in your terminal to install the hooks:

   ```bash
   pre-commit install
   ```

   This command sets up the Git hooks to trigger the specified checks before each commit.

4. **Optional: Run Hooks on All Files:**

   To apply the hooks to all files (useful for initial setup or when hooks have been updated), execute:

   ```bash
   pre-commit run --all-files
   ```

### **c. Benefits of Pre-Commit Hooks**

- **Immediate Feedback:** Developers receive instant feedback on code quality and test results before committing.
- **Consistent Codebase:** Automated formatting and linting ensure a consistent code style across the project.
- **Prevent Broken Commits:** Tests must pass for a commit to succeed, reducing the likelihood of introducing bugs.

### **d. Important Considerations**

- **Hook Bypass:** While pre-commit hooks are powerful, they can be bypassed using force commits. Relying solely on pre-commit hooks is not recommended; always use CI pipelines as an additional safeguard.
- **Collaborator Setup:** Ensure all collaborators install and configure pre-commit hooks by running `pre-commit install`. You can automate this step using setup scripts or include instructions in the project's main `README.md`.

---

## **Test Modules**

Each test module corresponds to a specific component or functionality within the BioNeuralNet package. Below is a brief overview of each test module and its purpose.

- ### **test_gnns.py**

  **Purpose:**  
  Tests the Graph Neural Network (GNN) embedding models (`GCN`, `GAT`, `SAGE`, `GIN`) within the `network_embedding` module.

  **Key Tests:**

  - **Initialization Tests:** Ensure that each GNN model initializes correctly with given configurations.
  - **Forward Pass Tests:** Verify that the forward pass produces outputs of expected shapes.
  - **Invalid Input Tests:** Confirm that models raise appropriate exceptions when provided with invalid inputs.

- ### **test_node2vec.py**

  **Purpose:**  
  Tests the `Node2VecEmbedding` class responsible for generating Node2Vec embeddings.

  **Key Tests:**

  - **Initialization Tests:** Ensure that the `Node2VecEmbedding` class initializes correctly with given configurations.
  - **Run Method Tests:** Verify that embeddings are generated correctly and returned in the expected format.
  - **Error Handling Tests:** Confirm that appropriate exceptions are raised when required inputs are missing or invalid.

- ### **test_utils.py**

  **Purpose:**  
  Tests utility functions within the `utils` module, such as file searching and path validation.

  **Key Tests:**

  - **find_files Tests:** Ensure that the `find_files` function correctly identifies files based on glob patterns.
  - **validate_paths Tests:** Verify that `validate_paths` correctly validates the existence of provided paths and handles missing paths appropriately.

- ### **test_smccnet.py**

  **Purpose:**  
  Tests the `SmCCNet` class responsible for generating adjacency matrices using the SmCCNet algorithm.

  **Key Tests:**

  - **Initialization Tests:** Ensure that the `SmCCNet` class initializes correctly with given configurations.
  - **Run Method Tests:** Verify that adjacency matrices are generated correctly and returned in the expected format.
  - **Error Handling Tests:** Confirm that appropriate exceptions are raised when required input files are missing or configurations are invalid.

- ### **test_wgcna.py**

  **Purpose:**  
  Tests the `WGCNA` class responsible for generating adjacency matrices using the WGCNA algorithm.

  **Key Tests:**

  - **Initialization Tests:** Ensure that the `WGCNA` class initializes correctly with given configurations.
  - **Run Method Tests:** Verify that adjacency matrices are generated correctly and returned in the expected format.
  - **Error Handling Tests:** Confirm that appropriate exceptions are raised when required input files are missing or configurations are invalid.

- ### **test_hierarchical.py**

  **Purpose:**  
  Tests the `HierarchicalClustering` class responsible for performing agglomerative hierarchical clustering.

  **Key Tests:**

  - **Data Loading Tests:** Ensure that the adjacency matrix is loaded correctly from a CSV file.
  - **Clustering Tests:** Verify that the clustering algorithm runs successfully and produces expected results.
  - **Result Saving Tests:** Confirm that cluster labels and adjacency matrices are saved correctly.
  - **Error Handling Tests:** Check that appropriate exceptions are raised when required inputs are missing or configurations are invalid.

- ### **test_pagerank.py**

  **Purpose:**  
  Tests the `PageRank` class responsible for clustering nodes based on personalized PageRank.

  **Key Tests:**

  - **Data Loading Tests:** Ensure that the graph, omics data, and phenotype data are loaded correctly.
  - **Personalization Vector Tests:** Verify that the personalization vector is generated correctly.
  - **PageRank Execution Tests:** Ensure that the PageRank algorithm runs successfully and produces expected results.
  - **Sweep Cut Tests:** Confirm that the sweep cut method identifies clusters correctly.
  - **Result Saving Tests:** Check that clustering results are saved correctly.
  - **Error Handling Tests:** Verify that appropriate exceptions are raised when required inputs are missing or invalid.

---

## **Running the Tests**

### **Using Pytest**

1. **Navigate to the Project Root:**

   Ensure you're in the root directory of your project where the `tests/` directory resides.

   ```bash
   cd bioneuralnet
   ```

2. **Run All Tests:**

   Execute all tests within the `tests/` directory.

   ```bash
   pytest
   ```

3. **Run Tests with Verbose Output:**

   For more detailed output, use the `-v` flag.

   ```bash
   pytest -v
   ```

4. **Run Specific Test Module:**

   To run tests from a specific module, specify the path to the test file.

   ```bash
   pytest tests/test_gnns.py
   ```

### **Viewing Coverage Reports**

1. **Install Coverage Plugin:**

   Ensure `pytest-cov` is installed.

   ```bash
   pip install pytest-cov
   ```

2. **Run Tests with Coverage:**

   Generate a coverage report to see which parts of your code are exercised by tests.

   ```bash
   pytest --cov=bioneuralnet tests/
   ```

3. **Generate an HTML Coverage Report:**

   For a detailed, navigable coverage report, generate an HTML report.

   ```bash
   pytest --cov=bioneuralnet --cov-report=html tests/
   ```

   Open the generated `htmlcov/index.html` in your browser to view the coverage details.

### **Continuous Integration**

BioNeuralNet is already set up with a GitHub Actions workflow (`.github/workflows/python-app.yml`) that runs the tests automatically on code pushes and pull requests. 

**Key Features of the CI Workflow:**

- **Multi-Python Version Testing:** Tests run against multiple Python versions (e.g., 3.8, 3.9, 3.10) to ensure compatibility.
- **Dependency Installation:** Installs base, development, and CPU-specific dependencies.
- **Linting and Formatting Checks:** Runs Flake8 and Black to enforce code quality and style.
- **Test Execution:** Runs the entire test suite and generates coverage reports.
- **Coverage Reporting:** Uploads coverage reports to Codecov for analysis.

**Example Workflow Steps:**

```yaml
name: BioNeuralNet

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:

    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        # Install CPU-specific dependencies
        pip install -r requirements-cpu.txt

    - name: Lint with flake8
      run: |
        flake8 bioneuralnet tests docs

    - name: Format with black
      run: |
        black --check bioneuralnet tests docs

    - name: Run tests with pytest
      run: |
        pytest --cov=bioneuralnet --cov-report=xml tests/

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        files: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

---

## **Writing New Tests**

When adding new tests, adhere to the following guidelines to maintain consistency and effectiveness.

### **Best Practices**

1. **Test Naming Conventions:**

   - **Test Files:** Prefix with `test_` (e.g., `test_new_feature.py`).
   - **Test Functions:** Prefix with `test_` (e.g., `test_functionality()`).

2. **Isolate Tests:**

   - Each test should be independent, not relying on the state modified by other tests.

3. **Use Fixtures:**

   - Utilize `pytest` fixtures for setup and teardown operations.
   - Fixtures can provide reusable components like sample data, configurations, or mock objects.

4. **Descriptive Assertions:**

   - Use clear and descriptive assertion messages to facilitate easier debugging.

   ```python
   assert result == expected, f"Expected {expected}, but got {result}"
   ```

5. **Test Edge Cases:**

   - Beyond typical scenarios, include tests for boundary conditions, invalid inputs, and exceptional cases.

### **Using Fixtures**

Fixtures help manage setup and teardown for your tests, promoting code reuse and reducing redundancy.

**Example:**

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_dataframe():
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }
    df = pd.DataFrame(data)
    return df
```

**Usage in Tests:**

```python
def test_dataframe_sum(sample_dataframe):
    total = sample_dataframe.sum()
    assert total['A'] == 6, "Sum of column A should be 6"
    assert total['B'] == 15, "Sum of column B should be 15"
```

---

## **Troubleshooting**

- **Tests Failing Unexpectedly:**

  - **Check Dependencies:** Ensure all required packages are installed and up-to-date.
  - **Review Test Data:** Verify that the sample data fixtures provide the correct and expected data.

- **Coverage Report Incomplete:**

  - **Ensure All Modules Are Imported:** If a module isn't imported in tests, it won't be covered.
  - **Use `__init__.py` Correctly:** Proper package initialization ensures that all modules are discoverable by `pytest`.

- **CI Workflow Issues:**

  - **Check GitHub Actions Logs:** Review the logs in the Actions tab of your GitHub repository for detailed error messages.
  - **Verify Configuration:** Ensure that `.github/workflows/python-app.yml` is correctly configured to install dependencies and run tests.

---

## **Contributing**

Contributions to the testing suite are welcome. To maintain consistency and quality, please follow these guidelines:

1. **Add New Tests in Appropriate Modules:**

   - If testing a new component, create a corresponding `test_new_component.py` file.

2. **Follow Naming Conventions:**

   - Prefix test files and functions with `test_`.

3. **Write Clear and Concise Tests:**

   - Each test should focus on a single aspect or functionality.

4. **Use Fixtures Appropriately:**

   - Reuse fixtures for common setups.

5. **Run Tests Locally Before Committing:**

   ```bash
   pytest
   ```

6. **Ensure Tests Pass in CI:**

   - After pushing changes, verify that all tests pass in the GitHub Actions workflow.

---

## **Conclusion**

A well-documented and comprehensive testing is vital for the success and reliability of the our package. By following the guidelines outlined in this document, you can ensure that your tests are effective, maintainable, and provide valuable feedback during development.

Feel free to reach out or open an issue if you encounter any challenges or have suggestions for improving the testing suite.
