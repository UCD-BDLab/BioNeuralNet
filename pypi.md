## **Step-by-Step Guide to Publish BioNeuralNet to TestPyPI and PyPI**

### **Prerequisites**

Before proceeding, ensure you have the following:

1. **Accounts Created:**
   - **TestPyPI Account:** [TestPyPI](https://test.pypi.org/account/register/)
   - **PyPI Account:** [PyPI](https://pypi.org/account/register/)

2. **Package Files Ready:**
   - `setup.py`
   - `setup.cfg`
   - `MANIFEST.in`
   - `README.md`
   - Other necessary files (e.g., `requirements.txt`, `LICENSE`, etc.)

3. **Tools Installed:**
   - **Python 3.7 or Higher**
   - **Latest Versions of `setuptools` and `wheel`**
   - **`twine`**

4. **Version Control:**
   - Ensure your code is version-controlled (preferably with Git).

---

### **1. Verify Your `setup.py` and Other Configuration Files**

Ensure that your `setup.py`, `setup.cfg`, and other configuration files are correctly set up. A minimal `setup.py` might look like this:

```python
from setuptools import setup, find_packages

setup(
    name='bioneuralnet',
    version='0.1.0',
    author='Vicente Ramos',
    author_email='vicente.ramos@ucdenver.edu',
    description='A framework designed to analyze biological data through a series of computational steps. Capable of taking a network to a lower dimesional space. Opening the door for further analysis'
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/https://github.com/UCD-BDLab/BioNeuralNet',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        # List your package dependencies here
        'numpy',
        'pandas',
        'torch',
        # Add other dependencies
    ],
    include_package_data=True,  # To include files specified in MANIFEST.in
)
```

**Key Points:**

- **Versioning:** Start with an initial version like `0.1.0`. Follow [Semantic Versioning](https://semver.org/) for future updates.
- **Dependencies:** Ensure all dependencies are listed in `install_requires`.
- **`include_package_data=True`:** Ensures non-Python files specified in `MANIFEST.in` are included.

---

### **2. Install Required Tools**

Ensure you have the latest versions of `setuptools`, `wheel`, and `twine`. Upgrade them using `pip`:

```bash
pip install --upgrade setuptools wheel twine
```

---

### **3. Build Your Package**

Navigate to your project’s root directory (where `setup.py` is located) and build the source and wheel distributions.

```bash
python setup.py sdist bdist_wheel
```

**What This Does:**

- **`sdist`:** Creates a source distribution (a `.tar.gz` file).
- **`bdist_wheel`:** Creates a built distribution (a `.whl` file).

After running the command, you should see a `dist/` directory containing the distribution files:

```
bioneuralnet/
├── dist/
│   ├── bioneuralnet-0.1.0-py3-none-any.whl
│   └── bioneuralnet-0.1.0.tar.gz
```

---

### **4. Upload to TestPyPI**

Uploading to TestPyPI allows you to verify that your package uploads correctly and can be installed as expected without affecting the real PyPI repository.

**a. Upload Using Twine**

Use `twine` to upload your package to TestPyPI.

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

**b. Enter Credentials**

When prompted, enter your **TestPyPI** username and password.

**c. Verify the Upload**

After successful upload, navigate to [TestPyPI](https://test.pypi.org/project/bioneuralnet/) and search for your package to ensure it's listed.

---

### **5. Test Installation from TestPyPI**

Before uploading to the official PyPI, ensure that your package can be installed from TestPyPI.

**a. Create a Virtual Environment (Recommended)**

```bash
python3 -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate
```

**b. Install Your Package from TestPyPI**

Use `pip` to install your package from TestPyPI. Note that TestPyPI is a separate index from PyPI.

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bioneuralnet
```

**Explanation:**

- **`--index-url`:** Specifies the primary index to search (TestPyPI).
- **`--extra-index-url`:** Allows `pip` to fall back to the official PyPI for dependencies not found in TestPyPI.

**c. Verify Installation**

Try importing your package in Python to ensure it works as expected.

```python
import bioneuralnet
print(bioneuralnet.__version__)
```

---

### **6. Address Any Issues**

If you encounter any issues during installation or while using the package, address them by:

- **Reviewing Error Messages:** Identify and fix issues in your code or configuration.
- **Updating the Package Version:** Increment the version number in `setup.py` following semantic versioning (e.g., from `0.1.0` to `0.1.1`).
- **Rebuilding and Re-uploading:** After fixes, rebuild the package and re-upload to TestPyPI.

---

### **7. Upload to the Official PyPI**

Once satisfied with the TestPyPI installation and functionality, proceed to upload your package to the official PyPI repository.

**a. Upload Using Twine**

```bash
twine upload dist/*
```

**b. Enter Credentials**

When prompted, enter your **PyPI** username and password.

**c. Verify the Upload**

After successful upload, navigate to [PyPI](https://pypi.org/project/bioneuralnet/) and search for your package to ensure it's listed.

---

### **8. Install from PyPI and Final Verification**

Ensure that your package can be installed from the official PyPI.

**a. Create a New Virtual Environment (Recommended)**

```bash
python3 -m venv prod-env
source prod-env/bin/activate  # On Windows: prod-env\Scripts\activate
```

**b. Install Your Package from PyPI**

```bash
pip install bioneuralnet
```

**c. Verify Installation**

Import your package in Python to confirm it's working correctly.

```python
import bioneuralnet
print(bioneuralnet.__version__)
```

---

### **9. Keep Your Package Updated**

For future updates:

1. **Increment Version Number:** Update the `version` in `setup.py` following semantic versioning.
2. **Rebuild Distributions:**

   ```bash
   python setup.py sdist bdist_wheel
   ```

3. **Upload to TestPyPI and Repeat Testing Steps.**
4. **Upload to PyPI Once Verified.**

---

## **Additional Tips**

- **Check Package Name Availability:**
  - Ensure that `bioneuralnet` is available on both TestPyPI and PyPI. If the name is taken, consider using a unique name or your GitHub username as a prefix.

- **Automate Builds with GitHub Actions:**
  - Consider setting up CI/CD pipelines to automate building and uploading your package upon new releases.

- **Secure Your Credentials:**
  - Avoid hardcoding credentials. Use environment variables or keyring integrations with `twine`.

- **Maintain a Changelog:**
  - Keep a `CHANGELOG.md` to document changes, enhancements, and fixes across versions.

- **Provide Comprehensive Documentation:**
  - Host detailed documentation on platforms like [Read the Docs](https://readthedocs.org/) and link it in your `README.md`.

- **Monitor PyPI Releases:**
  - Use tools like [Release Drafter](https://github.com/release-drafter/release-drafter) to manage and draft releases.

---

## **Summary Checklist**

To ensure a smooth publishing process, follow this checklist:

1. **Before Uploading:**
   - [ ] Verify `setup.py` and other configuration files.
   - [ ] Update the package version appropriately.
   - [ ] Ensure all dependencies are listed.

2. **Building the Package:**
   - [ ] Install/upgrade `setuptools`, `wheel`, and `twine`.
   - [ ] Run `python setup.py sdist bdist_wheel`.

3. **Uploading to TestPyPI:**
   - [ ] Use `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`.
   - [ ] Verify upload on TestPyPI.

4. **Testing Installation:**
   - [ ] Create a virtual environment.
   - [ ] Install from TestPyPI and verify functionality.

5. **Uploading to PyPI:**
   - [ ] Use `twine upload dist/*`.
   - [ ] Verify upload on PyPI.

6. **Final Verification:**
   - [ ] Install from PyPI in a new virtual environment.
   - [ ] Confirm package functionality.

7. **Post-Upload:**
   - [ ] Update documentation if necessary.
   - [ ] Inform stakeholders or users about the new release.

---
