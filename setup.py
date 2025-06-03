from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

# ---- Helpers --------------------------------------------------------------
def parse_requirements(path: Path):
    """Return a list of requirements, ignoring comments and empty lines."""
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements

# ---- Metadata -------------------------------------------------------------
PACKAGE_NAME = "language_driven_exploration"   # all-lowercase, no spaces
VERSION      = "0.1.0"
DESCRIPTION  = "Code for language-driven exploration"
AUTHOR       = "Nicolas Pfitzer"

# ---- Files ----------------------------------------------------------------
readme_file   = ROOT / "README.md"
requirements_file = ROOT / "requirements.txt"

LONG_DESCRIPTION = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
INSTALL_REQUIRES = parse_requirements(requirements_file) if requirements_file.exists() else []

# ---- Setup ----------------------------------------------------------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=">=3.9",
    packages=find_packages(exclude=("evaluations*", "experiments*","outputs*", "checkpoints*")),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            # lets users call `my-deployment` on the CLI
            "my-deployment=deployment.my_deployment:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
