"""
Setup script for detector package.

This package provides tools for training and evaluating object detection models
based on YOLOv7 architecture.

The package includes:
- Training and inference scripts
- Model configuration files (cfg/*.yaml)
- Dataset configuration files (data/*.yaml)
- Pre-trained weights (ckpoints/*.pt) - copied to ~/.detector/ckpoints/ during installation

Usage:
    pip install -e .  # Install in development mode
    pip install .     # Install as a regular package

After installation, you can use the CLI commands:
    detector-train --data data/data.yaml --epochs 100
    detector-test --data data/data.yaml --weights ~/.detector/ckpoints/best.pt
    detector-detect --weights ~/.detector/ckpoints/best.pt --source image.jpg

Default paths:
    - Default cfg: cfg/yolov7-tiny-silu-sqnet.yaml (inside package)
    - Default weights: ~/.detector/ckpoints/opencar.pt (user directory)
    - Default data: data/data_test.yaml (inside package)
    - Default hyp: data/hyp.scratch.tiny.yaml (inside package)
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path
import os
import shutil

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "ARCHITECTURE.md").read_text(encoding='utf-8') if (this_directory / "ARCHITECTURE.md").exists() else ""

# Read version from VERSION.txt
VERSION_FILE = this_directory / "VERSION.txt"
VERSION = VERSION_FILE.read_text(encoding='utf-8').strip() if VERSION_FILE.exists() else "1.0.0"

# Package metadata
NAME = "detector"
AUTHOR = "Onserve Detector Team"
AUTHOR_EMAIL = "detector_observe@example.com"
DESCRIPTION = "Object Detection Model Training and Inference Package based on YOLO"
LONG_DESCRIPTION = long_description
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/example/detector-train"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/example/detector-train/issues",
    "Source Code": "https://github.com/example/detector-train",
}
LICENSE = "MIT"
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Requirements
REQUIRED = [
    # "torch>=1.7.0",
    # "torchvision>=0.8.0",
    # "numpy>=1.18.5",
    # "opencv-python>=4.1.1",
    # "PyYAML>=5.3.1",
    # "tqdm>=4.46.0",
    # "tensorboard",
    # "pandas>=1.1.4",
    # "seaborn>=0.11.0",
    # "matplotlib>=3.3.0",
    # "Pillow>=7.1.2",
    # "scipy>=1.4.1",
    # "thop>=0.0.31",  # FLOPs computation
]

# Optional dependencies
EXTRAS = {
    # "gui": ["flask>=2.0.0"],  # Training GUI web interface
    # "wandb": ["wandb>=0.12.0"],  # Weights & Biases logging
    # "dev": [
    #     "pytest>=6.0.0",
    #     "pytest-cov>=2.10.0",
    #     "black>=21.5b1",
    #     "flake8>=3.8.0",
    #     "isort>=5.6.0",
    # ],
}

# Entry points for CLI
ENTRY_POINTS = {
    "console_scripts": [
        "detector-train=detector.cli.train_cli:train_command",
        "detector-test=detector.cli.test_cli:test_command",
        "detector-detect=detector.cli.detect:detect_command",
        "detector-show=detector.show_detector_results:show_command",
        "detector-predict=detector.cli.predict_bbox:predict_command",
        "compute-anchors=detector.cli.compute_anchors:compute_anchors_command",
        "print-anchors=detector.cli.print_anchors:print_anchors_command",
        "detector-trim=detector.cli.model_trim:model_trim_command",
        "detector-rename=detector.cli.model_rename_classes:model_rename_command",
        "export-onnx=detector.cli.export_onnx:export_onnx_command",
        "export-fake-rgbt=detector.cli.export_fake_rgbt:export_fake_rgbt_command",
        "export-scale-conf=detector.cli.export_scale_conf:export_scale_conf_command",
        "detector-analyze=detector.cli.analyze_list:analyze_list_command",
        "detector-generate-templates=detector.cli.generate_templates:generate_templates_command",
        "detector-gui=detector_gui.cli:gui_command",
    ],
}

# Find packages
PACKAGES = find_packages(exclude=["tests", "tests.*"])

# Data files to include (ckpoints/*.pt will be installed to both package and user directory)
PACKAGE_DATA = {
    "detector": [
        "cfg/*.yaml",           # Model configuration files
        "data/*.yaml",          # Dataset configuration files
        "data/*.txt",           # Dataset related text files
    ],
    "detector_gui": [
        "templates/*.html",     # GUI HTML templates
        "static/*.css",         # GUI CSS styles
    ],
}


def copy_ckpoints_to_user_dir():
    """Copy checkpoint files to user directory."""
    # Get the user's home directory
    home_dir = Path.home()
    
    # Determine the target directory based on the operating system
    if os.name == 'nt':  # Windows
        target_dir = home_dir / ".detector" / "ckpoints"
    else:  # Linux and other Unix-like systems
        target_dir = home_dir / ".detector" / "ckpoints"
    
    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the source directory for ckpoints
    # The ckpoints directory is in the project root, not inside the detector package
    project_root = Path(__file__).parent
    source_ckpoints_dir = project_root / "ckpoints"
    
    if source_ckpoints_dir.exists() and source_ckpoints_dir.is_dir():
        print(f"Copying checkpoint files to {target_dir}...")
        
        # Copy all .pt files from source to target
        for pt_file in source_ckpoints_dir.glob("*.pt"):
            target_file = target_dir / pt_file.name
            shutil.copy2(pt_file, target_file)
            print(f"  Copied: {pt_file.name}")
        
        # Also copy the README.md if it exists
        readme_file = source_ckpoints_dir / "README.md"
        if readme_file.exists():
            target_readme = target_dir / "README.md"
            shutil.copy2(readme_file, target_readme)
            print(f"  Copied: README.md")
        
        print(f"Checkpoint files installed to: {target_dir}")
    else:
        print(f"Warning: ckpoints directory not found at {source_ckpoints_dir}")


class CustomInstallCommand(install):
    """Custom installation command to handle ckpoints installation to user directory."""
    
    def run(self):
        # Run the default installation first
        install.run(self)
        # Copy checkpoint files to user directory
        copy_ckpoints_to_user_dir()


class CustomDevelopCommand(develop):
    """Custom develop command to handle ckpoints installation to user directory for development mode."""
    
    def run(self):
        # Run the default develop installation first
        develop.run(self)
        # Copy checkpoint files to user directory
        copy_ckpoints_to_user_dir()


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=PACKAGES,
    python_requires=">=3.7",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    package_data=PACKAGE_DATA,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
)
