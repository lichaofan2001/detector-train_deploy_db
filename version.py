#!/usr/bin/env python3
"""
Version Management Script for Detector Project

This script provides utilities for managing project versions following semantic versioning:
- MAJOR.MINOR.PATCH format
- MAJOR: Major architecture changes or incompatible API modifications
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

Usage:
    python version.py                    # Show current version
    python version.py --major            # Increment major version
    python version.py --minor            # Increment minor version
    python version.py --patch            # Increment patch version
    python version.py --set 2.0.0        # Set specific version
"""

import sys
import re
from pathlib import Path


class VersionManager:
    def __init__(self, version_file=None):
        if version_file is None:
            version_file = Path(__file__).parent / 'VERSION.txt'
        self.version_file = Path(version_file)
        self.version = self._read_version()

    def _read_version(self):
        """Read version from VERSION.txt file."""
        if not self.version_file.exists():
            return "1.0.0"
        return self.version_file.read_text(encoding='utf-8').strip()

    def _write_version(self, version):
        """Write version to VERSION.txt file."""
        self.version_file.write_text(version + '\n', encoding='utf-8')
        self.version = version

    def _parse_version(self, version_str):
        """Parse version string into major, minor, patch components."""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}. Expected format: MAJOR.MINOR.PATCH")
        return tuple(map(int, match.groups()))

    def _format_version(self, major, minor, patch):
        """Format version components into version string."""
        return f"{major}.{minor}.{patch}"

    def get_version(self):
        """Get current version."""
        return self.version

    def increment_major(self):
        """Increment major version (e.g., 1.0.0 -> 2.0.0)."""
        major, minor, patch = self._parse_version(self.version)
        new_version = self._format_version(major + 1, 0, 0)
        self._write_version(new_version)
        print(f"Major version incremented: {self.version} -> {new_version}")
        return new_version

    def increment_minor(self):
        """Increment minor version (e.g., 1.0.0 -> 1.1.0)."""
        major, minor, patch = self._parse_version(self.version)
        new_version = self._format_version(major, minor + 1, 0)
        self._write_version(new_version)
        print(f"Minor version incremented: {self.version} -> {new_version}")
        return new_version

    def increment_patch(self):
        """Increment patch version (e.g., 1.0.0 -> 1.0.1)."""
        major, minor, patch = self._parse_version(self.version)
        new_version = self._format_version(major, minor, patch + 1)
        self._write_version(new_version)
        print(f"Patch version incremented: {self.version} -> {new_version}")
        return new_version

    def set_version(self, version_str):
        """Set specific version."""
        self._parse_version(version_str)  # Validate format
        self._write_version(version_str)
        print(f"Version set: {self.version} -> {version_str}")
        return version_str


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) == 1:
        manager = VersionManager()
        print(f"Current version: {manager.get_version()}")
        return

    manager = VersionManager()
    arg = sys.argv[1]

    if arg == '--major':
        manager.increment_major()
    elif arg == '--minor':
        manager.increment_minor()
    elif arg == '--patch':
        manager.increment_patch()
    elif arg == '--set':
        if len(sys.argv) < 3:
            print("Error: --set requires a version argument")
            print("Usage: python version.py --set 2.0.0")
            sys.exit(1)
        manager.set_version(sys.argv[2])
    else:
        print(f"Unknown argument: {arg}")
        print("Usage:")
        print("  python version.py                    # Show current version")
        print("  python version.py --major            # Increment major version")
        print("  python version.py --minor            # Increment minor version")
        print("  python version.py --patch            # Increment patch version")
        print("  python version.py --set 2.0.0        # Set specific version")
        sys.exit(1)


if __name__ == '__main__':
    main()