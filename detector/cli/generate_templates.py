"""
Generate templates CLI for detector package.

This module provides the command line interface for generating template files
(data.yaml, cfg.yaml, hyp.yaml) in a user-specified directory.
"""

import sys
import argparse
import shutil
import logging
from pathlib import Path
from typing import Dict, List

from detector import (
    get_cfg_path,
    get_data_path,
)

logger = logging.getLogger(__name__)


def get_template_sources() -> List[Dict[str, Path]]:
    """
    Get the list of template source and destination mappings.
    
    Returns:
        List of dictionaries containing 'source', 'dest', and 'name' keys.
        Note: 'dest' will be set relative to the output directory by the caller.
    """
    return [
        {
            'source': get_data_path('data_test.yaml'),
            'dest_name': 'data.yaml',
            'name': 'data.yaml'
        },
        {
            'source': get_cfg_path('yolov7-tiny-silu-sqnet.yaml'),
            'dest_name': 'cfg.yaml',
            'name': 'cfg.yaml'
        },
        {
            'source': get_data_path('hyp.finetune.tiny.yaml'),
            'dest_name': 'hyp.yaml',
            'name': 'hyp.yaml'
        },
    ]


def generate_templates(output_dir: Path, force: bool = False) -> tuple:
    """
    Generate template files in the specified output directory.
    
    Args:
        output_dir: Directory to place the template files
        force: If True, overwrite existing files; otherwise skip them
    
    Returns:
        Tuple of (copied_count, skipped_count)
    
    Raises:
        FileNotFoundError: If a template source file does not exist
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get template source files
    templates = get_template_sources()
    
    # Copy each template file
    copied_count = 0
    skipped_count = 0
    
    for template in templates:
        source = template['source']
        dest = output_dir / template['dest_name']
        name = template['name']
        
        # Check if source file exists
        if not source.exists():
            logger.error(f"Template source not found: {source}")
            raise FileNotFoundError(f"Template source not found: {source}")
        
        # Check if destination file already exists
        if dest.exists():
            if force:
                logger.info(f"Overwriting existing file: {dest}")
            else:
                logger.warning(f"Skipping existing file: {dest} (use --force to overwrite)")
                skipped_count += 1
                continue
        
        # Copy the file
        shutil.copy2(source, dest)
        logger.info(f"Copied: {source} -> {dest}")
        copied_count += 1
    
    return copied_count, skipped_count


def generate_templates_command():
    """
    Command line entry point for generating templates.
    
    This function is called when the user runs 'detector-generate-templates' command.
    It parses command line arguments and calls generate_templates() to copy
    template files to the specified output directory.
    """
    parser = argparse.ArgumentParser(
        description='Generate template files (data.yaml, cfg.yaml, hyp.yaml) in a specified directory.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory to place the template files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files if they already exist'
    )
    
    opt = parser.parse_args()
    
    try:
        # Generate templates
        output_dir = Path(opt.output)
        copied_count, skipped_count = generate_templates(output_dir, opt.force)
        
        # Print summary
        print(f"\nTemplate generation complete:")
        print(f"  Copied: {copied_count} file(s)")
        print(f"  Skipped: {skipped_count} file(s)")
        print(f"  Output directory: {output_dir.absolute()}")
        
        if copied_count > 0:
            print(f"\nYou can now edit these files and use them with:")
            print(f"  detector-train --data {output_dir / 'data.yaml'} --cfg {output_dir / 'cfg.yaml'} --hyp {output_dir / 'hyp.yaml'}")
    
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    generate_templates_command()
