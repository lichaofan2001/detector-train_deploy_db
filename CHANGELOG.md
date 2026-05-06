# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-17

### Added
- Initial release of Detector Training Management System
- Web-based GUI for model training and inference
- Support for YOLO model training and testing
- Real-time training log streaming
- Model management and export functionality
- Inference testing with video and image support
- Version management system
- Support for both fast inference (YOLO) and large model inference
- Advanced inference options:
  - Save only images with detected targets
  - Ignore specific class IDs
  - Video frame naming with timestamp information

### Features
- Training configuration management
- Real-time training progress monitoring
- Model weight management
- ONNX export support
- Multi-model support
- Batch inference for videos and image folders
- Customizable inference parameters

### Technical
- Flask-based web interface
- Support for CUDA acceleration
- Integration with YOLOv7 architecture
- Cross-platform support (Windows/Linux)
- Semantic versioning system