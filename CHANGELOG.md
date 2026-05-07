# 版本更新日志

格式说明：
- **MAJOR.MINOR.PATCH** 版本号
  - MAJOR: 主版本，不兼容的API修改
  - MINOR: 次版本，向后兼容的新功能
  - PATCH: 补丁版本，向后兼容的bug修复

---

## [1.1.0] - 2026-05-06

### 新增功能
- 训练日志持久化：日志文件保存到 `yolo_train_tool/logs/` 目录
- 页面刷新恢复：刷新页面后可继续查看训练日志
- 新增 `/api/train/log/history` 接口获取历史日志

### 问题修复
- 修复 ONNX 导出时 TracerWarning 警告问题
  - `torch.onnx.export()` 输入参数改为元组形式 `(img,)`
  - `Model.forward()` 添加 `torch.onnx.is_in_onnx_export()` 检测
  - `Model.forward_once()` 添加 `torch.onnx.is_in_onnx_export()` 检测
- 修复模型管理界面指标显示错误：修正 `results.txt` 解析索引从 `[4-7]` 改为 `[8-11]`

### 变更说明
- 日志文件目录从 `project_root/logs/` 改为 `project_root/yolo_train_tool/logs/`
- 前端 `sessionStorage` 用于保存当前训练 session 信息

---

## [1.0.0] - 初始版本

### 功能
- 基础训练功能
- ONNX 导出功能
- 模型管理界面
