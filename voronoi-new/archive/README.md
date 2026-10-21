# 原始代码快照

`original_full14/` 保存首次生成完整 14-10.20 结果时使用的脚本，包括当时的固定输入输出路径和旧后端依赖，供追溯。请使用本包根目录的 `render_video.py` 或 `run_14.ps1` 运行。

其中 `fast_voronoi.py`、`baseline_video.py` 是当时导入链中的旧实现。根目录的整理版本只开放此次实际使用的 BVH 后端，移除了主流程对旧后端的导入依赖，数值结果由真实帧逐像素检查验证。

原脚本的哈希与 `evidence/validation.json` 对照；本包所有交付文件的哈希见根目录 `PACKAGE_MANIFEST.json`。完整视频、大缓存和逐帧长日志仍位于原实验结果目录，通过文档及配置引用。
