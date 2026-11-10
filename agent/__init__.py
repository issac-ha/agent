# AI Agent 学习项目
import sys
import os

# 修复 Python 环境路径问题：把正确的 site-packages 加入 sys.path
# （你的 Python 安装在 D:\python\python3.10.6，但 site 模块指向了 D:\python）
_site_packages = r"D:\python\python3.10.6\lib\site-packages"
if os.path.isdir(_site_packages) and _site_packages not in sys.path:
    sys.path.insert(0, _site_packages)
