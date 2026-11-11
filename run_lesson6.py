"""第六课运行入口：LangChain 框架入门"""
import sys
import os

def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from agent.lesson6_langchain import main

if __name__ == "__main__":
    main()
