# helper/logging.py
import logging
import sys
import colorama
colorama.init(autoreset=True)

COLORS = {
    "INFO": "\033[0m",       # 白色
    "WARNING": "\033[93m",   # 黃色
    "ERROR": "\033[91m",     # 紅色
    "DEBUG": "\033[94m",     # 藍色
    "RESET": "\033[0m"
}

class SimpleColorFormatter(logging.Formatter):
    def format(self, record):
        # 取得對應顏色，若無對應則使用 RESET
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        # 格式化訊息: 顏色 + 原始訊息 + 重置顏色
        return f"{color}{record.getMessage()}{reset}"

def setup_logger(name: str = __name__, log_level: int = logging.DEBUG):
    """
    通用 Logger 設定函式
    使用方式:
        from helper.logging import setup_logger
        logger = setup_logger(__name__)
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 檢查是否已經有 Handler
    if not logger.handlers:
        # 建立 Console Handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # 設定輸出格式
        formatter = SimpleColorFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 防止 log 傳遞給 root logger (避免重複印出)
        logger.propagate = False

    return logger