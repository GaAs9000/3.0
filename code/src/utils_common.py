"""
通用工具函数模块
提供跨模块使用的通用功能，避免代码重复
"""

def safe_rich_debug(message: str, category: str = None):
    """
    安全的rich_output调用，避免ImportError
    
    Args:
        message: 调试消息
        category: 消息分类（可选）
    """
    try:
        from rich_output import rich_debug
        if category:
            rich_debug(message, category)
        else:
            rich_debug(message)
    except ImportError:
        # 静默处理，rich_output模块不可用时不输出调试信息
        pass

def safe_rich_warning(message: str):
    """
    安全的rich_output警告调用
    
    Args:
        message: 警告消息
    """
    try:
        from rich_output import rich_warning
        rich_warning(message)
    except ImportError:
        print(f"WARNING: {message}")

def safe_rich_error(message: str):
    """
    安全的rich_output错误调用

    Args:
        message: 错误消息
    """
    try:
        from rich_output import rich_error
        rich_error(message)
    except ImportError:
        print(f"ERROR: {message}")

def safe_rich_success(message: str):
    """
    安全的rich_output成功调用

    Args:
        message: 成功消息
    """
    try:
        from rich_output import rich_success
        rich_success(message)
    except ImportError:
        print(f"✅ {message}")
