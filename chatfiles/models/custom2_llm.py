# 从TYPING库中导入必要的函数和类型声明
from typing import Any, List, Mapping, Optional

# 导入所需的类和接口
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


# 定义一个名为CUSTOMLLM的子类，继承自LLM类
class CustomLLM(LLM):
    n: int  # 类的成员变量，类型为整型

    # 用于指定该子类对象的类型
    @property
    def _llm_type(self) -> str:
        return "custom"

    # 重写基类方法，根据用户输入的PROMPT来响应用户，返回字符串
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

# 语法解释：
# 1. 使用TYPING库中的相关类型进行类型声明
# 2. 使用继承实现自定义LLM类的功能扩展
# 3. 通过重写父类的方法以实现特定的功能需求
# 4. 使用@property装饰器很好地实现了对私有变量和方法的封装和保护
# 5. _identifying_params属性和_llm_type属性分别用于标识和记录各自对象的属性和类型信息。
