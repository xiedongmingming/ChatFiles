import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"

logger = logging.getLogger()

logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

#######################################################################################
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

EMBEDDING_MODEL = "text2vec"

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
    "moss": "fnlp/moss-moon-003-sft",
}

LLM_MODEL = "chatglm-6b-int8"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#######################################################################################
LLM_LORA_PATH = ""  # 默认为空，如果有请直接指定文件夹路径

USE_LORA = True if LLM_LORA_PATH else False

STREAMING = True  # LLM streaming reponse

USE_PTUNING_V2 = False  # Use p-tuning-v2 PrefixEncoder

LOAD_IN_8BIT = True  # MOSS load in 8bit

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# 基于上下文的PROMPT模版，请务必保留："{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

SENTENCE_SIZE = 500  # 文本分句长度

CHUNK_SIZE = 250  # 匹配后单段上下文长度

LLM_HISTORY_LEN = 3  # LLM input history length

VECTOR_SEARCH_TOP_K = 5  # return top-k text chunk from vector store

VECTOR_SEARCH_SCORE_THRESHOLD = 0  # 知识检索内容相关度SCORE，数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex

logger.info(
    f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
"""
)

OPEN_CROSS_DOMAIN = False  # 是否开启跨域，默认为FALSE，如果需要开启，请设置为TRUE
