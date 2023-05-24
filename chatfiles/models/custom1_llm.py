import torch

from langchain.llms.base import LLM

from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext

from transformers import pipeline

from typing import Optional, List, Mapping, Any

##############################################################################
# define prompt helper

max_input_size = 2048  # set maximum input size

num_output = 256  # set number of output tokens

max_chunk_overlap = 20  # set maximum chunk overlap

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


class CustomLLM(LLM):
    #
    model_name = "facebook/opt-iml-max-30b"

    pipeline = pipeline(
        "text-generation",
        model=model_name,
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        #
        prompt_length = len(prompt)

        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        return response[prompt_length:]  # only return newly generated tokens

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        #
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        #
        return "custom"


llm_predictor = LLMPredictor(llm=CustomLLM())

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

documents = SimpleDirectoryReader('./ data').load_data()

index = GPTListIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query("<query_text>")

print(response)
