from typing import Any, Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwen:
    """Load and execute Qwen directly with Transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the local Qwen model.

        Args:
            model_name: Hugging Face model identifier.
        """
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 300,
    ) -> str:
        """Generate an answer from system and user prompts.

        Args:
            system_prompt: Instructions given to the model.
            user_prompt: Question and retrieved context.
            max_new_tokens: Maximum number of generated tokens.

        Returns:
            Generated text without the original prompt.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs: dict[str, Any] = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(self.device) for key, value in model_inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def chunk(self) -> Generator:
        pass
