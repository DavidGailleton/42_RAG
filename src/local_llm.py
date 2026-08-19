from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwen:
    """Load and run a local Qwen causal language model.

    Attributes:
        model_name: Hugging Face identifier of the loaded model.
        tokenizer: Tokenizer associated with the model.
        model: Loaded causal language model.
        device: Torch device used for inference.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the local Qwen model.

        Args:
            model_name: Hugging Face model identifier.
        """
        self.model_name = model_name

        # Transformers uses dynamic factory classes. Mark the boundary as Any
        # because the concrete types depend on the selected model.
        tokenizer_factory: Any = AutoTokenizer
        model_factory: Any = AutoModelForCausalLM

        self.tokenizer: Any = tokenizer_factory.from_pretrained(model_name)
        self.model: Any = model_factory.from_pretrained(
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
        """Generate text from system and user prompts.

        Args:
            system_prompt: Instructions controlling model behavior.
            user_prompt: User question and retrieved supporting context.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text with the input prompt and special tokens removed.

        Raises:
            ValueError: If ``max_new_tokens`` is not positive.
            RuntimeError: If model inference fails.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

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

        prompt: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs: dict[str, torch.Tensor] = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(self.device) for key, value in model_inputs.items()
        }

        with torch.inference_mode():
            output_ids: torch.Tensor = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]

        decoded = cast(
            str,
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ),
        )
        return decoded.strip()
