from __future__ import annotations

from typing import Callable, List

from .schema import SearchResult
from .qwen_gen import build_citation_prompt, build_baseline_upgrade_prompt


class QwenGenerator:
    def __init__(self, model_path: str = "models/Qwen2.5-3B-Instruct", adapter_path: str = None, load_in_4bit: bool = False) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False,
                quantization_config=bnb_config,
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False,
            )
        if self.adapter_path:
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        self._load()
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        gen_kwargs.update(kwargs)
        # Force do_sample=True (temperature/top_p only work when do_sample=True)
        gen_kwargs["do_sample"] = True

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs
            )
        response = self._tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True,
        )
        return response.strip()

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 512, **kwargs) -> List[str]:
        """Batch generate for multiple prompts (single forward pass).

        Args:
            prompts: List of input prompts.
            max_new_tokens: Max new tokens per sample.
            **kwargs: Passed to model.generate() (e.g., temperature).

        Returns:
            List of generated responses (one per prompt).
        """
        self._load()
        import torch

        # Save original padding side (decoder-only models need left-padding)
        original_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = 'left'

        # Apply chat template to each prompt
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Batch tokenize with padding (now uses left-padding)
        inputs = self._tokenizer(texts, padding=True, return_tensors="pt").to(self._model.device)

        # Do NOT restore padding_side yet! model.generate() checks it.
        # It must remain 'left' during generation.

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        gen_kwargs.update(kwargs)
        if gen_kwargs.get("do_sample"):
            gen_kwargs.setdefault("temperature", 0.7)
            gen_kwargs.setdefault("top_p", 0.9)

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs
                )
        finally:
            self._tokenizer.padding_side = original_padding_side

        # Decode each sample in the batch
        prompt_width = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(prompts)):
            generated = outputs[i][prompt_width:]
            decoded = self._tokenizer.decode(generated, skip_special_tokens=True)
            results.append(decoded.strip())

        return results

    def answer_with_context(
        self, question: str, results: list[SearchResult], mode: str = "citation"
    ) -> str:
        if mode == "citation":
            prompt = build_citation_prompt(question, results)
        else:
            from .rag_pipeline import make_extractive_answer
            extractive = make_extractive_answer(question, results)
            prompt = build_baseline_upgrade_prompt(question, extractive)
        return self.generate(prompt)

    def batch_answer(
        self,
        questions: list[str],
        all_results: list[list[SearchResult]],
        mode: str = "citation",
    ) -> list[str]:
        answers = []
        for question, results in zip(questions, all_results):
            answers.append(self.answer_with_context(question, results, mode))
        return answers

    @property
    def device(self) -> str:
        if self._model is None:
            return "not_loaded"
        return str(next(self._model.parameters()).device)
