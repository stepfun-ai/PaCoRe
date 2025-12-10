import os
from functools import cached_property
import asyncio
import copy
import random
from typing import Optional

import jinja2
from loguru import logger

from pacore.utils import load, async_chat_completion, save_jsonl
class Exp:
    # Model & API
    model_name: str = "default"
    api_base: str = "http://localhost:8000/v1/chat/completions"

    # Data
    data_path: str = "data.jsonl"
    output_dir: str = "outputs"

    # Generation
    max_tokens: int = 131072
    temperature: float = 1.0
    top_p: float = 1.0
    timeout_seconds: float = 7200
    retry_times: int = 5
    stream: bool = False

    # Parallel thinking
    num_responses_per_round: list[int] = [4,]

    # Concurrency
    max_concurrent: int = 64
    random_seed: int = 42

    @cached_property
    def file_path(self):
        import inspect

        return inspect.getabsfile(self.__class__)

    @property
    def exp_name(self):
        base_name = os.path.basename(self.file_path).split(".")[0]
        return base_name

    @property
    def output_path(self)->str:
        return os.path.join(self.output_dir, f"{self.exp_name}", "results.jsonl")


    def build_data(self)->list[dict]:
        return load(self.data_path)

    @staticmethod
    def parse_answer(response: str) -> str:
        """
        pick everything after last </think>, if not satisfied, return original response
        NOTE: for proposer, we do not care about truncation in answer actually
        """
        res = response
        boa_string = r"</think>"

        # split base on boa_string
        boa_splits = response.split(boa_string)
        if len(boa_splits) > 1:
            res = boa_splits[-1]

        return res

    @staticmethod
    def format_controller_prompt(
        original_prompt: str, ref_responses: list[str]
    ) -> str:
        controller_prompt_text = """\
You are given a problem and a list of reference responses. Your job is to analyze these references and provide your own response.

Original Problem:
{{ original_prompt }}

Reference Responses:
{% for response in ref_responses %}
Reference {{ loop.index }}:
{{ response }}
{% endfor %}

Now, based on the original problem and reference responses above, please provide your own comprehensive solution.
"""
        template = jinja2.Template(controller_prompt_text)
        return template.render(
            original_prompt=original_prompt, ref_responses=ref_responses
        )

    def __init__(self):
        self._semaphore: Optional[asyncio.Semaphore] = None

        # just add one always for num_responses_per_round
        self.num_responses_per_round = self.num_responses_per_round + [1]
        logger.info(f"num_responses_per_round: {self.num_responses_per_round}")

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def _call_api(self, messages: list[dict]) -> dict:
        async with self._get_semaphore():
            return await async_chat_completion(
                messages=messages,
                model=self.model_name,
                api_base=self.api_base,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout_seconds=self.timeout_seconds,
                retry_times=self.retry_times,
                stream=self.stream,
            )

    async def _run_parallel_calls(self, messages: list[dict], num_calls: int) -> list[dict]:
        """
        Run parallel API calls and return list of response dicts.
        Each dict contains 'content' and 'reasoning_content' keys.
        """
        tasks = [asyncio.create_task(self._call_api(messages)) for _ in range(num_calls)]
        responses = []
        for idx, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                resp = await coro
                choice = resp.get("choices", [{}])[0]
                content = choice["message"].get("content", "")
                reasoning_content = choice["message"].get("reasoning_content", "")
                responses.append({
                    "content": content,
                    "reasoning_content": reasoning_content,
                })
            except Exception as e:
                logger.error(f"API call failed: {e}")
                responses.append({"content": "", "reasoning_content": ""})
        # Shuffle to avoid ordering bias
        responses.sort(key=lambda x: len(x["content"]))
        random.Random(self.random_seed).shuffle(responses)
        return responses

    async def process_single(self, messages: list[dict], request_id: str = "") -> dict:
        """Process single request through all rounds."""
        all_rounds: list[list[dict]] = []

        # Get original prompt
        original_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                original_prompt = msg.get("content", "")
                break

        logger.info(f"[{request_id}] {len(self.num_responses_per_round)} rounds")

        for round_idx, num in enumerate(self.num_responses_per_round):
            logger.info(f"[{request_id}] Round {round_idx}: {num} responses")

            if round_idx == 0:
                current_messages = messages
            else:
                prev_answers = [self.parse_answer(r["content"]) for r in all_rounds[-1] if r.get("content")]
                current_messages = copy.deepcopy(messages)
                for i in range(len(current_messages) - 1, -1, -1):
                    if current_messages[i].get("role") == "user":
                        current_messages[i]["content"] = self.format_controller_prompt(
                            original_prompt, prev_answers
                        )
                        break

            responses = await self._run_parallel_calls(current_messages, num)
            all_rounds.append(responses)

        logger.info(f"[{request_id}] Done")

        final_round = all_rounds[-1][0] if all_rounds[-1] else {}
        return {
            "request_id": request_id,
            "final_response": final_round.get("content", "") if final_round else "",
            "final_reasoning_content": final_round.get("reasoning_content", "") if final_round else "",
            "round_responses": all_rounds,
        }

    async def async_run(self):
        """Async entry point."""
        logger.info(f"saving to {self.output_path}")
        data = self.build_data()
        logger.info(f"Loaded {len(data)} items from {self.data_path}")

        async def process_item(idx, item):
            messages = item.get("messages", [])
            request_id = item.get("id", f"req_{idx}")
            try:
                return await self.process_single(messages, request_id)
            except Exception as e:
                logger.error(f"[{request_id}] Failed: {e}")
                return {"request_id": request_id, "error": str(e)}

        # Process all requests concurrently
        tasks = [process_item(idx, item) for idx, item in enumerate(data)]
        results = await asyncio.gather(*tasks)

        save_jsonl(list(results), self.output_path)
        logger.info(f"Saved {len(results)} results to {self.output_path}")
        return results

    def run(self):
        """Main entry point. Call this to execute the experiment."""
        return asyncio.run(self.async_run())


if __name__ == "__main__":
    exp = Exp()
    exp.run()
