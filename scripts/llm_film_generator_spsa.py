"""LLM-based FiLM parameter generator for action modulation (Qwen2.5)."""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
import json
import re
import os

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from peft import PeftModel




class EpisodeHistory:
    """Store episode history for feedback to LLM"""
    def __init__(self, max_size: int=20):
        self.history = []
        self.max_size = max_size

    def add_episode(self, episode_num: int, gamma: np.ndarray, beta: np.ndarray, ep_reward: float, success: bool):
        """Add episode data to history."""
        self.history.append({
            "episode": episode_num,
            "gamma": gamma.tolist(),
            "beta": beta.tolist(),
            "reward": float(ep_reward),
            "success": bool(success)
        })

        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size:]

        # self.history = sorted(self.history, key=lambda x: x["reward"], reverse=False)[:self.max_size]

    def get_param_buffer_string(self, prompt_id) -> str:
        """Format history as text for LLM prompt."""
        if not self.history:
            return "No previous episodes."
        
        if prompt_id == 4:
            lines = []
            for ep in self.history:

                all_params = ep["gamma"] + ep["beta"]

                params_str = "; ".join(f"param[{i}]: {v:.2f}" for i, v in enumerate(all_params))
                line = f"{params_str}, f(params): {ep['reward']:.2f}"
                lines.append(line)

            result = "\n".join(lines)

            print(f"[EpisodeHistory] Formatted table:\n{result}")
            return result

        elif prompt_id in [1, 2, 3]:
            table_str = "| Iterations | Gamma (dim_0, dim_1, ..., dim_15) | Beta (dim_0, dim_1, ..., dim_15) | Total Reward |\n"

            for ep in self.history:
                gamma_str = "[" + ", ".join(f"{g:.3f}" for g in ep["gamma"]) + "]"
                beta_str = "[" + ", ".join(f"{b:.3f}" for b in ep["beta"]) + "]"
                
                table_str += f"| {ep['episode']} | {gamma_str} | {beta_str} | {ep['reward']:.3f} |\n"

                print(f"[EpisodeHistory] Formatted table:\n{table_str}")
            return table_str


    def __len__(self):
        return len(self.history)
    
class LLMFiLMGenerator(nn.Module):
    """
    LLM-based FiLM generator using Qwen2.5 for action modulation.

    Uses local Qwen2.5 model to generate gamma and beta parameters for correcting VLA actions.
    """

    def __init__(
        self,
        bottleneck_dim: int = 16,
        model_name: str = "google/gemma-3-4b-it",
        save_dir: str = "llm_film_logs",
        device: str = "cuda",
        max_episodes: int = 400,
        optimum_reward: float = 0.0,
        step_size: float = 0.01,
        rank = 32
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.model_name = model_name
        self.save_dir = save_dir
        self.device = device
        self.max_episodes = max_episodes
        self.optimum_reward = optimum_reward
        self.step_size = step_size
        self.rank = rank
        self.scratchpad = ""
        os.makedirs(self.save_dir, exist_ok=True)

        # History for feedback
        self.episode_history = EpisodeHistory(max_size=20)

        # Configure model loading
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        model_source =  os.environ["MODEL_SOURCE"]

        self.model = PeftModel.from_pretrained(self.model, model_source)

        # Load Qwen model and tokenizer
        print(f"[LLMFiLM] Loading model: {model_name}:")
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model.eval()

    def add_episode_result(self, episode_num: int, gamma: np.ndarray, beta: np.ndarray, total_reward: float, success: bool):
        """Add episode results to history"""
        self.episode_history.add_episode(
            episode_num=episode_num,
            gamma=gamma,
            beta=beta,
            ep_reward=total_reward,
            success=success
        )

    def build_prompt(self, instruction: str, episode_num: int, total_episodes: int, prompt_id: int = 1) -> str:
        """Build a prompt."""

        if prompt_id == 1:
            history_text = self.episode_history.get_param_buffer_string(prompt_id=1)
            # Props+
            # Pick and place task
            prompt = f"""You are good global optimizer, helping me find the global maximum optimal policy in the following environment within {total_episodes} iterations.

            # Regarding the policy and weights:
            - The policy acts as a corrective layer on top of a base VLA policy, using FiLM modulation applied to the VLA's 16-dimensional internal bottleneck layer.
            - There are 32 weights in total: 16 for scaling (gamma) and 16 for shifting (beta), corresponding to the bottleneck dimensions [dim_0, dim_1, ..., dim_15].
            - Modulation Formula: Corrected_Feature = gamma ⊙ Bottleneck_Feature + beta

            # Here's how we will interact:
            1. I will provide you max steps ({total_episodes}), along with training examples which include weights for policy modulation and their corresponding rewards.
            2. You will provide the response in the **exact** following format:
                * {{"gamma": [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10, g_11, g_12, g_13, g_14, g_15], "beta": [b_0, b_1, b_2, b_3, b_4, b_5. b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15], "explanation": "reasoning"}}
            3. I will then provide the reward and the current iteration number.
            4. You will repeat the steps from 2-3 until we will reach a maximum number of iterations.

            # Remember:
            1. **The global optimum reward is exactly {self.optimum_reward:.1f}.** If current reward is much lower, you need to EXPLORE.
            2. **DO NOT PROPOSE PREVIOUSLY SEEN PARAMS**: This is critical.
            3. **SEARCH SPACE CONSTRAINT**: All Gamma and Beta values MUST be within the range of **[-2.0, 2.0]**. Do not propose any value outside this range.
            4. Analyze the history below and explain WHICH dimensions you are changing and WHY.
            5. **Minimum change per dimension is 0.001.** Any change smaller than 0.001 is forbidden.

            Next, You will see examples of the parameters and their corresponding function values f(parameters).
            {history_text}

            Now you are at iteration {episode_num} out of {total_episodes}. Please provide the results in the indicated format."""

        elif prompt_id == 2:
            history_text = self.episode_history.get_param_buffer_string(prompt_id=1)
            # Props
            # Pick and place task
            prompt = f"""You are good global optimizer, helping me find the global maximum optimal policy in the following environment within {total_episodes} iterations.

            # Regarding the policy and weights:
            - The policy acts as a corrective layer on top of a base VLA policy, using FiLM modulation applied to the VLA's 16-dimensional internal bottleneck layer.
            - There are 32 weights in total: 16 for scaling (gamma) and 16 for shifting (beta), corresponding to the bottleneck dimensions [dim_0, dim_1, ..., dim_15].
            - Modulation Formula: Corrected_Feature = gamma ⊙ Bottleneck_Feature + beta

            # Here's how we will interact:
            1. I will provide you max steps ({total_episodes}), along with training examples which include weights for policy modulation and their corresponding rewards.
            2. You will provide the response in the **exact** following format:
                * {{"gamma": [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10, g_11, g_12, g_13, g_14, g_15], "beta": [b_0, b_1, b_2, b_3, b_4, b_5. b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15], "explanation": "reasoning"}}
            3. I will then provide the reward and the current iteration number.
            4. You will repeat the steps from 2-3 until we will reach a maximum number of iterations.

            # Remember:
            1. **The global optimum reward is exactly {self.optimum_reward:.1f}.** If current reward is lower than that this is local optimum. You should explore instead of exploiting.
            2. **DO NOT PROPOSE PREVIOUSLY SEEN PARAMS**: This is critical.
            3. **SEARCH SPACE CONSTRAINT**: All Gamma and Beta values MUST be within the range of **[-1.0, 1.0]**. Do not propose any value outside this range.
            5. **ANALYZE THE HISTORY**: Do not propose parameters randomly.
            6. **Minimum change per dimension is 0.01.** Any change smaller than 0.01 is forbidden.

            Next, You will see examples of the parameters and their corresponding function values f(parameters).
            {history_text}

            Now you are at iteration {episode_num} out of {total_episodes}. Please provide the results in the indicated format."""

        elif prompt_id == 3:
            history_text = self.episode_history.get_param_buffer_string(prompt_id=1)
            # Props+
            # Pick and place task
            prompt = f"""You are a strong global optimizer. Your goal is to find the global maximum of a black-box function within {total_episodes} iterations.

            # Optimization Problem
            - You are optimizing a 32-dimensional continuous parameter vector.
            - The first 16 values correspond to vector gamma: [g_0, g_1, ..., g_15]
            - The next 16 values correspond to vector beta: [b_0, b_1, ..., b_15]
            - You will propose values for both vectors.

            # Here's how we will interact:
            1. I will provide you max steps ({total_episodes}), along with training examples which include weights for policy modulation and their corresponding rewards.
            2. You will provide the response in the **exact** following format:
                * {{"gamma": [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10, g_11, g_12, g_13, g_14, g_15], "beta": [b_0, b_1, b_2, b_3, b_4, b_5. b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15], "explanation": "reasoning"}}
            3. I will then provide the reward and the current iteration number.
            4. You will repeat the steps from 2-3 until we will reach a maximum number of iterations.

            # Remember:
            1. **The global optimum reward is exactly {self.optimum_reward:.1f}.** If current reward is lower than that this is local optimum. You should explore instead of exploiting.
            2. **DO NOT PROPOSE PREVIOUSLY SEEN PARAMS**: This is CRITICAL.
            3. **SEARCH SPACE CONSTRAINT**: All Gamma and Beta values MUST be within the range of **[-1.5, 1.5]**. Do not propose any value outside this range.
            4. **Minimum change per dimension is 0.001.** Any change smaller than 0.001 is forbidden.

            Next, You will see examples of the parameters and their corresponding function values f(parameters).
            {history_text}

            Now you are at iteration {episode_num} out of {total_episodes}. Please provide the results in the indicated format."""

        elif prompt_id == 4:
            history_text = self.episode_history.get_param_buffer_string(prompt_id=4)

            scratchpad_block = self.scratchpad if self.scratchpad else "(empty)"
            
            prompt = f"""You are a numerical optimizer. Find the global minimum of f(params) over {self.rank} continuous variables in [-6, 6].

            **Known approximate optimum**: {self.optimum_reward:.1f}
            
            The history rows are evaluation results: iter, params, f(params).
            The scratchpad carries all controller state between steps. Read it carefully, use it to decide your move, then output an updated version.
            If the scratchpad is empty: start fresh using the most recent visible point as the base.

            ### Evaluation history:
            {history_text}
            ### Incoming scratchpad:
            {scratchpad_block}

            Output exactly three blocks in this order:

            **<think>**
            Reason through the step:
            - **State Summary**: best visible row and how close it is to {self.optimum_reward:.1f}. Mention the running best from the scratchpad if it differs.
            - **Trend Analysis**: compare the last two visible rows; compute Delta f = f(t) - f(t-1) explicitly.
            - **Scratchpad Readout**: if scratchpad is empty, state that explicitly and describe initialization. Else, summarize the current state of the scratchpad and how it informs your next move.
            - **Strategy & Pacing**: is this a probe or an exploit? How does the current schedule set the step size?
            - **Formula Sheet**: write out every formula you will use before substituting any numbers.
            - **Step Evaluation**: plug in values and compute per coordinate — show values explicitly. Derive the x_next. For higher dimensions, keep context bounded: show detailed arithmetic for only a few representative coordinates, and then provide compact all-dimension computed vectors.
            </think>

            **<scratchpad>**
            Updated controller state for the next step. Values only. Carry exactly what the next decision needs.
            </scratchpad>

            **<param>**
            params[0]: <x0>, params[1]: <x1>, ..., params[{self.rank - 1}]: <x{self.rank - 1}>
            </param>

            **Constraints:**
            - No param set may match any row already in the visible history.
            - Every parameter must have exactly 2 decimal places and lie inside [-6.00, 6.00].
            - Strictly three blocks in the order above; no text outside them.

            **Now at step {episode_num} of {total_episodes}.**"""

        return prompt

    def parse_llm_response(self, response: str, prompt_id) -> Tuple[np.ndarray, np.ndarray, str]:
        """Parse the LLM response to extract gamma, beta, and explanation."""

        if prompt_id == 4:
            # Be tolerant to non-string return types from different LLM wrappers.
            if response is None:
                print("[LLMFiLM] Response is None")
                return np.ones(self.bottleneck_dim, dtype=np.float32), np.zeros(self.bottleneck_dim, dtype=np.float32), ""
            if not isinstance(response, str):
                print(f"[LLMFiLM] Response is not a string: {type(response)}; coercing to str")
                response = str(response)

            # Extract <scratchpad> block
            scratchpad_match = re.search(r"\*\*<scratchpad>\*\*([\s\S]*?)\*\*</scratchpad>\*\*", response, re.IGNORECASE)
            if not scratchpad_match:
                scratchpad_match = re.search(r"<scratchpad>([\s\S]*?)</scratchpad>", response, re.IGNORECASE)
            self.scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else self.scratchpad

            # Extract <think> block (for logging)
            think_match = re.search(r"<think>([\s\S]*?)</think>", response, re.IGNORECASE)
            explanation = think_match.group(1).strip() if think_match else ""
    
            # Extract <param> block if present; otherwise parse the full response.
            param_match = re.search(r"<param>([\s\S]*?)</param>", response, re.IGNORECASE)
            param_block = param_match.group(1).strip() if param_match else response

            dim = self.bottleneck_dim

            # Preferred parse: params[i]: value and split first 16 as gamma, last 16 as beta.
            params_by_index: dict = {}
            for m in re.finditer(r"params\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)", param_block, re.IGNORECASE):
                params_by_index[int(m.group(1))] = float(m.group(2))

            if len(params_by_index) >= 2 * dim:
                gamma = np.array([params_by_index[i] for i in range(dim)], dtype=np.float32)
                beta = np.array([params_by_index[i + dim] for i in range(dim)], dtype=np.float32)
            else:
                # Backward-compatible parse: gamma[i] / beta[i].
                gamma_values: dict = {}
                for m in re.finditer(r"gamma\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)", param_block, re.IGNORECASE):
                    gamma_values[int(m.group(1))] = float(m.group(2))

                beta_values: dict = {}
                for m in re.finditer(r"beta\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)", param_block, re.IGNORECASE):
                    beta_values[int(m.group(1))] = float(m.group(2))

                if len(gamma_values) >= dim and len(beta_values) >= dim:
                    gamma = np.array([gamma_values[i] for i in range(dim)], dtype=np.float32)
                    beta = np.array([beta_values[i] for i in range(dim)], dtype=np.float32)
                else:
                    # Last-resort parse: first 32 numeric values in order (no labels required).
                    raw_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", param_block)
                    if len(raw_numbers) >= 2 * dim:
                        numeric = np.array([float(v) for v in raw_numbers[: 2 * dim]], dtype=np.float32)
                        gamma = numeric[:dim]
                        beta = numeric[dim : 2 * dim]
                    else:
                        print(
                            f"[LLMFiLM] Incomplete parse: got {len(params_by_index)} params, "
                            f"{len(gamma_values)} gamma, {len(beta_values)} beta, "
                            f"{len(raw_numbers)} raw numbers from:\n{param_block}"
                        )
                        return np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32), explanation
    
            # print(f"[LLMFiLM] Parsed gamma: {gamma}, beta: {beta}")
            # print(f"[LLMFiLM] Explanation: {explanation}")
    
            return gamma, beta, explanation

        elif prompt_id in [1, 2, 3]:

            try:
                # Remove markdown code blocks if present
                cleaned = response.strip()

                if "```" in cleaned:
                    # Extract content between code blocks
                    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)

                    if code_match:
                        cleaned = code_match.group(1).strip()

                # Try to find JSON object
                json_match = re.search(r"\{[\s\S]*?\"gamma\"[\s\S]*?\"beta\"[\s\S]*?\}", cleaned)

                if json_match:
                    data = json.loads(json_match.group())

                    gamma = np.array(data["gamma"], dtype=np.float32)
                    beta = np.array(data["beta"], dtype=np.float32)
                    explanation = data.get("explanation", "")

                    print(f"[LLMFiLM] Parsed gamma: {gamma}, beta: {beta}")
                    print(f"[LLMFiLM] Explanation: {explanation}")

                    return gamma, beta, explanation

                else:
                    print(f"[LLMFiLM] No JSON found in cleaned response: {cleaned}")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[LLMFiLM] Failed to parse response: {e}")
                print(f"[LLMFiLM] Response was: {response}")

            # Fallback: return gamma=1, beta=0, empty explanation
            return np.ones(self.bottleneck_dim, dtype=np.float32), np.zeros(self.bottleneck_dim, dtype=np.float32), ""

    def call_llm(self, prompt: str, max_new_tokens: int = 512) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}], 
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)

            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.processor.decode(new_tokens, skip_special_tokens=True)
            print(f"[LLMFiLM] Raw LLM response:\n{response}")
            return response

        except Exception as e:
            print(f"[LLMFiLM] Error during LLM call: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def generate_film_params(self, instruction: str, episode_num: int, total_episodes: int, device: torch.device = None, prompt_id: int = 1) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generate FiLM parameters using LLM with feedback.
        """
        if device is None:
            device = torch.device("cpu")

        # Episode 0 uses base parameters without LLM intervention
        if episode_num == 0:
            # For the first episode, return base parameters
            self.scratchpad = ""
            gamma = np.ones(self.bottleneck_dim, dtype=np.float32)
            beta = np.zeros(self.bottleneck_dim, dtype=np.float32)
            explanation = "Initial episode: using base parameters"

        else:
            # Build step-wise prompt
            prompt = self.build_prompt(
                instruction=instruction,
                episode_num=episode_num,
                total_episodes=total_episodes,
                prompt_id=prompt_id
            )

            tokens = self.processor.tokenizer.encode(prompt)
            num_tokens = len(tokens)

            print(f"[LLMFiLM] Episode {episode_num} Prompt Token Count: {num_tokens}")

            response = self.call_llm(prompt, max_new_tokens=4096)

            gamma, beta, explanation = self.parse_llm_response(response, prompt_id=prompt_id)

        gamma_t = torch.tensor(gamma, dtype=torch.float32, device=device)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=device)

        return gamma_t, beta_t, explanation