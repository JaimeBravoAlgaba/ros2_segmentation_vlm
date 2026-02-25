from typing import Dict, List

import torch
from sam3.model.sam3_image_processor import Sam3Processor


class Sam3MultiPromptProcessor(Sam3Processor):
    """
    Extension of Sam3Processor that supports encoding a fixed set of text prompts
    once and then re-using them for many images.

    Usage:
        proc = Sam3MultiPromptProcessor(model)
        proc.set_text_prompts(["grass", "road", "person"])

        for img in images:
            results = proc.segment_image_with_text_prompts(img)
            # results is a list of dicts, one per prompt
    """

    def __init__(
        self,
        model,
        resolution: int = 1008,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ):
        super().__init__(
            model=model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        # Cached prompts and their corresponding text encoder outputs
        self._cached_prompts: List[str] = []
        self._cached_text_outputs: List[Dict[str, torch.Tensor]] = []

    @torch.inference_mode()
    def encode_text_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Encode a single text prompt and return its text encoder outputs.

        This version matches the tensor-layout handling used in set_text_prompts().
        """
        if not prompt:
            raise ValueError("prompt string must not be empty")

        batched = self.model.backbone.forward_text([prompt], device=self.device)

        per_prompt: Dict[str, torch.Tensor] = {}
        for k, v in batched.items():
            if not torch.is_tensor(v):
                print("Warning: non-tensor output in text encoding:", k)
                per_prompt[k] = v
                continue

            # Match the same logic used in set_text_prompts()
            if k in ("language_features", "language_embeds", "additional_text_features"):
                # Expected: [seq_len, num_prompts, dim] (or generally prompt dim at axis=1)
                if v.ndim >= 2:
                    per_prompt[k] = v[:, 0:1, ...]   # keep prompt dim of 1
                else:
                    per_prompt[k] = v

            elif k in ("language_mask", "additional_text_mask"):
                # Expected: [num_prompts, seq_len] (prompt dim at axis=0)
                if v.ndim >= 1:
                    per_prompt[k] = v[0:1, ...]      # keep prompt dim of 1
                else:
                    per_prompt[k] = v

            else:
                # Other tensors: keep as-is
                per_prompt[k] = v

        return per_prompt


    @torch.inference_mode()
    def set_text_prompts(self, prompts: List[str]):
        """
        Encode and cache a fixed list of text prompts in a single batched call.

        This should be called once at the beginning (or whenever you change the
        text prompts). The resulting encodings will be re-used for all subsequent
        images.

        Args:
            prompts: list of text prompts, e.g. ["grass", "road", "person"]
        """
        if len(prompts) == 0:
            raise ValueError("prompts list must not be empty")

        self._cached_prompts = list(prompts)
        self._cached_text_outputs = []

        # Single batched call: encode all prompts at once
        batched = self.model.backbone.forward_text(prompts, device=self.device)
        num_prompts = len(prompts)

        # Split batched outputs into per-prompt dicts, respecting the actual
        # layout of the tensors:
        #   - language_features: [seq_len, num_prompts, dim]  -> slice dim=1
        #   - language_embeds:   [*, num_prompts, *]          -> slice dim=1
        #   - language_mask:     [num_prompts, seq_len]       -> slice dim=0
        #   - additional_text_*: follow the same pattern
        for i in range(num_prompts):
            per_prompt: Dict[str, torch.Tensor] = {}
            for k, v in batched.items():
                if not torch.is_tensor(v):
                    # Non-tensor metadata: just share
                    print("Warning: non-tensor output in text encoding:", k)
                    per_prompt[k] = v
                    continue

                # Handle the common SAM3 language keys explicitly
                if k in ("language_features", "language_embeds", "additional_text_features"):
                    # caption dimension is the 2nd axis: [:, prompt, ...]
                    if v.ndim >= 2:
                        # Keep a "batch"/caption dim of 1
                        per_prompt[k] = v[:, i : i + 1, ...]
                    else:
                        per_prompt[k] = v

                elif k in ("language_mask", "additional_text_mask"):
                    # caption dimension is the 1st axis: [prompt, ...]
                    if v.ndim >= 1:
                        per_prompt[k] = v[i : i + 1, ...]
                    else:
                        per_prompt[k] = v

                else:
                    # Any other tensors are treated as shared (not per-prompt)
                    per_prompt[k] = v

            self._cached_text_outputs.append(per_prompt)

    @torch.inference_mode()
    def segment_image_with_text_prompts(self, image, state: Dict = None):
        """
        Encode the image and run inference for each cached text prompt.

        This will:
          1) run the vision backbone once for `image`
          2) for each cached text prompt, combine its cached text features
             with the image features and run `_forward_grounding`

        Args:
            image: PIL image or tensor, same as `set_image` in Sam3Processor
            state: optional state dict to reuse/extend; if None a new one is created

        Returns:
            List[Dict] of length `len(self._cached_prompts)`, where each dict has:
                {
                    "prompt": <prompt string>,
                    "masks": Tensor[..., H, W] (bool),
                    "masks_logits": Tensor[..., H, W] (float),
                    "boxes": Tensor[..., 4],
                    "scores": Tensor[...],
                }
        """
        if not self._cached_prompts or not self._cached_text_outputs:
            raise ValueError(
                "You must call set_text_prompts(...) before "
                "segment_image_with_text_prompts(...)."
            )

        # Step 1: run the image through the backbone (once)
        if state is None:
            state = {}
        state = self.set_image(image, state=state)

        if "backbone_out" not in state:
            raise RuntimeError(
                "Image encoding failed: 'backbone_out' is missing from state."
            )

        # Reuse the same backbone_out dict, only patch language fields
        backbone_out = state["backbone_out"]

        # Make sure we have a geometric prompt once
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        all_results = []

        # Step 2: for each cached prompt, patch in its text features and run grounding
        for prompt, text_outputs in zip(
            self._cached_prompts, self._cached_text_outputs
        ):
            # Update language-related entries in backbone_out (in-place)
            backbone_out.update(text_outputs)

            # Shallow copy of state so outputs (masks, boxes, scores) don't clobber
            prompt_state: Dict = dict(state)

            # Run the actual SAM3 grounding
            prompt_state = self._forward_grounding(prompt_state)

            # Collect results for this prompt
            all_results.append(
                {
                    "prompt": prompt,
                    "masks": prompt_state["masks"],
                    "masks_logits": prompt_state["masks_logits"],
                    "boxes": prompt_state["boxes"],
                    "scores": prompt_state["scores"],
                }
            )

        return all_results
