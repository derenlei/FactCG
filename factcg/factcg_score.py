from .inference import Inferencer
from typing import List


class FactCGScore:
    def __init__(self, model_name: str = 'microsoft/deberta-v3-large', batch_size: int = 32, ckpt_path: str = None, verbose=True, use_hf_ckpt=True) -> None:
        if 'deberta' not in model_name.lower() and use_hf_ckpt:
            raise ValueError(
                "We only support deberta-v3-large for huggingface checkpoint")
        self.model = Inferencer(
            ckpt_path=ckpt_path,
            model_name=model_name,
            batch_size=batch_size,
            verbose=verbose,
            use_hf_ckpt=use_hf_ckpt
        )

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.eval(contexts, claims)[0].tolist()
