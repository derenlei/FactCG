from .inference import Inferencer
from typing import List


class FactCGScore:
    def __init__(self, model_name: str, batch_size: int, ckpt_path: str, verbose=True) -> None:
        self.model = Inferencer(
            ckpt_path=ckpt_path,
            model_name=model_name,
            batch_size=batch_size,
            verbose=verbose
        )

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.eval(contexts, claims)[0].tolist()
