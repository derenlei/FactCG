# FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data

Pytorch implementation of our EMNLP 2025 paper: [FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data](https://arxiv.org/pdf/2501.17144).

We propose a fact-checker to detect Large language Model ungrounded hallucinations and a synthetic data generation method to collect high quality training data.

We leverage LLM to convert text documents into content graphs. Graph data, in semi-structured form, is easier to manipulate in both neural and symbolic ways than text. Thus allow us to create high quality and high granularity syhthetic data with high control.

<p align="center"><img src="figs/cg2c_doc.png" width="1000"/></p>

Prior research on training grounded factuality classification models to detect hallucinations in large language models (LLMs) has relied on public natural language inference (NLI) data and synthetic data. However, conventional NLI datasets are not well-suited for document-level reasoning, which is critical for detecting LLM hallucinations. Recent approaches to document-level synthetic data generation involve iteratively removing sentences from documents and annotating factuality using LLM-based prompts. While effective, this method is computationally expensive for long documents and limited by the LLM's capabilities. In this work, we analyze the differences between existing synthetic training data used in state-of-the-art models and real LLM output claims. Based on our findings, we propose a novel approach for synthetic data generation, CG2C, that leverages multi-hop reasoning on context graphs extracted from documents. Our fact checker model, \ours, demonstrates improved performance with more connected reasoning, using the same backbone models. Experiments show it even outperforms GPT-4-o on the LLM-A{\small GGRE}F{\small ACT} benchmark with much smaller model size.

If you find the repository or FactCG helpful, please cite the following paper
```bibtex
@inproceedings{lei2025factcg,
  title={FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data},
  author={Lei, Deren and Li, Yaxi and Li, Siyao and Hu, Mengya and Xu, Rui and Archer, Ken and Wang, Mingyu and Ching, Emily and Deng, Alex},
  journal={NAACL},
  year={2025}
}
```

## Training
To reproduce FactCG-DBT with 2-stage  training
```
sh train.sh
```

## Evaluation
1.  Evaluate on LLM-Aggrefact Benchmark
```
./benchmark.sh
```
Note:
* you can evaulate different fact-checkers: `FactCG`, `Minicheck`, `AlignScore`, `SummaC-ZS` and `SummaC-CV`
* you can choose `threshold_setting` as `tune` for selecting the best threshold per best dev set performance in LLM-Aggrefact, or as `fixed` for fixing threshold to 0.5.

2.  Evaluation on Connected Reasoning

```
python3 wice_connected_reasoning.py --method factcg
```
