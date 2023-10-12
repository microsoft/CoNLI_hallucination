# Text Hallucination Detection and Reduction: CoNLI

### We will soon release the code and datasets

Python implementation of our paper: [Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations](https://arxiv.org/abs/2310.03951).

We propose a generic post-edit framework based on OpenAI GPT to effectively detect and mitigate hallucinations.
<p align="center"><img src="fig/CoNLI.png" width="850"/></p>

Large language models (LLMs) can generate fluent natural language texts when given relevant documents as background context. This ability has attracted considerable interest in developing industry applications of LLMs. However, LLMs are prone to generate hallucinations that are not supported by the provided sources. In this paper, we propose a hierarchical framework to detect and mitigate such ungrounded hallucination. Our framework uses Chain of Natural Language Inference (CoNLI) for hallucination detection and hallucination reduction via post-editing. Our approach achieves state-of-the-art performance on hallucination detection and enhances text quality through rewrite, using LLMs without any fine-tuning or domain-specific prompt engineering. We show that this simple plug-and-play framework can serve as an effective choice for hallucination detection and reduction, achieving competitive performance across various contexts.

If you find the repository or CoNLI helpful, please cite the following paper
```bibtex
@article{lei2023chain,
  title={Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations},
  author={Lei, Deren and Li, Yaxi and Hu, Mengya and Wang, Mingyu and Yun, Vincent and Ching, Emily and Kamal, Eslam and others},
  journal={arXiv preprint arXiv:2310.03951},
  year={2023}
}
```
