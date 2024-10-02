# Large Language Models Can Self-Correct with Key Condition Verification (EMNLP 2024)

## Introduction & Setup

This repository contains the code for the paper [Large Language Models Can Self-Correct with Key Condition Verification](https://arxiv.org/abs/2405.14092) (Accepted to EMNLP Main 2024). ProCo first prompts an LLM to generate an initial response, then iterates a verify-then-correct process to progressively identify and correct (probably) false responses.
Extensive experiments on eight complex reasoning datasets demonstrate the effectiveness and efficiency of ProCo.

![image](https://github.com/wzy6642/ProCo/blob/main/framework.png)


 - Run `run.py` to generate the answer to the given question

```python
python run.py --data_index 0
```

## Experimental Results

![image](https://github.com/wzy6642/ProCo/blob/main/experiments.png)


## Citing ProCo
```markdown
@inproceedings{wu2024proco,
  title={Large Language Models Can Self-Correct with Key Condition Verification}, 
  author={Zhenyu Wu and Qingkai Zeng and Zhihan Zhang and Zhaoxuan Tan and Chao Shen and Meng Jiang},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024},
}
```

## License

This project is licensed under the Apache-2.0 License.
