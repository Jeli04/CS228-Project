# Multimodal Differential Transformer

<div align="center">
    <img src="images/overview.png" alt="Project Overview" width="800">
</div>

*Credit to DALL-E*

---

<div align="center" style="margin: 20px 0;">
    <a href="https://drive.google.com/file/d/1PVzOSapdhu_CjMikqycHYtwpx-iCsPE1/view?usp=sharing" style="text-decoration: none; font-size: 16px; font-weight: bold; padding: 10px 20px; background-color: #007acc; color: white; border-radius: 5px;">
        📄 Project Report
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://www.youtube.com/watch?v=vAmKB7iPkWw" style="text-decoration: none; font-size: 16px; font-weight: bold; padding: 10px 20px; background-color: #007acc; color: white; border-radius: 5px;">
        🎥 Reference Source Code/Video
    </a>
</div>

---

## Brief Overview

This project is originally our CS228 (Deep Learning) final project. We explored the integration of [Differential Attention](https://arxiv.org/pdf/2410.05258) into the text-vision model **PaliGemma 3B** to address challenges posed by noisy information and limited context windows. 

We utilized LoRA fine-tuning and adapted/modified Differential Attention into an existing pretrained model for fine-tuning. Based on the first iteration of experiments, we demonstrated potential improvements over a baseline vanilla fine-tune on the [Multimodal Needle In Haystack Evaluation](https://arxiv.org/pdf/2406.11230).

Further information can be found in our report linked above. There are plans to explore this project more through better evaluations and possibly expanding to the Phi model family.

---

### ✅ Todo
  - [ ] Reconduct further evaluations
  - [ ] Experiment with Phi3

## Installation

1. **Prerequisites**  
   ```bash
   # Better in conda env
   pip install -r requirements.txt

## Experiments
    Our main modifications to the model can be found in modeling_gemma.py and modeling_siglip.py. 

1. **Finetune Original Base Model**  
   ```bash
   python3 finetune_original.py
2. **Finetune Our Model**  
   ```bash
   python3 finetune_original.py


## Experiments
    Our main modifications to the model can be found in modeling_gemma.py and modeling_siglip.py. 

1. **Finetune Original Base Model**  
   ```bash
   python3 finetune_original.py
2. **Finetune Our Model**  
   ```bash
   python3 finetune_original.py


## Evaluation

1. **TODO**  

