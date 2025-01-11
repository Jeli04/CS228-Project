# Multimodal Differential Transformer
<div align="center">
    <img src="images/overview.png" alt="Project Overview" width="800">
</div>

Credit to Dalle

[View the Project PDF](path/to/project-overview.pdf)

## Brief Overview
This project is orignally our CS228 (Deep Learning) final project. We explored the integration of [Differential Attention](https://arxiv.org/pdf/2410.05258) into the text-vision model **PaliGemma 3B** to address challenges posed by noisy information and limited context windows. We utilized LoRA fine-tuning, and adpated/modified the Differential Attention into an existing pretrained model for finetuning. Based on the first iteration of experiments, we demonstrated potential improvements over a baseline vanilla finetune on the [Multimodal Needle In Haystack Evaluation](https://arxiv.org/pdf/2406.11230). Further information can be found in our report linked below. There is still plans to explore this project more through better evaluations and possibly expanding to the Phi model family. 

<div align="center" style="display: flex; justify-content: center; gap: 20px;">

<a href="https://drive.google.com/file/d/1PVzOSapdhu_CjMikqycHYtwpx-iCsPE1/view?usp=sharing" style="text-decoration: none; font-size: 16px; font-weight: bold; background-color: #f2f2f2; padding: 10px 20px; border-radius: 5px; border: 1px solid #ccc;">
📄 Project Report
</a>

<a href="https://www.youtube.com/watch?v=vAmKB7iPkWw" style="text-decoration: none; font-size: 16px; font-weight: bold; background-color: #f2f2f2; padding: 10px 20px; border-radius: 5px; border: 1px solid #ccc;">
🎥 Reference Source Code
</a>

</div>

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

