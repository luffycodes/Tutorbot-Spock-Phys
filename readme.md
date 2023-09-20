## Code Soliloquies for Reliable Calculations

This repository contains code for the paper: [Code Soliloquies for Reliable Calculations](https://arxiv.org/abs/2305.13272)

### Inference
To use the model, first install the [fastchat](https://github.com/lm-sys/FastChat/) library, and then follow the steps here:
1. Update the [conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py) from our [conversation_inference](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/fastchat/conversation_inference.py) repository in the FastChat folder.
2. Update the [inference.py](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/fastchat/inference.py) from our repository in the FastChat folder.
3. Then run the model using the following command:
      - ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m fastchat.serve.cli --model-path /data/ss164/spock_physics/noether-llama-vicuna-ep25-70b/ --num-gpus 8 --temperature 0.7 --max-gpu-memory 20GiB```
      - Note that code soliloquy is not implemented in UI

### Creating synthetic conversational dataset to train Higgs
#### Example of generating conversational dataset using GPT
1. Run the [generate_conversations.py](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/prompts/conversation_gen/generate_conversations.py) 
2. Remember to put [openai.api_key](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/prompts/conversation_gen/generate_conversations.py#L14)
3. Use the training instructions from [fastchat](https://github.com/lm-sys/FastChat/) library and run [train_higgs_lora.py](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/fastchat/train_higgs_lora.py).


If you use this work, please cite:
Code Soliloquies for Reliable Calculations
https://arxiv.org/abs/2305.13272
```
@misc{sonkar2023class,
      title={CLASS Meet SPOCK: An Education Tutoring Chatbot based on Learning Science Principles}, 
      author={Shashank Sonkar and Lucy Liu and Debshila Basu Mallick and Richard G. Baraniuk},
      year={2023},
      eprint={2305.13272},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

