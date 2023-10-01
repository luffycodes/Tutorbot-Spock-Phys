## Code Soliloquies for Accurate Calculations in Large Language Models

This repository contains code for the paper: Code Soliloquies for Accurate Calculations in Large Language Models

Model: https://huggingface.co/anonuseranonuser/tutorbot-spock-phys

**Demo:**
![](https://github.com/luffycodes/Tutorbot-Spock-Phys/blob/main/higgs-demo-github.gif)

### Inference
To use the model, first install the [fastchat](https://github.com/lm-sys/FastChat/) library, and then follow the steps here:
1. Update the [conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py) from our [conversation_inference](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/fastchat/conversation_inference.py) repository in the FastChat folder.
2. Update the [inference.py](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/fastchat/inference.py) from our repository in the FastChat folder.
3. Then run the model using the following command:
      - ```
        python3 -m fastchat.serve.cli --model-path anonuseranonuser/tutorbot-spock-phys --num-gpus 8 --temperature 0.7 --max-gpu-memory 20GiB
        ```
      - Note that code soliloquy is not implemented in UI

### Creating synthetic conversational dataset to train Higgs
#### Example of generating conversational dataset using GPT
1. Run the [generate_conversations.py](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/prompts/conversation_gen/generate_conversations.py).
2. Remember to put [openai.api_key](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/prompts/conversation_gen/generate_conversations.py#L14).
3. Use the training instructions from [fastchat](https://github.com/lm-sys/FastChat/) library and run [train_higgs_lora.py](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/fastchat/train_higgs_lora.py).

### Descriptions of GPT-Tutorbot state prompts used to implement code soliloquy in [conversation_gen](https://anonymous.4open.science/r/Tutorbot-Spock-Phys-FC5C/prompts/conversation_gen)

- deciding_state.txt = The 'deciding_state' is the first state of code soliloquy in which the GPT-tutorbot determines whether a calculation is needed for its response to the student. If it determines a calculation is needs is needed, it generates the description for Python code for performing that calculation and gpt-tutorbot transitions to 'use_python_state'; if it decides that no calculation is needed, the gpt-tutorbot transitions to 'no_python_state'.
- use_python_state.txt = In this state, gpti-tutorbot generates Python code based on Python code description generated during 'deciding_state'.
- received_python_state.txt = This is the final state of gpt-tutorbot soliloquy (when it is using python) in which it generate tutorbot's response based on Python code output.
- no_python_state.txt = This is the final state of gpt-tutorbot soliloquy (when it is not using python) in which it generate gpt-tutorbot's response.



If you use this work, please cite:
Code Soliloquies for Accurate Calculations in Large Language Models

