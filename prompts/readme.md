### Descriptions of GPT-Tutorbot state prompts in [conversation_gen](https://github.com/luffycodes/Tutorbot-Spock-Phys/tree/main/prompts/conversation_gen)

- initial_state.txt = The very first state in the conversation, used to prompt the gpt-tutorbot to start the conversation
- deciding_state.txt = The 'deciding_state' is the first state of code soliloquy in which the GPT-tutorbot determines whether a calculation is needed for its response to the student. If it determines a calculation is needs is needed, it generates the description for Python code for performing that calculation and gpt-tutorbot transitions to 'use_python_state'; if it decides that no calculation is needed, the gpt-tutorbot transitions to 'no_python_state'.
- use_python_state.txt = In this state, gpti-tutorbot generates Python code based on Python code description generated during 'deciding_state'.
- received_python_state.txt = This is the final state of gpt-tutorbot soliloquy (when it is using python) in which it generate tutorbot's response based on Python code output.
- no_python_state.txt = This is the final state of gpt-tutorbot soliloquy (when it is not using python) in which it generate gpt-tutorbot's response.
