from transformers import AutoTokenizer, AutoModelForCausalLM

#only for testing - we should use framework like lamma.cpp (for testing cause no GPUS) or some better gpu optimized library (or just fck it cause for our use case it probably odesnt matter much) 

#doesnt fit
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

from transformers import AutoTokenizer, AutoModelForCausalLM

#super small and super stupid
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

#how to prompt efficiently
messages = [

    {"role": "user", "content": "You are a psychologist talking with a user of an app for psychological consulting. User tells you what's troubling him and you try to help him. You have information about his emotional state inferred from the way he talks [SPEECH EMOTION DETECTION - SEMANTIC+ACOUSTIC] and his expression [VIDEO EMOTION DETECTION]. You are also given a guide how to deal with a specific topic the user wants to talk about. [RAG]. User prompt: I drank a lot of alcohol yesterday and I feel guilty about it. Visible Emotion: Sad. Do not generate a conversation, only answer users prompt."},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

generated_ids = model.generate(model_inputs, max_new_tokens=300, do_sample=True)

print(tokenizer.batch_decode(generated_ids)[0])