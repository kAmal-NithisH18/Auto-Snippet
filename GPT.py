import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2_codegen").to("cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2_codegen")

"""model_name = "gpt2-medium"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)"""
print("[GPT] Model loaded succesfully!")
def fintuned_GPT(input_text):

    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")

    output = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(input_text):]


def GPT_nofintune(input_text):
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device.type)
    output = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(input_text):]

if __name__ == "__main__" :
    print(fintuned_GPT("def add(a,b)"))