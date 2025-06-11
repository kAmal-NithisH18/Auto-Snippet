from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
print("[CODET5] loaded succesfully!")
#Python code prefix
prefix = """def calculate_area(radius):
    pi = 3.14
    area ="""


def codeT5_generate(prefix):
    print("called")
    input_text = f"complete: {prefix}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)


    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )


    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

if __name__ == "__main__":
    print(codeT5_generate(prefix))