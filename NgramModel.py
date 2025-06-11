import os
import re
import nbformat
from collections import defaultdict, Counter
import random

print("[NGRAM MODEL LOADING]", "-"*30)
def read_python_files(folder_path):
    code_data = []
    for filename in os.listdir(folder_path):

        if filename.endswith(".py"):  # Only process Python files
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                code_data.append(file.read())


        if filename.endswith(".ipynb"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                #print(os.path.join(folder_path, filename))
                try:
                  notebook = nbformat.read(file, as_version=4)
                except:
                  pass

            python_code = []
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    python_code.append(cell.source)
            all_code = '\n\n'.join(python_code)
            code_data.append(all_code)

    return code_data


folder_path = "Dataset"
python_code_list = read_python_files(folder_path)
#print(f"Loaded {len(python_code_list)} Python files.")


''' Remove Comments and docstring '''


def remove_comments_and_docstrings(code):
    # Remove docstrings
    code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
    # This removes multiline comments when """ ''' are used , and re.DOTALL ensures multiline i.e multiple lines

    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    return code

clean_code_list = [remove_comments_and_docstrings(code) for code in python_code_list]

clean_code = "\n".join(clean_code_list)
#print(len(clean_code))

''' Tokenize the cleaned data'''
import tokenize
from io import BytesIO

def tokenize_python_code(code):

    tokens = []
    try:
        token_generator = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
        for tok in token_generator:
            if tok.type in {tokenize.ENCODING, tokenize.NEWLINE, tokenize.ENDMARKER}:
                continue  # Skip unnecessary tokens
            tokens.append(tok.string)  # Append token string
    except Exception as e:
        print(f"Error tokenizing code: {e}")
    return tokens

tokens = tokenize_python_code(clean_code)
#print(tokens[:50])


class NGramModel:
    def __init__(self, n=5):
        self.n = n  # Only 5-gram model with backoff to lower orders
        self.models = {i: defaultdict(Counter) for i in range(1, n + 1)}  # 1-gram to 5-gram

    def train(self, tokenized_code):
        """Train 1-gram to 5-gram models."""
        for order in range(1, self.n + 1):
            for i in range(len(tokenized_code) - order + 1):
                context = tuple(tokenized_code[i:i + order - 1])
                next_token = tokenized_code[i + order - 1]
                self.models[order][context][next_token] += 1

    def _get_next_token(self, context):
        """Backoff from 5-gram to unigram to find a next token."""
        for order in reversed(range(1, self.n + 1)):
            sub_context = tuple(context[-(order - 1):]) if order > 1 else ()
            if sub_context in self.models[order]:
                next_token = random.choices(
                    list(self.models[order][sub_context].keys()),
                    weights=self.models[order][sub_context].values()
                )[0]
                return next_token
        return None  # No valid token found

    def generate(self, start_tokens, length=20):
        """Generate code using 5-gram model with backoff."""
        output = list(start_tokens)
        op = []
        for _ in range(length):
            context = output[-(self.n - 1):]  # Last 4 tokens for 5-gram
            next_token = self._get_next_token(context)
            if next_token:
                output.append(next_token)
                op.append(next_token)
            else:
                break  # Stop if no continuation found
        return ' '.join(op) + " "

ngram_model = NGramModel(n=5)
ngram_model.train(tokens)
print("[NGRAM] Model trained successfully!")


if __name__ == "__main__":
    start_sequence = 'import matplotlib.pyplot'
    generated_code = ngram_model.generate(start_sequence, length=5)
    print(len)
    print("Generated Code:\n", generated_code)