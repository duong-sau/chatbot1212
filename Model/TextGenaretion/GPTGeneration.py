from transformers import pipeline
if __name__ == '__main__':
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    text = generator("import pandas as pd \n import numpy as np\n " , do_sample=True, min_length=200)
    print(text)