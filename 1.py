#imports
import re


with open("Hyprog_LLM\dataset\Hobbit.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
#print("Total number of character:", len(raw_text))

raw_text_test = raw_text[:52]

#print(raw_text_test)

result = re.split(r'([,.])|\s', raw_text)

print(result)