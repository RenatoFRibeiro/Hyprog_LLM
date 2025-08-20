#imports
import re

# tokenizing the hobbit book
with open("Hyprog_LLM\dataset\Hobbit.txt", "r", encoding="utf-8") as f:
    raw_hobbit = f.read()

split_hobbit = re.split(r'([,.:;?_!"()\']|--|\s)', raw_hobbit)

tokenized_hobbit = [
    item for item in split_hobbit 
    if (item is not None and 
        isinstance(item, str) and 
        item.strip())
]

print(len(tokenized_hobbit))
