## test id

with open("debates.txt", "r") as f:
    text = ''.join([line.replace("\n", "  ") for line in f])

print(text)
print("--------")
print(text[33:74])
print(text[643:742])
print(text[744:871])
print(text[569:625])
print(text[3361:3435])

print(len(text))
