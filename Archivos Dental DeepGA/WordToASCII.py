import numpy as np

def word_to_ascii(word):
    ascii_array = np.array([ord(char) for char in word])
    return ascii_array

# Clasificate categories:

class1 = "Necrosis pulpar"
class2 = "Previamente iniciado"
class3 = "Previamente tratado"
class4 = "Pulpa normal"
class5 = "Pulpitis irreversible asintomatica" #34
class6 = "Pulpitis irreversible sintomatica"
class7 = "Pulpitis reversible"

print("Nectrosis pulpar:", word_to_ascii(class1))
print("Previamente iniciado:", word_to_ascii(class2))
print("Previamente tratado:", word_to_ascii(class3))
print("Pulpa normal:", word_to_ascii(class4))
print("Pulpitis irreversible asintomatica:", word_to_ascii(class5))
print("Pulpitis irreversible sintomatica:", word_to_ascii(class6))
print("Pulpitis reversible:", word_to_ascii(class7))