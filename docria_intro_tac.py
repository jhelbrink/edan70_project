"""
A small tour with docria
"""

from docria.storage import DocumentIO

# We read the eval file of TAC 2017
doc = DocumentIO.read('tac_docria/en/eng.2017.eval.docria')

# We print it
print(doc)

# We print its content
print(list(doc))
# Twice
print(list(doc))
# Nothing. It is an iterator...

# We read it again
doc = DocumentIO.read('tac_docria/en/eng.2017.eval.docria')

# We access the next element
a = next(doc)
# and print it
print(a)
# We know its content: The Texts in main and the Layers.
# Let us access the text
print(a.texts['main'])

# What are the layers
print(a.layers.keys())
# And one layer
print(a.layers['tac/entity/gold'])

# Let us print the schemas
a.printschema()
