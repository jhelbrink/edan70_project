from docria.storage import DocumentIO
from docria.algorithm import dfs

reader = DocumentIO.read("squad10/dev.docria")

# The first text
n1 = next(reader)
print(n1)

# The second one
n2 = next(reader)
print(n2)

# Let us print the first context (a paragraph that covers a set of questions and their answers)
text = n1.texts['context']
print(text)
# And the questions
questions = n1.texts['questions']
print(questions)
# And the answers
answers = n1.layers['answers'][0]['text']
print(answers)

# The structure
n1.printschema()

# Accessing the answers to a specific question
print(n1.layers['questions'][0]['answers'][0]['text'])

# And the first sentence of the first context
sentence = n1.layers['contexts'][5]['sentences'][0]['text']
print(sentence)

# We can access all the sentences too
sentence = n1.layers['contexts/sentence'][0]['text']
print(sentence)

# And the parse tree of the first sentence in the first question
root_node = n1["questions"][0]["sentences"][0]["parse_tree"]
print(root_node)

# Let us traverse it and print the leaves using a deep-first search (dfs)
# Longer
leaves = list(map(lambda n: n["value"],
                  list(dfs(root_node, lambda n: n["children"] if "children" in n else [],
                           lambda n: "children" not in n))))
print(leaves)

# Shorter
from docria.algorithm import get_prop

leaves = list(map(lambda n: n["value"], list(
    dfs(n1["questions"][0]["sentences"][0]["parse_tree"], get_prop("children", []), lambda n: "children" not in n))))
print(leaves)

# Beware: There might out of order words and we should check the indexes.
