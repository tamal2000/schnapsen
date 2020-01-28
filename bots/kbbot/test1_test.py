import sys
from kb import KB, Boolean, Integer, Constant

# Define our symbols
p = Boolean('p')
q = Boolean('q')
r = Boolean('r')

# Create a new knowledge base
kb = KB()

# Add clauses
kb.add_clause(p,~q)
kb.add_clause(p,r)
kb.add_clause(p)
kb.add_clause(~q)
kb.add_clause(r)



# Print all models of the knowledge base
for model in kb.models():
    print(model)

# Print out whether the KB is satisfiable (if there are no models, it is not satisfiable)
print(kb.satisfiable())
