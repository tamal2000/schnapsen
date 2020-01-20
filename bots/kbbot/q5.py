import sys
from kb import KB, Boolean, Integer, Constant

'''
(P↔¬Q) is entailed by the knowledge base 
KB
(P∨Q)∧(Q→R)∧(R→¬P)

'''

# Define our symbols
P = Boolean('P')
Q = Boolean('Q')
R = Boolean('R')

# Create a new knowledge base
kb = KB()

# Add clauses
kb.add_clause(P, Q)
kb.add_clause(~Q, R)
kb.add_clause(~R, ~P)
kb.add_clause(~A, ~C, ~D)

# Print all models of the knowledge base
for model in kb.models():
    print(model)

# Print out whether the KB is satisfiable (if there are no models, it is not satisfiable)
print(kb.satisfiable())
