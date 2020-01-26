import sys
from kb import KB, Boolean, Integer, Constant

'''
Question:
Repeat the previous process to prove or disprove exercise 7 from the worksheet 
whether (P↔¬Q) is entailed by the knowledge base (P∨Q)∧(Q→R)∧(R→¬P).
Provide the clauses you added to the script, and a screenshot or copy the result.

KB:
P v Q
~Q v R
~R v ~P

entailment:
P <-> ~Q
= P -> ~Q ^ ~Q -> P
= (~P v ~Q) ^ (Q v P)

neg:
~((~P v ~Q) ^ (Q v P))
= ~(~P v ~Q) v ~(Q v P)
= ~(~P v ~Q) v ~(Q v P)
= ...


uitwerking:
P v ~Q
Q v ~P 
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
# Negated conclusion
kb.add_clause(P, ~Q)
kb.add_clause(Q, ~P)

# Print all models of the knowledge base
for model in kb.models():
    print(model)

# Print out whether the KB is satisfiable (if there are no models, it is not satisfiable)
print(kb.satisfiable())
