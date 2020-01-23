import sys
from kb import KB, Boolean, Integer, Constant

# Define our propositional symbols
# J1 is true if the card with index 1 is a jack, etc
# You need to initialise all variables that you need for you strategies and game knowledge.
# Add those variables here.. The following list is complete for the Play Jack strategy.
J0 = Boolean('j0')
J1 = Boolean('j1')
J2 = Boolean('j2')
J3 = Boolean('j3')
J4 = Boolean('j4')
J5 = Boolean('j5')
J6 = Boolean('j6')
J7 = Boolean('j7')
J8 = Boolean('j8')
J9 = Boolean('j9')
J10 = Boolean('j10')
J11 = Boolean('j11')
J12 = Boolean('j12')
J13 = Boolean('j13')
J14 = Boolean('j14')
J15 = Boolean('j15')
J16 = Boolean('j16')
J17 = Boolean('j17')
J18 = Boolean('j18')
J19 = Boolean('j19')

PJ0 = Boolean('pj0')
PJ1 = Boolean('pj1')
PJ2 = Boolean('pj2')
PJ3 = Boolean('pj3')
PJ4 = Boolean('pj4')
PJ5 = Boolean('pj5')
PJ6 = Boolean('pj6')
PJ7 = Boolean('pj7')
PJ8 = Boolean('pj8')
PJ9 = Boolean('pj9')
PJ10 = Boolean('pj10')
PJ11 = Boolean('pj11')
PJ12 = Boolean('pj12')
PJ13 = Boolean('pj13')
PJ14 = Boolean('pj14')
PJ15 = Boolean('pj15')
PJ16 = Boolean('pj16')
PJ17 = Boolean('pj17')
PJ18 = Boolean('pj18')
PJ19 = Boolean('pj19')

# Create a new knowledge base
kb = KB()

# 0, 5, 10, 15 - Aces
# 1, 6, 11, 16 - 10s
# 2, 7, 12, 17 - Kings
# 3, 8, 13, 18 - Queens
# 4, 9, 14, 19 - Jacks

def general_information(kb):
	"""This adds information which cards are cheap, i.e. Kings, Queens and 
	Jacks.
	
	Args:
		kb (KB): The kb to add the clauses to
	"""

	kb.add_clause(J2)
	kb.add_clause(J3)
	kb.add_clause(J4)
	kb.add_clause(J7)
	kb.add_clause(J8)
	kb.add_clause(J9)
	kb.add_clause(J12)
	kb.add_clause(J13)
	kb.add_clause(J14)
	kb.add_clause(J17)
	kb.add_clause(J18)
	kb.add_clause(J19)

def strategy_knowledge(kb):
	"""DEFINITION OF THE STRATEGY
	The strategy to play cheap cards first.
	For all x, PJ(x) <-> J(x)
	
	Args:
		kb (KB): the kb to add the clauses to
	"""
	kb.add_clause(~PJ2, J2)
	kb.add_clause(~J2, PJ2)

	kb.add_clause(~PJ3, J3)
	kb.add_clause(~J3, PJ3)

	kb.add_clause(~PJ4, J4)
	kb.add_clause(~J4, PJ4)

	kb.add_clause(~PJ7, J7)
	kb.add_clause(~J7, PJ7)

	kb.add_clause(~PJ8, J8)
	kb.add_clause(~J8, PJ8)

	kb.add_clause(~PJ9, J9)
	kb.add_clause(~J9, PJ9)

	kb.add_clause(~PJ12, J12)
	kb.add_clause(~J12, PJ12)

	kb.add_clause(~PJ13, J13)
	kb.add_clause(~J13, PJ13)

	kb.add_clause(~PJ14, J14)
	kb.add_clause(~J14, PJ14)

	kb.add_clause(~PJ17, J17)
	kb.add_clause(~J17, PJ17)

	kb.add_clause(~PJ18, J18)
	kb.add_clause(~J18, PJ18)

	kb.add_clause(~PJ19, J19)
	kb.add_clause(~J19, PJ19)

general_information(kb)
strategy_knowledge(kb)

kb.add_clause(PJ2)


print('kb:', kb)
# print all models of the knowledge base
for model in kb.models():
    print(model)

# print out whether the KB is satisfiable (if there are no models, it is not satisfiable)
print(kb.satisfiable())
