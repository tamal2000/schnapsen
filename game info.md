**Get the size of the stock:**
Let 'state' be the state you're given and let's say you want the size of the stock. Then the following a should do the trick:

size_of_stock = state.get_stock_size()


**Find out if I'm player 1 or 2**

me = state.whose_turn()

**Print the (abbreviated) cards in your hand**


cards_hand = state.hand()

for i, card in enumerate(cards_hand):

	rank, suit = util.get_card_name(card)

	print('Card {} in the hand is {} of {}'.format(i, rank,suit))
	
