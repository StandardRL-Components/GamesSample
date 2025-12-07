def policy(env):
    # This policy maximizes reward by leveraging perfect knowledge of card values to always match pairs with minimal movement.
    # It minimizes moves by targeting the closest unmatched pair, ensuring optimal pathing and avoiding unnecessary flips.
    if env.game_over or env.mismatch_pause_counter > 0:
        return [0, 0, 0]
    
    current_col, current_row = env.cursor_pos
    unmatched_cards = [card for card in env.cards if not card['is_matched']]
    
    if env.first_selection is not None:
        target_value = env.first_selection['value']
        target_card = next((card for card in unmatched_cards if card['value'] == target_value and card['id'] != env.first_selection['id']), None)
        if target_card is None:
            return [0, 0, 0]
        target_col, target_row = target_card['col'], target_card['row']
        if current_col == target_col and current_row == target_row:
            return [0, 1, 0]
        if current_col < target_col:
            return [4, 0, 0]
        elif current_col > target_col:
            return [3, 0, 0]
        else:
            if current_row < target_row:
                return [2, 0, 0]
            else:
                return [1, 0, 0]
    
    value_groups = {}
    for card in unmatched_cards:
        if card['value'] not in value_groups:
            value_groups[card['value']] = []
        value_groups[card['value']].append(card)
    
    best_pair = None
    min_cost = float('inf')
    for value, cards in value_groups.items():
        if len(cards) < 2:
            continue
        card1, card2 = cards[0], cards[1]
        cost = abs(current_col - card1['col']) + abs(current_row - card1['row']) + abs(card1['col'] - card2['col']) + abs(card1['row'] - card2['row'])
        if cost < min_cost:
            min_cost = cost
            best_pair = card1
    
    if best_pair is None:
        return [0, 0, 0]
    
    target_col, target_row = best_pair['col'], best_pair['row']
    if current_col == target_col and current_row == target_row:
        return [0, 1, 0]
    if current_col < target_col:
        return [4, 0, 0]
    elif current_col > target_col:
        return [3, 0, 0]
    else:
        if current_row < target_row:
            return [2, 0, 0]
        else:
            return [1, 0, 0]