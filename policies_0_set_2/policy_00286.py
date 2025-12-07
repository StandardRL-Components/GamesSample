def policy(env):
    # Strategy: Place words in correct sentence order by selecting next needed word, moving it to right of placed words, then down into sentence area. Submit only when all words are placed in correct order to maximize reward from successful submissions while avoiding penalties.
    sentence_words = env.current_sentence_text.split()
    placed_tiles = sorted([t for t in env.word_tiles if t['is_placed']], key=lambda t: t['pos'].x)
    placed_words = [t['text'] for t in placed_tiles]
    
    if placed_words == sentence_words:
        if not env.last_shift_held:
            return [0, 0, 1]
        return [0, 0, 0]
    
    next_word_index = len(placed_words)
    if next_word_index >= len(sentence_words):
        return [0, 0, 1] if not env.last_shift_held else [0, 0, 0]
    next_word = sentence_words[next_word_index]
    
    desired_tile_index = None
    for i, tile in enumerate(env.word_tiles):
        if tile['text'] == next_word and not tile['is_placed']:
            desired_tile_index = i
            break
            
    if desired_tile_index is None:
        return [0, 0, 1] if not env.last_shift_held else [0, 0, 0]
        
    if env.selected_tile_index is None:
        return [0, 1, 0]
        
    if env.selected_tile_index != desired_tile_index:
        return [0, 1, 0]
        
    tile = env.word_tiles[env.selected_tile_index]
    if placed_tiles:
        target_x = placed_tiles[-1]['pos'].x + placed_tiles[-1]['rect'].width + 5
    else:
        target_x = env.sentence_area.left + 10
        
    if tile['pos'].x < target_x:
        return [4, 0, 0]
    elif tile['pos'].x > target_x:
        return [3, 0, 0]
    else:
        return [2, 0, 0]