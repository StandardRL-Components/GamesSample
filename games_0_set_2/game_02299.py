
# Generated: 2025-08-28T04:24:06.133453
# Source Brief: brief_02299.md
# Brief Index: 2299

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Hold SHIFT to move the cursor on the grid, otherwise it moves on your letter hand. Press SPACE to select a letter or place it on the grid."
    )

    game_description = (
        "A strategic word-building puzzle. Place letters from your hand onto the 10x10 grid to form words. Form 5 words before you run out of 10 moves to win."
    )

    auto_advance = False

    # A built-in dictionary to avoid file I/O
    WORD_LIST = {
        'ACE', 'ACT', 'ADD', 'ADOPT', 'ADS', 'AFT', 'AGE', 'AGO', 'AID', 'AIM', 'AIR', 'ALL', 'ALSO', 'AMP', 'AND', 'ANT',
        'ANY', 'APE', 'APT', 'ARC', 'ARE', 'ARK', 'ARM', 'ART', 'ASH', 'ASK', 'AUNT', 'AWAKE', 'BAD', 'BAG', 'BAN', 'BAT',
        'BEE', 'BEG', 'BET', 'BID', 'BIG', 'BIN', 'BIT', 'BITE', 'BOAT', 'BOG', 'BOND', 'BOOK', 'BOOM', 'BOT', 'BOTH', 'BOW',
        'BOX', 'BOY', 'BUG', 'BUN', 'BUS', 'BUT', 'BUY', 'BYE', 'CAB', 'CALL', 'CAN', 'CAP', 'CAR', 'CARD', 'CAT', 'CAVE',
        'COB', 'COD', 'COIN', 'COLD', 'CON', 'COO', 'COP', 'COST', 'COT', 'COW', 'CRY', 'CUB', 'CUP', 'CURB', 'CUT', 'DAD',
        'DAM', 'DAY', 'DEAL', 'DEEP', 'DEER', 'DEN', 'DEW', 'DID', 'DIE', 'DIG', 'DIM', 'DIP', 'DO', 'DOE', 'DOG', 'DOLL',
        'DOOM', 'DOT', 'DRAG', 'DREAM', 'DRY', 'DUB', 'DUD', 'DUE', 'DUG', 'DUMP', 'DUST', 'EACH', 'EAR', 'EARN', 'EAST',
        'EAT', 'EBB', 'EGG', 'EGO', 'ELF', 'ELK', 'END', 'ERA', 'ETC', 'EVEN', 'EVER', 'EXIT', 'EYE', 'FACE', 'FACT', 'FAD',
        'FADE', 'FALL', 'FAN', 'FAR', 'FARM', 'FAST', 'FAT', 'FATE', 'FAWN', 'FED', 'FEE', 'FEEL', 'FEW', 'FIB', 'FIG',
        'FILL', 'FILM', 'FIND', 'FIN', 'FIRE', 'FIRM', 'FISH', 'FIT', 'FIVE', 'FLAG', 'FLAP', 'FLEW', 'FLIP', 'FLY', 'FOG',
        'FOIL', 'FOLD', 'FONT', 'FOOD', 'FOR', 'FORK', 'FORM', 'FOUR', 'FOX', 'FROG', 'FROM', 'FUEL', 'FUN', 'FUR', 'GAG',
        'GAP', 'GAS', 'GEL', 'GEM', 'GET', 'GIFT', 'GIG', 'GIVE', 'GLAD', 'GLOW', 'GLUE', 'GO', 'GOAD', 'GOAL', 'GOAT',
        'GOLD', 'GONE', 'GOOD', 'GOT', 'GRAB', 'GRID', 'GRIN', 'GROW', 'GUM', 'GUN', 'GUT', 'GUY', 'GYM', 'HAD', 'HAG',
        'HALF', 'HAM', 'HAS', 'HAT', 'HAVE', 'HAY', 'HE', 'HEAL', 'HEAR', 'HEAT', 'HELP', 'HEM', 'HEN', 'HER', 'HERE',
        'HEY', 'HID', 'HIDE', 'HIGH', 'HIM', 'HIP', 'HIS', 'HIT', 'HOG', 'HOLD', 'HOLE', 'HOME', 'HOP', 'HOPE', 'HOT',
        'HOW', 'HUG', 'HUGE', 'HUM', 'HUT', 'I', 'ICE', 'ICY', 'IF', 'ILL', 'IMP', 'IN', 'INK', 'INN', 'INTO', 'ION',
        'IRON', 'IS', 'IT', 'ITS', 'IVY', 'JAB', 'JAG', 'JAM', 'JAR', 'JAW', 'JAY', 'JET', 'JIG', 'JOB', 'JOG', 'JOT',
        'JOY', 'JUG', 'JUMP', 'JUST', 'KEEP', 'KEY', 'KID', 'KILL', 'KIND', 'KING', 'KIT', 'KITE', 'LAB', 'LACE', 'LAD',
        'LAKE', 'LAMB', 'LAMP', 'LAP', 'LAST', 'LATE', 'LAW', 'LAY', 'LAZY', 'LEAD', 'LEAF', 'LED', 'LEFT', 'LEG', 'LEND',
        'LET', 'LID', 'LIE', 'LIFE', 'LIKE', 'LIME', 'LINE', 'LION', 'LIP', 'LIST', 'LIT', 'LIVE', 'LOAD', 'LOAF', 'LOG',
        'LONE', 'LONG', 'LOOK', 'LOOM', 'LOOP', 'LOSE', 'LOST', 'LOT', 'LOUD', 'LOVE', 'LOW', 'LUCK', 'LUG', 'LUMP', 'MAD',
        'MADE', 'MAIL', 'MAIN', 'MAKE', 'MAN', 'MANY', 'MAP', 'MARE', 'MARK', 'MAT', 'MATE', 'ME', 'MEAN', 'MEAT', 'MEET',
        'MEN', 'MESS', 'MET', 'MID', 'MILD', 'MILE', 'MILK', 'MIND', 'MINE', 'MINT', 'MISS', 'MIST', 'MIX', 'MOM', 'MOOD',
        'MOON', 'MOP', 'MORE', 'MOST', 'MOTH', 'MOVE', 'MOW', 'MUD', 'MUG', 'MY', 'MYTH', 'NAB', 'NAG', 'NAIL', 'NAME',
        'NAP', 'NAVY', 'NEAR', 'NEAT', 'NECK', 'NEED', 'NEST', 'NET', 'NEW', 'NIB', 'NIL', 'NINE', 'NO', 'NOD', 'NONE',
        'NOON', 'NOR', 'NOSE', 'NOT', 'NOTE', 'NOW', 'NUN', 'NUT', 'OAF', 'OAK', 'OAR', 'OAT', 'ODD', 'OF', 'OFF', 'OFTEN',
        'OH', 'OIL', 'OK', 'OLD', 'ON', 'ONCE', 'ONE', 'ONLY', 'ONTO', 'OR', 'ORB', 'OUR', 'OUT', 'OVAL', 'OWE', 'OWL',
        'OWN', 'PACE', 'PACK', 'PAD', 'PAGE', 'PAIN', 'PAIR', 'PAL', 'PAN', 'PANT', 'PAP', 'PAR', 'PART', 'PASS', 'PAST',
        'PAT', 'PATH', 'PAW', 'PAY', 'PEA', 'PEACE', 'PEG', 'PEN', 'PENT', 'PER', 'PET', 'PHOTO', 'PIE', 'PIG', 'PIN',
        'PIPE', 'PIT', 'PLACE', 'PLAN', 'PLAY', 'PLOD', 'PLOT', 'PLOW', 'PLUG', 'PLUS', 'POD', 'POEM', 'POINT', 'POKE',
        'POLE', 'POND', 'POOL', 'POOR', 'POP', 'POST', 'POT', 'POUR', 'PUN', 'PUP', 'PUSH', 'PUT', 'QUIT', 'QUIZ', 'RACE',
        'RACK', 'RAD', 'RAFT', 'RAG', 'RAIN', 'RAKE', 'RAM', 'RAN', 'RANK', 'RAP', 'RAT', 'RATE', 'RAW', 'RAY', 'READ',
        'REAL', 'RED', 'REST', 'RIB', 'RICE', 'RICH', 'RID', 'RIDE', 'RIG', 'RIM', 'RING', 'RIP', 'RISE', 'RISK', 'ROAD',
        'ROB', 'ROCK', 'ROD', 'RIDE', 'ROLL', 'ROOF', 'ROOM', 'ROOT', 'ROPE', 'ROT', 'ROW', 'RUB', 'RUDE', 'RUG', 'RUIN',
        'RULE', 'RUN', 'RUSH', 'RUST', 'RUT', 'SACK', 'SAD', 'SAFE', 'SAGE', 'SAID', 'SAIL', 'SALE', 'SALT', 'SAME',
        'SAND', 'SANG', 'SAT', 'SAW', 'SAY', 'SEA', 'SEAL', 'SEAM', 'SEE', 'SEED', 'SEEK', 'SEEM', 'SELL', 'SEND', 'SENT',
        'SET', 'SEVEN', 'SEW', 'SHADE', 'SHED', 'SHE', 'SHIP', 'SHOE', 'SHOP', 'SHOT', 'SHOW', 'SHY', 'SICK', 'SIDE',
        'SIGN', 'SILLY', 'SIN', 'SING', 'SINK', 'SIP', 'SIR', 'SIT', 'SIX', 'SIZE', 'SKIN', 'SKY', 'SLAB', 'SLAM', 'SLAP',
        'SLID', 'SLIM', 'SLIP', 'SLOB', 'SLOG', 'SLOT', 'SLOW', 'SLUG', 'SLUM', 'SO', 'SOAK', 'SOAP', 'SOB', 'SOCK',
        'SOD', 'SOFA', 'SOFT', 'SOIL', 'SOLD', 'SOME', 'SON', 'SONG', 'SOON', 'SOUL', 'SOUP', 'SOW', 'SOY', 'SPIN',
        'SPOT', 'STAY', 'STEM', 'STEP', 'STIR', 'STOP', 'SUB', 'SUCH', 'SUD', 'SUE', 'SUM', 'SUN', 'SURE', 'TAB', 'TACK',
        'TAD', 'TAG', 'TAIL', 'TAKE', 'TALE', 'TALK', 'TALL', 'TAM', 'TAN', 'TANK', 'TAP', 'TAR', 'TASK', 'TAX', 'TEA',
        'TEAM', 'TEAR', 'TED', 'TEE', 'TELL', 'TEN', 'TEND', 'TENT', 'TEST', 'THAN', 'THAT', 'THE', 'THEM', 'THEN',
        'THEY', 'THIN', 'THIS', 'THUD', 'TICK', 'TIDE', 'TIDY', 'TIE', 'TILL', 'TIME', 'TIN', 'TINY', 'TIP', 'TIRE',
        'TO', 'TOAD', 'TOE', 'TOG', 'TOLD', 'TOM', 'TON', 'TOO', 'TOOK', 'TOOL', 'TOP', 'TORN', 'TOT', 'TOUR', 'TOW',
        'TOWN', 'TOY', 'TRAP', 'TRAY', 'TREE', 'TRIM', 'TRIP', 'TROT', 'TRY', 'TUB', 'TUCK', 'TUG', 'TUNE', 'TURN',
        'TWO', 'UGLY', 'UM', 'UP', 'UPON', 'US', 'USE', 'VAN', 'VASE', 'VAST', 'VAT', 'VET', 'VEX', 'VIA', 'VOW', 'WAD',
        'WAG', 'WAR', 'WAS', 'WASH', 'WAX', 'WAY', 'WE', 'WEAK', 'WEB', 'WED', 'WEE', 'WEEK', 'WELL', 'WENT', 'WERE',
        'WET', 'WHAT', 'WHEN', 'WHIM', 'WHO', 'WHY', 'WIDE', 'WIG', 'WIN', 'WIND', 'WING', 'WINK', 'WIRE', 'WISE', 'WISH',
        'WITH', 'WOKE', 'WOLF', 'WOOD', 'WOOL', 'WORD', 'WORK', 'WORM', 'YAK', 'YAM', 'YAP', 'YARD', 'YARN', 'YAWN',
        'YEA', 'YEAR', 'YELL', 'YES', 'YET', 'YOU', 'YOUR', 'ZAP', 'ZEN', 'ZIG', 'ZINC', 'ZIP', 'ZOO', 'ZOOM'
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_l = pygame.font.Font(None, 48)
        self.font_m = pygame.font.Font(None, 32)
        self.font_s = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (60, 65, 75)
        self.COLOR_LETTER = (220, 220, 220)
        self.COLOR_UI_TEXT = (200, 200, 200)
        self.COLOR_CURSOR = (70, 130, 250)
        self.COLOR_CURSOR_GRID = (250, 170, 80)
        self.COLOR_SELECTED = (60, 200, 255)
        self.COLOR_INVALID = (220, 50, 50)
        self.COLOR_VALID = (50, 220, 120)

        # Game constants
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_X = (self.width - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_Y = 50
        self.HAND_SIZE = 7
        self.MAX_MOVES = 10
        self.WORDS_TO_WIN = 5

        self.grid = None
        self.letter_hand = None
        self.grid_cursor = None
        self.hand_cursor = None
        self.selected_letter_info = None
        self.focus_on_grid = None
        self.moves_left = None
        self.words_found_count = None
        self.score = None
        self.game_over = None
        self.found_words_set = None
        self.feedback_flash = None
        self.newly_formed_word_info = None
        self.rng = None
        
        self.reset()
        # self.validate_implementation() # Uncomment to run validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), "", dtype='<U1')
        self.letter_hand = self._generate_hand()
        self.grid_cursor = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.hand_cursor = self.HAND_SIZE // 2
        self.selected_letter_info = None
        self.focus_on_grid = False
        self.moves_left = self.MAX_MOVES
        self.words_found_count = 0
        self.score = 0
        self.game_over = False
        self.found_words_set = set()
        self.feedback_flash = None
        self.newly_formed_word_info = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.feedback_flash = None
        self.newly_formed_word_info = []
        reward = 0
        
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.focus_on_grid = shift_held

        # --- Handle Movement ---
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1
            
            if self.focus_on_grid:
                self.grid_cursor[0] = np.clip(self.grid_cursor[0] + dx, 0, self.GRID_SIZE - 1)
                self.grid_cursor[1] = np.clip(self.grid_cursor[1] + dy, 0, self.GRID_SIZE - 1)
            else:
                self.hand_cursor = np.clip(self.hand_cursor + dx, 0, self.HAND_SIZE - 1)

        # --- Handle Action (Space Press) ---
        if space_pressed:
            if not self.focus_on_grid: # Selecting a letter from hand
                self.selected_letter_info = {
                    "letter": self.letter_hand[self.hand_cursor],
                    "hand_index": self.hand_cursor
                }
            elif self.selected_letter_info: # Placing a letter on grid
                gx, gy = self.grid_cursor
                if self.grid[gy, gx] == "":
                    # Valid placement
                    letter_to_place = self.selected_letter_info["letter"]
                    self.grid[gy, gx] = letter_to_place
                    reward += 1 # Base reward for placing a letter
                    
                    newly_found_words = self._check_words(gx, gy)
                    if newly_found_words:
                        # sound: word_found.wav
                        for word in newly_found_words:
                            reward += 5 * len(word)
                        self.words_found_count += len(newly_found_words)

                    self._replenish_letter(self.selected_letter_info["hand_index"])
                    self.selected_letter_info = None
                    self.moves_left -= 1
                    # sound: place_letter.wav
                else: # Cell occupied
                    self.feedback_flash = {"color": self.COLOR_INVALID, "pos": self.grid_cursor, "type": "grid"}
                    # sound: invalid_move.wav
            else: # Trying to place with no letter selected
                self.feedback_flash = {"color": self.COLOR_INVALID, "pos": self.hand_cursor, "type": "hand"}
                # sound: invalid_move.wav

        # --- Check Termination ---
        terminated = (self.moves_left <= 0) or (self.words_found_count >= self.WORDS_TO_WIN)
        if terminated and not self.game_over:
            if self.words_found_count >= self.WORDS_TO_WIN:
                reward += 50 # Win bonus
                # sound: game_win.wav
            else:
                # sound: game_lose.wav
                pass
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_hand(self):
        vowels = "AEIOU"
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        hand = []
        for _ in range(3): hand.append(self.rng.choice(list(vowels)))
        for _ in range(4): hand.append(self.rng.choice(list(consonants)))
        self.rng.shuffle(hand)
        return hand

    def _replenish_letter(self, index):
        vowels = "AEIOU"
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        # Weighted replenish: ~40% vowel
        if self.rng.random() < 0.4:
            self.letter_hand[index] = self.rng.choice(list(vowels))
        else:
            self.letter_hand[index] = self.rng.choice(list(consonants))

    def _check_words(self, x, y):
        new_words = []
        
        # Horizontal check
        start_x = x
        while start_x > 0 and self.grid[y, start_x - 1] != "":
            start_x -= 1
        end_x = x
        while end_x < self.GRID_SIZE - 1 and self.grid[y, end_x + 1] != "":
            end_x += 1
        
        if end_x > start_x:
            word_h = "".join(self.grid[y, start_x:end_x + 1])
            if word_h in self.WORD_LIST and word_h not in self.found_words_set:
                new_words.append(word_h)
                self.found_words_set.add(word_h)
                self.newly_formed_word_info.append({"start": (start_x, y), "end": (end_x, y)})

        # Vertical check
        start_y = y
        while start_y > 0 and self.grid[start_y - 1, x] != "":
            start_y -= 1
        end_y = y
        while end_y < self.GRID_SIZE - 1 and self.grid[end_y + 1, x] != "":
            end_y += 1

        if end_y > start_y:
            word_v = "".join(self.grid[start_y:end_y + 1, x])
            if word_v in self.WORD_LIST and word_v not in self.found_words_set:
                new_words.append(word_v)
                self.found_words_set.add(word_v)
                self.newly_formed_word_info.append({"start": (x, start_y), "end": (x, end_y)})
        
        return new_words

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_SIZE * self.CELL_SIZE))
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_Y + i * self.CELL_SIZE))

        # Draw letters on grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter = self.grid[r, c]
                if letter != "":
                    text_surf = self.font_l.render(letter, True, self.COLOR_LETTER)
                    text_rect = text_surf.get_rect(center=(self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2))
                    self.screen.blit(text_surf, text_rect)
        
        # Draw newly formed word highlights
        for info in self.newly_formed_word_info:
            start_pos = (self.GRID_X + info["start"][0] * self.CELL_SIZE, self.GRID_Y + info["start"][1] * self.CELL_SIZE)
            end_pos = (self.GRID_X + info["end"][0] * self.CELL_SIZE, self.GRID_Y + info["end"][1] * self.CELL_SIZE)
            rect = pygame.Rect(start_pos[0], start_pos[1], end_pos[0] - start_pos[0] + self.CELL_SIZE, end_pos[1] - start_pos[1] + self.CELL_SIZE)
            
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_VALID, 70))
            self.screen.blit(s, rect.topleft)

        # Draw cursors
        cursor_color = self.COLOR_CURSOR_GRID if self.focus_on_grid else self.COLOR_CURSOR
        
        # Grid cursor
        gx, gy = self.grid_cursor
        grid_cursor_rect = pygame.Rect(self.GRID_X + gx * self.CELL_SIZE, self.GRID_Y + gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, cursor_color, grid_cursor_rect, 3 if self.focus_on_grid else 1, border_radius=4)
        
        # Hand cursor and letters
        hand_cell_width = 40
        hand_total_width = self.HAND_SIZE * hand_cell_width
        hand_start_x = (self.width - hand_total_width) // 2
        hand_y = self.height - 40

        for i in range(self.HAND_SIZE):
            is_selected_letter = self.selected_letter_info and self.selected_letter_info["hand_index"] == i
            letter_color = self.COLOR_SELECTED if is_selected_letter else self.COLOR_LETTER
            
            letter_surf = self.font_m.render(self.letter_hand[i], True, letter_color)
            center_x = hand_start_x + i * hand_cell_width + hand_cell_width // 2
            letter_rect = letter_surf.get_rect(center=(center_x, hand_y))
            self.screen.blit(letter_surf, letter_rect)

            if i == self.hand_cursor:
                hand_cursor_rect = pygame.Rect(hand_start_x + i * hand_cell_width, hand_y - 15, hand_cell_width, 30)
                pygame.draw.rect(self.screen, cursor_color, hand_cursor_rect, 3 if not self.focus_on_grid else 1, border_radius=4)
        
        # Draw feedback flash
        if self.feedback_flash:
            color = (*self.feedback_flash["color"], 128)
            if self.feedback_flash["type"] == "grid":
                pos = self.feedback_flash["pos"]
                rect = pygame.Rect(self.GRID_X + pos[0] * self.CELL_SIZE, self.GRID_Y + pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            else: # hand
                pos = self.feedback_flash["pos"]
                rect = pygame.Rect(hand_start_x + pos * hand_cell_width, hand_y - 15, hand_cell_width, 30)
            
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_m.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_m.render(moves_text, True, self.COLOR_UI_TEXT)
        moves_rect = moves_surf.get_rect(centerx=self.width // 2, y=10)
        self.screen.blit(moves_surf, moves_rect)

        # Words Found
        words_text = f"WORDS: {self.words_found_count}/{self.WORDS_TO_WIN}"
        words_surf = self.font_m.render(words_text, True, self.COLOR_UI_TEXT)
        words_rect = words_surf.get_rect(right=self.width - 20, y=10)
        self.screen.blit(words_surf, words_rect)
        
        if self.game_over:
            result_text = "YOU WIN!" if self.words_found_count >= self.WORDS_TO_WIN else "GAME OVER"
            result_color = self.COLOR_VALID if self.words_found_count >= self.WORDS_TO_WIN else self.COLOR_INVALID
            
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_surf = self.font_l.render(result_text, True, result_color)
            result_rect = result_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(result_surf, result_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "words_found": self.words_found_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        assert info['score'] == 0
        assert self.moves_left == self.MAX_MOVES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # Test a specific action sequence
        self.reset()
        # 1. Select a letter (no shift, space press)
        self.step([self.hand_cursor, 1, 0])
        assert self.selected_letter_info is not None, "Failed to select letter"
        # 2. Move on grid (shift, move right)
        self.step([4, 0, 1])
        assert self.grid_cursor[0] == (self.GRID_SIZE // 2) + 1, "Failed to move on grid"
        # 3. Place the letter (shift, space press)
        obs, reward, term, trunc, info = self.step([0, 1, 1])
        assert self.moves_left == self.MAX_MOVES - 1, "Failed to decrement moves"
        assert self.selected_letter_info is None, "Failed to clear selected letter"
        assert reward >= 1, "Placement reward not given"

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Word Grid Game ---")
    print(env.game_description)
    print(env.user_guide)

    # Use a dummy screen for display
    pygame.display.set_caption("Word Grid")
    display_screen = pygame.display.set_mode((env.width, env.height))

    action = [0, 0, 0] # No-op, no space, no shift
    
    while not done:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # --- Step Environment ---
        # Only step if an action is taken, to simulate turn-based play for humans
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward}, Info: {info}")

        # --- Render ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play

    env.close()