
# Generated: 2025-08-27T13:13:51.334614
# Source Brief: brief_00282.md
# Brief Index: 282

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # A curated list of common English words (3-7 letters) for the game.
    # Sourced from common word lists and filtered for length.
    WORD_LIST = {
        'ART', 'AND', 'ARE', 'ARM', 'ACT', 'AID', 'ACE', 'ADD', 'AGE', 'AGO', 'AIR', 'ALL', 'ANY', 'APE', 'APT', 'ARC',
        'ASK', 'ATE', 'AWL', 'AXE', 'BAD', 'BAG', 'BAN', 'BAR', 'BAT', 'BED', 'BEE', 'BEG', 'BET', 'BID', 'BIG', 'BIN',
        'BIT', 'BOB', 'BOG', 'BOX', 'BOY', 'BUG', 'BUM', 'BUN', 'BUS', 'BUT', 'BUY', 'BYE', 'CAB', 'CAD', 'CAM', 'CAN',
        'CAP', 'CAR', 'CAT', 'COB', 'COD', 'COG', 'CON', 'COP', 'COT', 'COW', 'CRY', 'CUB', 'CUD', 'CUE', 'CUP', 'CUR',
        'CUT', 'DAB', 'DAD', 'DAM', 'DAN', 'DAY', 'DEN', 'DEW', 'DID', 'DIE', 'DIG', 'DIM', 'DIN', 'DIP', 'DOG', 'DON',
        'DOT', 'DRY', 'DUB', 'DUD', 'DUE', 'DUG', 'DYE', 'EAR', 'EAT', 'EEL', 'EGG', 'EGO', 'ELK', 'END', 'EON', 'ERA',
        'EWE', 'EYE', 'FAD', 'FAN', 'FAR', 'FAT', 'FED', 'FEE', 'FEW', 'FIB', 'FIG', 'FIN', 'FIR', 'FIT', 'FIX', 'FLY',
        'FOG', 'FOR', 'FOX', 'FRY', 'FUM', 'FUN', 'FUR', 'GAG', 'GAL', 'GAP', 'GAS', 'GAY', 'GEL', 'GEM', 'GET', 'GIG',
        'GIN', 'GIP', 'GOD', 'GOT', 'GUM', 'GUN', 'GUT', 'GUY', 'GYM', 'HAD', 'HAG', 'HAM', 'HAS', 'HAT', 'HAY', 'HEM',
        'HEN', 'HER', 'HEW', 'HEX', 'HEY', 'HID', 'HIM', 'HIP', 'HIS', 'HIT', 'HOG', 'HOP', 'HOT', 'HOW', 'HUB', 'HUE',
        'HUG', 'HUM', 'HUT', 'ICE', 'ICY', 'ILL', 'IMP', 'INK', 'INN', 'ION', 'IRE', 'IRK', 'IVY', 'JAB', 'JAG', 'JAM',
        'JAR', 'JAW', 'JAY', 'JET', 'JIB', 'JIG', 'JOB', 'JOG', 'JOT', 'JOY', 'JUG', 'JUT', 'KEG', 'KEN', 'KEY', 'KID',
        'KIN', 'KIP', 'KIT', 'LAB', 'LAD', 'LAG', 'LAM', 'LAP', 'LAW', 'LAX', 'LAY', 'LEA', 'LED', 'LEE', 'LEG', 'LET',
        'LID', 'LIE', 'LIP', 'LIT', 'LOB', 'LOG', 'LOP', 'LOT', 'LOW', 'LUG', 'LYE', 'MAD', 'MAN', 'MAP', 'MAR', 'MAT',
        'MAW', 'MAY', 'MEN', 'MET', 'MEW', 'MID', 'MIG', 'MIM', 'MOB', 'MOD', 'MOL', 'MOM', 'MOP', 'MOW', 'MUD', 'MUG',
        'MUM', 'NAB', 'NAG', 'NAP', 'NAY', 'NET', 'NEW', 'NIB', 'NIL', 'NIP', 'NIT', 'NOB', 'NOD', 'NOR', 'NOT', 'NOW',
        'NUN', 'NUT', 'OAF', 'OAK', 'OAR', 'OAT', 'ODD', 'ODE', 'OFF', 'OFT', 'OHO', 'OIL', 'OLD', 'ONE', 'OPT', 'ORB',
        'ORE', 'OUR', 'OUT', 'OWE', 'OWL', 'OWN', 'PAD', 'PAL', 'PAM', 'PAN', 'PAP', 'PAR', 'PAT', 'PAW', 'PAY', 'PEA',
        'PEG', 'PEN', 'PEP', 'PER', 'PET', 'PEW', 'PIE', 'PIG', 'PIN', 'PIP', 'PIT', 'PLY', 'POD', 'POI', 'POP', 'POT',
        'PRO', 'PRY', 'PUB', 'PUD', 'PUG', 'PUN', 'PUP', 'PUT', 'QUO', 'RAD', 'RAG', 'RAM', 'RAN', 'RAP', 'RAT', 'RAW',
        'RAY', 'RED', 'REF', 'REG', 'REP', 'REV', 'RIB', 'RID', 'RIG', 'RIM', 'RIN', 'RIP', 'ROB', 'ROD', 'ROE', 'ROT',
        'ROW', 'RUB', 'RUE', 'RUG', 'RUM', 'RUN', 'RUT', 'RYE', 'SAD', 'SAG', 'SAP', 'SAT', 'SAW', 'SAY', 'SEA', 'SEC',
        'SEE', 'SET', 'SEW', 'SEX', 'SHY', 'SIB', 'SIC', 'SIM', 'SIN', 'SIP', 'SIR', 'SIS', 'SIT', 'SIX', 'SKI', 'SKY',
        'SLY', 'SOB', 'SOD', 'SON', 'SOP', 'SOW', 'SOY', 'SPY', 'STY', 'SUB', 'SUE', 'SUM', 'SUN', 'SUP', 'TAB', 'TAD',
        'TAG', 'TAN', 'TAP', 'TAR', 'TEA', 'TED', 'TEE', 'TEN', 'THE', 'THO', 'THY', 'TIC', 'TIE', 'TIL', 'TIN', 'TIP',
        'TOE', 'TOG', 'TOM', 'TON', 'TOO', 'TOP', 'TOW', 'TOY', 'TRY', 'TUB', 'TUG', 'TUI', 'TUM', 'TUN', 'TUT', 'TWO',
        'USE', 'VAN', 'VAT', 'VET', 'VIE', 'VOW', 'WAB', 'WAD', 'WAG', 'WAN', 'WAR', 'WAS', 'WAX', 'WAY', 'WEB', 'WED',
        'WEE', 'WET', 'WHO', 'WHY', 'WIG', 'WIN', 'WIT', 'WOE', 'WOK', 'WON', 'WOT', 'WOW', 'WRY', 'YAK', 'YAM', 'YAP',
        'YAR', 'YAW', 'YEA', 'YEN', 'YEP', 'YES', 'YET', 'YIN', 'YIP', 'YOU', 'YUK', 'ZAP', 'ZED', 'ZEE', 'ZEN', 'ZIG',
        'ZIP', 'ZOO', 'ABOUT', 'ABOVE', 'ACTOR', 'ACUTE', 'ADAPT', 'ADMIT', 'ADOPT', 'ADORE', 'ADULT', 'AFTER', 'AGAIN',
        'AGENT', 'AGILE', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIEN', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW',
        'ALONE', 'ALONG', 'ALTER', 'AMONG', 'ANGER', 'ANGLE', 'ANKLE', 'APPLE', 'APPLY', 'APRON', 'ARENA', 'ARGUE',
        'ARISE', 'ARMOR', 'ARRAY', 'ARROW', 'ASIDE', 'ASSET', 'AUDIO', 'AUDIT', 'AVOID', 'AWARD', 'AWARE', 'AWFUL',
        'BADGE', 'BASIC', 'BEACH', 'BEARD', 'BEAST', 'BEGIN', 'BEING', 'BELLY', 'BELOW', 'BENCH', 'BIBLE', 'BLADE',
        'BLAME', 'BLANK', 'BLAST', 'BLEED', 'BLEND', 'BLESS', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOAST', 'BOOST',
        'BRAIN', 'BRAND', 'BRAVE', 'BREAD', 'BREAK', 'BREED', 'BRICK', 'BRIDE', 'BRIEF', 'BRING', 'BROAD', 'BROWN',
        'BRUSH', 'BUILD', 'BUNCH', 'BURST', 'BUYER', 'CABIN', 'CABLE', 'CAMEL', 'CANAL', 'CANDY', 'PANIC', 'PAPER',
        'PARTY', 'PASTE', 'PAUSE', 'PEACE', 'PHASE', 'PHONE', 'PHOTO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN',
        'PLANE', 'PLANT', 'PLATE', 'POINT', 'POKER', 'POLAR', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME',
        'PRINT', 'PRIOR', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN', 'QUEST', 'QUICK', 'QUIET', 'QUITE', 'QUOTE',
        'RACE', 'RADIO', 'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'REACT', 'READ', 'READY', 'REAL', 'REALM',
        'REASON', 'REBEL', 'RECALL', 'RECIPE', 'REFER', 'RELAX', 'RENT', 'REPLY', 'RHYTHM', 'RICE', 'RICH', 'RIDE',
        'RIFLE', 'RIGHT', 'RING', 'RISE', 'RISK', 'RIVAL', 'RIVER', 'ROAD', 'ROBOT', 'ROCKET', 'ROLE', 'ROLL', 'ROOF',
        'ROOM', 'ROOT', 'ROPE', 'ROSE', 'ROUND', 'ROUTE', 'ROYAL', 'RULER', 'RUMOR', 'RURAL', 'SADLY', 'SAFE',
        'SAIL', 'SAKE', 'SALAD', 'SALE', 'SALT', 'SAME', 'SAMPLE', 'SAND', 'SAVE', 'SCALE', 'SCARE', 'SCENE',
        'SCOPE', 'SCORE', 'SEAT', 'SECTOR', 'SEEK', 'SEEM', 'SELL', 'SEND', 'SENSE', 'SERVE', 'SEVEN', 'SHADE',
        'SHADOW', 'SHAKE', 'SHAPE', 'SHARE', 'SHARK', 'SHARP', 'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIP',
        'SHIRT', 'SHOCK', 'SHOE', 'SHOOT', 'SHOP', 'SHORE', 'SHORT', 'SHOT', 'SHOW', 'SHOWER', 'SHRUG', 'SICK',
        'SIDE', 'SIGH', 'SIGHT', 'SIGN', 'SIGNAL', 'SILENT', 'SILK', 'SILLY', 'SILVER', 'SIMPLE', 'SIMPLY', 'SINCE',
        'SING', 'SINGLE', 'SISTER', 'SITE', 'SIZE', 'SKILL', 'SKIN', 'SKIRT', 'SKY', 'SLAP', 'SLAVE', 'SLEEP',
        'SLICE', 'SLIDE', 'SLIGHT', 'SLIP', 'SLOW', 'SMALL', 'SMART', 'SMELL', 'SMILE', 'SMOKE', 'SMOOTH', 'SNACK',
        'SNAKE', 'SNOW', 'SOCCER', 'SOCIAL', 'SOFT', 'SOIL', 'SOLAR', 'SOLID', 'SOLVE', 'SOME', 'SONG', 'SOON',
        'SORRY', 'SORT', 'SOUL', 'SOUND', 'SOUP', 'SOURCE', 'SOUTH', 'SPACE', 'SPARE', 'SPARK', 'SPEAK', 'SPEED',
        'SPELL', 'SPEND', 'SPICE', 'SPIDER', 'SPILL', 'SPIN', 'SPIRIT', 'SPLIT', 'SPOON', 'SPORT', 'SPOT', 'SPRAY',
        'SPREAD', 'SPRING', 'SQUARE', 'STABLE', 'STAFF', 'STAGE', 'STAIR', 'STAKE', 'STAND', 'STAR', 'STARE',
        'START', 'STATE', 'STAY', 'STEAK', 'STEAL', 'STEAM', 'STEEL', 'STEP', 'STICK', 'STIFF', 'STILL', 'STING',
        'STOCK', 'STONE', 'STORE', 'STORM', 'STORY', 'STOVE', 'STRAIN', 'STREAM', 'STREET', 'STRESS', 'STRIKE',
        'STRING', 'STRIP', 'STROKE', 'STRONG', 'STUDIO', 'STUDY', 'STUFF', 'STYLE', 'SUCH', 'SUGAR', 'SUIT',

    }
    # Letter frequencies for a more natural distribution in the letter bag.
    LETTER_FREQUENCIES = (
        'E' * 12 + 'A' * 9 + 'I' * 9 + 'O' * 8 + 'N' * 6 + 'R' * 6 + 'T' * 6 + 'L' * 4 + 'S' * 4 + 'U' * 4 +
        'D' * 4 + 'G' * 3 + 'B' * 2 + 'C' * 2 + 'M' * 2 + 'P' * 2 + 'F' * 2 + 'H' * 2 + 'V' * 2 + 'W' * 2 +
        'Y' * 2 + 'K' * 1 + 'J' * 1 + 'X' * 1 + 'Q' * 1 + 'Z' * 1
    )

    user_guide = "Controls: Use arrow keys to move the cursor, Shift to cycle through your letters, and Space to place a letter."
    game_description = "Strategically place letters on the grid to form words. Form 5 valid words to win the game."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.GRID_SIZE = 10
        self.WIN_CONDITION = 5
        self.MAX_STEPS = 1000
        self.HAND_SIZE = 5
        self.BAG_SIZE = 50

        # Visuals
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 65)
        self.COLOR_LETTER = (220, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0, 100) # RGBA for transparency
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_HAND_BG = (40, 40, 55)
        self.COLOR_HAND_LETTER = (150, 180, 255)
        self.COLOR_HAND_SELECTED = (255, 200, 0)
        self.COLOR_VALID_WORD = (50, 220, 120)
        self.COLOR_INVALID_MOVE = (220, 50, 50)

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_m = pygame.font.SysFont("Consolas", 24)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_s = pygame.font.SysFont("Consolas", 18)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.letter_bag = None
        self.letter_hand = None
        self.cursor_pos = None
        self.selected_letter_idx = None
        self.words_found_count = None
        self.found_words = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.last_shift_held = None
        self.flashers = None
        
        self.reset()
        
        # Self-check to ensure implementation correctness
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), ' ', dtype=str)
        
        # Create a shuffled bag of letters
        self.letter_bag = random.sample(self.LETTER_FREQUENCIES, len(self.LETTER_FREQUENCIES))[:self.BAG_SIZE]
        
        # Draw initial hand
        self.letter_hand = []
        for _ in range(self.HAND_SIZE):
            if self.letter_bag:
                self.letter_hand.append(self.letter_bag.pop())

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_letter_idx = 0
        
        self.words_found_count = 0
        self.found_words = set()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.flashers = [] # For visual feedback

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_action, shift_action = action
        space_held = space_action == 1
        shift_held = shift_action == 1
        
        reward = 0
        self.flashers.clear() # Clear last step's visual feedback

        # --- Action Handling ---
        # Movement
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # Cycle selected letter (on key press, not hold)
        if shift_held and not self.last_shift_held and self.letter_hand:
            self.selected_letter_idx = (self.selected_letter_idx + 1) % len(self.letter_hand)
            # sfx: UI_CYCLE_SOUND

        # Place letter (on key press, not hold)
        if space_held and not self.last_space_held and self.letter_hand:
            reward += self._place_letter()
            # sfx: PLACE_LETTER_SOUND
        
        # Update last action states for next step's press detection
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.words_found_count >= self.WIN_CONDITION:
                reward += 50 # Win bonus
                # sfx: WIN_SOUND
            else:
                reward += -50 # Loss penalty
                # sfx: LOSE_SOUND
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_letter(self):
        y, x = self.cursor_pos
        
        # Check if cell is occupied
        if self.grid[y, x] != ' ':
            self.flashers.append({'pos': (x, y), 'color': self.COLOR_INVALID_MOVE, 'type': 'cell'})
            return -0.5 # Penalty for invalid move

        # Get selected letter and place it
        letter_to_place = self.letter_hand[self.selected_letter_idx]
        self.grid[y, x] = letter_to_place
        
        # Replenish hand
        if self.letter_bag:
            self.letter_hand[self.selected_letter_idx] = self.letter_bag.pop()
        else:
            self.letter_hand.pop(self.selected_letter_idx)
        
        # Adjust selected index if necessary
        if self.letter_hand:
            self.selected_letter_idx = min(self.selected_letter_idx, len(self.letter_hand) - 1)

        # Check for new words and calculate reward
        newly_found_words, word_coords = self._check_words(x, y)
        placement_reward = 0
        if not newly_found_words:
            placement_reward = -0.1 # Penalty for non-productive placement
        else:
            for word in newly_found_words:
                if len(word) >= 4:
                    placement_reward += 5
                else: # len is 3
                    placement_reward += 1
                self.found_words.add(word)
                self.words_found_count += 1
                # sfx: WORD_FOUND_SOUND
            
            # Add flashers for all letters in the new words
            for wx, wy in word_coords:
                self.flashers.append({'pos': (wx, wy), 'color': self.COLOR_VALID_WORD, 'type': 'cell'})

        self.score += placement_reward
        return placement_reward

    def _check_words(self, px, py):
        new_words = []
        all_coords = set()

        # Horizontal check
        start_x = px
        while start_x > 0 and self.grid[py, start_x - 1] != ' ':
            start_x -= 1
        end_x = px
        while end_x < self.GRID_SIZE - 1 and self.grid[py, end_x + 1] != ' ':
            end_x += 1
        
        word_h = "".join(self.grid[py, start_x:end_x+1])
        if len(word_h) >= 3 and word_h in self.WORD_LIST and word_h not in self.found_words:
            new_words.append(word_h)
            for i in range(start_x, end_x + 1):
                all_coords.add((i, py))

        # Vertical check
        start_y = py
        while start_y > 0 and self.grid[start_y - 1, px] != ' ':
            start_y -= 1
        end_y = py
        while end_y < self.GRID_SIZE - 1 and self.grid[end_y + 1, px] != ' ':
            end_y += 1

        word_v = "".join(self.grid[start_y:end_y+1, px])
        if len(word_v) >= 3 and word_v in self.WORD_LIST and word_v not in self.found_words:
            new_words.append(word_v)
            for i in range(start_y, end_y + 1):
                all_coords.add((px, i))
                
        return new_words, all_coords

    def _check_termination(self):
        if self.words_found_count >= self.WIN_CONDITION:
            return True
        if not self.letter_hand and not self.letter_bag:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Centered grid
        grid_pixel_size = 360
        cell_size = grid_pixel_size / self.GRID_SIZE
        offset_x = (self.SCREEN_WIDTH - grid_pixel_size) / 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_size) / 2

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x + i * cell_size, offset_y), (offset_x + i * cell_size, offset_y + grid_pixel_size), 1)
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, offset_y + i * cell_size), (offset_x + grid_pixel_size, offset_y + i * cell_size), 1)

        # Draw letters, cursor, and flashers
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                
                # Draw flashers underneath letters
                for flasher in self.flashers:
                    if flasher['type'] == 'cell' and flasher['pos'] == (c, r):
                        pygame.draw.rect(self.screen, flasher['color'], cell_rect)
                
                # Draw placed letters
                letter = self.grid[r, c]
                if letter != ' ':
                    text_surf = self.font_l.render(letter, True, self.COLOR_LETTER)
                    text_rect = text_surf.get_rect(center=cell_rect.center)
                    self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_y, cursor_x = self.cursor_pos
        cursor_rect = pygame.Rect(offset_x + cursor_x * cell_size, offset_y + cursor_y * cell_size, cell_size, cell_size)
        cursor_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Score and Words Found
        score_text = self.font_m.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        words_text = self.font_m.render(f"Words: {self.words_found_count} / {self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        self.screen.blit(words_text, (20, 50))
        
        # Remaining letters in bag
        bag_text = self.font_s.render(f"Bag: {len(self.letter_bag)}", True, self.COLOR_UI_TEXT)
        bag_rect = bag_text.get_rect(bottomright=(self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 80))
        self.screen.blit(bag_text, bag_rect)

        # Player's Hand
        if self.letter_hand:
            hand_width = len(self.letter_hand) * 45
            hand_x_start = self.SCREEN_WIDTH - hand_width - 20
            
            for i, letter in enumerate(self.letter_hand):
                is_selected = (i == self.selected_letter_idx)
                box_rect = pygame.Rect(hand_x_start + i * 45, self.SCREEN_HEIGHT - 60, 40, 50)
                
                pygame.draw.rect(self.screen, self.COLOR_HAND_BG, box_rect, border_radius=4)
                
                text_color = self.COLOR_HAND_SELECTED if is_selected else self.COLOR_HAND_LETTER
                text_surf = self.font_l.render(letter, True, text_color)
                text_rect = text_surf.get_rect(center=box_rect.center)
                self.screen.blit(text_surf, text_rect)

                if is_selected:
                    pygame.draw.rect(self.screen, self.COLOR_HAND_SELECTED, box_rect, 2, border_radius=4)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_status = "YOU WIN!" if self.words_found_count >= self.WIN_CONDITION else "GAME OVER"
            status_text = self.font_l.render(win_status, True, self.COLOR_HAND_SELECTED)
            status_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(status_text, status_rect)
            
            final_score_text = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": self.words_found_count,
            "letters_in_hand": len(self.letter_hand),
            "letters_in_bag": len(self.letter_bag),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action components
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a separate screen for rendering if playing manually
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Grid")
    
    clock = pygame.time.Clock()
    
    while not done:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Handle movement keys
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key found (e.g., up over down)

        # Handle space and shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        if keys[pygame.K_r]: # Allow manual reset
            obs, info = env.reset()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for playability

    env.close()