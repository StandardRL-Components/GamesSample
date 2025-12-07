import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to add a letter to your word. "
        "Press shift to submit the word."
    )

    game_description = (
        "Find words in a grid of letters by connecting adjacent tiles. Find 5 words before the timer runs out to win!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_ROWS, GRID_COLS = 6, 8
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_CELL_SIZE = 48
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * GRID_CELL_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_ROWS * GRID_CELL_SIZE) // 2 + 30
    
    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_TILE = (60, 75, 90)
    COLOR_TILE_TEXT = (210, 220, 230)
    COLOR_SELECTOR = (0, 150, 255)
    COLOR_PATH = (40, 200, 120)
    COLOR_PATH_TEXT = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (255, 180, 0)
    COLOR_TIMER_BAR_BG = (70, 70, 70)
    COLOR_SUCCESS = (100, 255, 150)
    COLOR_FAIL = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_tile = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_feedback = pygame.font.SysFont("Impact", 32)
        
        self.word_list = self._get_word_list()
        self.trie = self._build_trie(self.word_list)
        self.letter_distribution = 'AAAAAAAAABBCCDDDDEEEEEEEEEEEEFFGGGHHIIIIIIIIIJKLLLLMMNNNNNNOOOOOOOOPPQRRRRRRSSSSSTTTTTTUUUUVVWWXYYZ'

        self.grid = []
        self.solvable_words = set()
        self.selector_pos = [0, 0]
        self.current_word_path = []
        self.words_found = set()
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_messages = []
        
        # self.reset() is called by the wrapper/user, not needed here for gym compliance
        # self.validate_implementation() is for debugging, not needed in final version

    def _get_word_list(self):
        # Curated list to ensure self-contained, solvable grids
        return {
            'ART', 'AND', 'ARE', 'ARM', 'ATE', 'BAT', 'BET', 'BIT', 'BUT', 'CAR', 'CAT', 'COT', 'CUP', 'CUT',
            'DAY', 'DEW', 'DID', 'DOG', 'DRY', 'EAT', 'EGG', 'END', 'FAR', 'FAT', 'FEW', 'FIT', 'FOR', 'FUN',
            'GET', 'GOT', 'GYM', 'HAD', 'HAS', 'HAT', 'HEN', 'HIM', 'HIS', 'HOT', 'HOW', 'INK', 'JAM', 'JAR',
            'JET', 'JOB', 'JOG', 'JOY', 'KEY', 'LET', 'LID', 'LOG', 'LOT', 'MAN', 'MAP', 'MAT', 'MEN', 'MIX',
            'MOM', 'MUD', 'NET', 'NEW', 'NOT', 'NOW', 'NUT', 'OAK', 'ODD', 'OFF', 'OIL', 'OLD', 'ONE', 'OWN',
            'PAN', 'PAT', 'PET', 'PIG', 'PIN', 'POT', 'PUT', 'RAT', 'RED', 'RUN', 'SAD', 'SAT', 'SAW', 'SAY',
            'SEA', 'SET', 'SIT', 'SKY', 'SON', 'SUN', 'TAP', 'TEN', 'THE', 'TIN', 'TOP', 'TOY', 'TRY', 'TWO',
            'USE', 'VAN', 'WAR', 'WAS', 'WAY', 'WET', 'WHO', 'WHY', 'WIN', 'YES', 'YET', 'YOU', 'ZAP', 'ZIP',
            'ABLE', 'ACID', 'AGED', 'ALSO', 'AREA', 'ARMY', 'AWAY', 'BABY', 'BACK', 'BALL', 'BAND', 'BANK',
            'BASE', 'BATH', 'BEAM', 'BEAN', 'BEAR', 'BEAT', 'BELL', 'BELT', 'BEND', 'BEST', 'BIRD', 'BLOW',
            'BLUE', 'BOAT', 'BODY', 'BOMB', 'BOND', 'BONE', 'BOOK', 'BOOM', 'BORN', 'BOSS', 'BOTH', 'BOWL',
            'BURN', 'BUSH', 'BUSY', 'CAKE', 'CALL', 'CALM', 'CAMP', 'CARD', 'CARE', 'CASE', 'CASH', 'CAST',
            'CAVE', 'CELL', 'CHAT', 'CHIP', 'CITY', 'CLAY', 'CLIP', 'CLUB', 'COAL', 'COAT', 'CODE', 'COLD',
            'COME', 'COOK', 'COOL', 'COPY', 'CORE', 'CORN', 'COST', 'CREW', 'CROP', 'DARK', 'DATA', 'DATE',
            'DAWN', 'DAYS', 'DEAL', 'DEAR', 'DEBT', 'DECK', 'DEEP', 'DEER', 'DESK', 'DIET', 'DIRT', 'DISK',
            'DIVE', 'DOCK', 'DOES', 'DOLL', 'DONE', 'DOOR', 'DOWN', 'DRAW', 'DREAM', 'DRINK', 'DRIVE',
            'DROP', 'DUST', 'DUTY', 'EACH', 'EARN', 'EASY', 'EDGE', 'ELSE', 'EVEN', 'EVER', 'EVIL', 'EXIT',
            'FACE', 'FACT', 'FAIL', 'FAIR', 'FALL', 'FARM', 'FAST', 'FATE', 'FEAR', 'FEED', 'FEEL', 'FEET',
            'FILE', 'FILL', 'FILM', 'FIND', 'FINE', 'FIRE', 'FIRM', 'FISH', 'FIVE', 'FLAG', 'FLAT', 'FLOW',
            'FOOD', 'FOOT', 'FORM', 'FOUR', 'FREE', 'FROM', 'FUEL', 'FULL', 'GAIN', 'GAME', 'GATE', 'GAVE',
            'GEAR', 'GENE', 'GIFT', 'GIRL', 'GIVE', 'GLAD', 'GOAL', 'GOLD', 'GONE', 'GOOD', 'GRAY', 'GREAT',
            'GREEN', 'GREY', 'GROW', 'HAIR', 'HALF', 'HALL', 'HAND', 'HANG', 'HARD', 'HARM', 'HATE', 'HAVE',
            'HEAD', 'HEAL', 'HEAR', 'HEAT', 'HELD', 'HELL', 'HELP', 'HERE', 'HERO', 'HIGH', 'HILL', 'HIRE',
            'HOLD', 'HOLE', 'HOME', 'HOPE', 'HOUR', 'HUGE', 'HUNT', 'HURT', 'IDEA', 'IDLE', 'INCH', 'INTO',
            'IRON', 'ITEM', 'JOIN', 'JOKE', 'JUMP', 'JUST', 'KEEP', 'KICK', 'KILL', 'KIND', 'KING', 'KNEE',
            'KNOW', 'LACK', 'LADY', 'LAKE', 'LAND', 'LAST', 'LATE', 'LAZY', 'LEAD', 'LEAF', 'LEFT', 'LEND',
            'LESS', 'LIFE', 'LIFT', 'LIKE', 'LINE', 'LINK', 'LIST', 'LIVE', 'LOAD', 'LOAN', 'LOCK', 'LONG',
            'LOOK', 'LOSE', 'LOSS', 'LOST', 'LOUD', 'LOVE', 'LUCK', 'LUNG', 'MADE', 'MAIL', 'MAIN', 'MAKE',
            'MALE', 'MANY', 'MARK', 'MASS', 'MEAL', 'MEAN', 'MEAT', 'MEET', 'MENU', 'MILD', 'MILE', 'MILK',
            'MIND', 'MINE', 'MISS', 'MOOD', 'MOON', 'MORE', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NAME', 'NAVY',
            'NEAR', 'NECK', 'NEED', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NONE', 'NOSE', 'NOTE', 'OKAY', 'ONCE',
            'ONLY', 'ONTO', 'OPEN', 'ORAL', 'OVER', 'PACE', 'PACK', 'PAGE', 'PAIN', 'PAIR', 'PARK', 'PART',
            'PASS', 'PAST', 'PATH', 'PAY', 'PEACE', 'PEAK', 'PICK', 'PINK', 'PIPE', 'PLAN', 'PLAY', 'PLOT',
            'PLUS', 'POEM', 'POET', 'POLE', 'POOL', 'POOR', 'POST', 'PULL', 'PUSH', 'RACE', 'RAIN', 'RANK',
            'RARE', 'RATE', 'READ', 'REAL', 'RELY', 'RENT', 'REST', 'RICE', 'RICH', 'RIDE', 'RING', 'RISE',
            'RISK', 'ROAD', 'ROCK', 'ROLE', 'ROLL', 'ROOF', 'ROOM', 'ROOT', 'ROSE', 'ROUGH', 'ROUND', 'RULE',
            'RUSH', 'SAFE', 'SAID', 'SAIL', 'SALE', 'SALT', 'SAME', 'SAND', 'SAVE', 'SEAT', 'SEED', 'SEEK',
            'SEEM', 'SELL', 'SEND', 'SENSE', 'SENT', 'SEVEN', 'SHIP', 'SHOP', 'SHOT', 'SHOW', 'SICK', 'SIDE',
            'SIGN', 'SING', 'SITE', 'SIZE', 'SKIN', 'SLOW', 'SNOW', 'SOFT', 'SOIL', 'SOLD', 'SOME', 'SONG',
            'SOON', 'SORT', 'SOUL', 'STAR', 'STAY', 'STEP', 'STOP', 'SUCH', 'SUIT', 'SUM', 'SURE', 'TAKE',
            'TALE', 'TALK', 'TALL', 'TANK', 'TAPE', 'TASK', 'TAXI', 'TEAM', 'TEAR', 'TELL', 'TEND', 'TERM',
            'TEST', 'TEXT', 'THAN', 'THAT', 'THEM', 'THEN', 'THEY', 'THIN', 'THIS', 'THUS', 'TIDE', 'TIDY',
            'TIME', 'TINY', 'TONE', 'TOOL', 'TOUR', 'TOWN', 'TREE', 'TRIP', 'TRUE', 'TUNE', 'TURN', 'TYPE',
            'UNIT', 'UPON', 'USED', 'USER', 'VAST', 'VERY', 'VIEW', 'VOTE', 'WAIT', 'WAKE', 'WALK', 'WALL',
            'WANT', 'WARM', 'WASH', 'WAVE', 'WEAK', 'WEAR', 'WEEK', 'WELL', 'WENT', 'WERE', 'WEST', 'WHAT',
            'WHEN', 'WIDE', 'WIFE', 'WILD', 'WILL', 'WIND', 'WINE', 'WING', 'WIRE', 'WISE', 'WISH', 'WITH',
            'WOOD', 'WORD', 'WORK', 'YARD', 'YEAR', 'YOUR', 'ZERO', 'ZONE',
            'AGENT', 'BOARD', 'BREAD', 'CHAIN', 'CHAIR', 'CHEST', 'CLEAN', 'CLOCK', 'CLOUD', 'DANCE',
        }

    def _build_trie(self, word_list):
        trie = {}
        _end = '_end_'
        for word in word_list:
            node = trie
            for char in word.upper():
                node = node.setdefault(char, {})
            node[_end] = True
        return trie

    def _generate_grid(self):
        while True:
            # 1. Create an empty grid
            grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
            
            # 2. Select words to embed
            words_to_embed = self.np_random.choice(list(self.word_list), size=15, replace=False)

            # 3. Try to place words
            placed_words = 0
            for word in words_to_embed:
                if placed_words >= 7: break
                word = word.upper()
                shuffled_positions = [(r, c) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
                self.np_random.shuffle(shuffled_positions)
                
                for r_start, c_start in shuffled_positions:
                    path = self._try_place_word(grid, word, r_start, c_start)
                    if path:
                        for i, (r, c) in enumerate(path):
                            grid[r][c] = word[i]
                        placed_words += 1
                        break
            
            # 4. Fill remaining spots
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if grid[r][c] == '':
                        grid[r][c] = self.np_random.choice(list(self.letter_distribution))
            
            solvable = self._find_all_words(grid)
            
            if len(solvable) >= 5:
                self.grid = grid
                self.solvable_words = solvable
                break

    def _try_place_word(self, grid, word, r, c):
        path = [(r, c)]
        
        def can_place(index, r_curr, c_curr):
            if index == len(word):
                return True
            
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r_curr + dr, c_curr + dc
                    if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in path:
                        if grid[nr][nc] == '' or grid[nr][nc] == word[index]:
                            neighbors.append((nr, nc))
            
            self.np_random.shuffle(neighbors)
            for nr, nc in neighbors:
                path.append((nr, nc))
                if can_place(index + 1, nr, nc):
                    return True
                path.pop()
            return False

        if (grid[r][c] == '' or grid[r][c] == word[0]) and can_place(1, r, c):
            return path
        return None

    def _find_all_words(self, grid):
        found_words = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self._dfs_solve(grid, r, c, self.trie, [], "", found_words)
        return found_words

    def _dfs_solve(self, grid, r, c, node, path, current_word, found_words):
        letter = grid[r][c]
        
        if letter not in node:
            return

        new_node = node[letter]
        new_path = path + [(r, c)]
        new_word = current_word + letter

        if '_end_' in new_node:
            found_words.add(new_word)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in new_path:
                    self._dfs_solve(grid, nr, nc, new_node, new_path, new_word, found_words)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_limit = 60.0
        self.time_remaining = self.time_limit
        
        self._generate_grid()
        
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.current_word_path = []
        self.words_found = set()
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_messages = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS
            
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Movement ---
            dr, dc = 0, 0
            if movement == 1: dr = -1 # Up
            elif movement == 2: dr = 1 # Down
            elif movement == 3: dc = -1 # Left
            elif movement == 4: dc = 1 # Right
            
            if dr != 0 or dc != 0:
                self.selector_pos[0] = np.clip(self.selector_pos[0] + dr, 0, self.GRID_ROWS - 1)
                self.selector_pos[1] = np.clip(self.selector_pos[1] + dc, 0, self.GRID_COLS - 1)

            # --- Handle Add Letter (Space Press) ---
            if space_held and not self.last_space_held:
                pos = tuple(self.selector_pos)
                is_valid_selection = True
                
                if pos in self.current_word_path:
                    is_valid_selection = False
                elif self.current_word_path and not self._is_adjacent(pos, self.current_word_path[-1]):
                    is_valid_selection = False
                
                if is_valid_selection:
                    self.current_word_path.append(pos)
                    reward += 0.1
                else:
                    reward -= 0.1
                    self._add_feedback("INVALID MOVE", self.COLOR_FAIL, 30)

            # --- Handle Submit Word (Shift Press) ---
            if shift_held and not self.last_shift_held and self.current_word_path:
                word_str = "".join([self.grid[r][c] for r, c in self.current_word_path]).upper()
                
                if word_str in self.solvable_words and word_str not in self.words_found:
                    self.words_found.add(word_str)
                    word_score = len(word_str)
                    self.score += word_score
                    reward += 2 * word_score
                    self._add_feedback(f"'{word_str}' +{word_score}", self.COLOR_SUCCESS, 60)
                else:
                    reward -= 1
                    reason = "ALREADY FOUND" if word_str in self.words_found else "INVALID WORD"
                    self._add_feedback(reason, self.COLOR_FAIL, 45)
                
                self.current_word_path = []
            
            self.last_space_held = space_held
            self.last_shift_held = shift_held
            
            # --- Check Termination ---
            if len(self.words_found) >= 5:
                self.win_condition_met = True
                self.game_over = True
                reward += 50
                self._add_feedback("YOU WIN!", self.COLOR_SUCCESS, 180)

            if self.time_remaining <= 0:
                self.time_remaining = 0
                if not self.game_over:
                    self.game_over = True
                    self._add_feedback("TIME'S UP!", self.COLOR_FAIL, 180)
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_adjacent(self, pos1, pos2):
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) == 1

    def _add_feedback(self, text, color, lifetime):
        self.feedback_messages.append([text, color, lifetime, 255])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Words Found
        words_found_text = self.font_ui.render(f"WORDS: {len(self.words_found)} / 5", True, self.COLOR_UI_TEXT)
        self.screen.blit(words_found_text, (self.SCREEN_WIDTH - words_found_text.get_width() - 20, 15))
        
        # Current Word
        current_word_str = "".join([self.grid[r][c] for r, c in self.current_word_path])
        word_display_text = self.font_ui.render(f"Current: {current_word_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(word_display_text, (20, self.SCREEN_HEIGHT - 30))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {max(0, math.ceil(self.time_remaining))}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 15))
        
        bar_width = 200
        bar_height = 10
        bar_x = self.SCREEN_WIDTH // 2 - bar_width // 2
        bar_y = 40
        
        time_ratio = self.time_remaining / self.time_limit if self.time_limit > 0 else 0
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, int(bar_width * time_ratio), bar_height), border_radius=5)

    def _render_feedback(self):
        for i in range(len(self.feedback_messages) - 1, -1, -1):
            msg = self.feedback_messages[i]
            msg[2] -= 1 # Decrement lifetime
            if msg[2] < 20:
                msg[3] = max(0, msg[3] - 13) # Fade out
            
            if msg[2] <= 0:
                self.feedback_messages.pop(i)
                continue
            
            text_surf = self.font_feedback.render(msg[0], True, msg[1])
            text_surf.set_alpha(msg[3])
            text_pos = (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50)
            self.screen.blit(text_surf, text_pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Grid Background ---
        grid_rect = pygame.Rect(self.GRID_MARGIN_X - 5, self.GRID_MARGIN_Y - 5,
                                self.GRID_COLS * self.GRID_CELL_SIZE + 10, self.GRID_ROWS * self.GRID_CELL_SIZE + 10)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # --- Render Tiles and Letters ---
        if self.grid:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    cell_x = self.GRID_MARGIN_X + c * self.GRID_CELL_SIZE
                    cell_y = self.GRID_MARGIN_Y + r * self.GRID_CELL_SIZE
                    
                    tile_color = self.COLOR_TILE
                    text_color = self.COLOR_TILE_TEXT
                    
                    if (r, c) in self.current_word_path:
                        tile_color = self.COLOR_PATH
                        text_color = self.COLOR_PATH_TEXT
                        
                    pygame.draw.rect(self.screen, tile_color, 
                                     (cell_x + 2, cell_y + 2, self.GRID_CELL_SIZE - 4, self.GRID_CELL_SIZE - 4), 
                                     border_radius=5)

                    letter_surf = self.font_tile.render(self.grid[r][c], True, text_color)
                    text_rect = letter_surf.get_rect(center=(cell_x + self.GRID_CELL_SIZE // 2, cell_y + self.GRID_CELL_SIZE // 2))
                    self.screen.blit(letter_surf, text_rect)

        # --- Render Path Lines ---
        if len(self.current_word_path) > 1:
            points = []
            for r, c in self.current_word_path:
                points.append((self.GRID_MARGIN_X + c * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
                               self.GRID_MARGIN_Y + r * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2))
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, points, 5)

        # --- Render Selector ---
        sel_r, sel_c = self.selector_pos
        sel_x = self.GRID_MARGIN_X + sel_c * self.GRID_CELL_SIZE
        sel_y = self.GRID_MARGIN_Y + sel_r * self.GRID_CELL_SIZE
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, 
                         (sel_x, sel_y, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), line_width, border_radius=7)

        self._render_ui()
        self._render_feedback()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "words_found": len(self.words_found),
            "solvable_words_count": len(self.solvable_words),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To run with a display, comment out the `os.environ` line at the top of the file.
    # This __main__ block is for human interaction and debugging.
    try:
        env = GameEnv()
        
        obs, info = env.reset(seed=42)
        terminated = False
        
        pygame.display.set_caption("Word Grid Game")
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        running = True
        while running:
            # --- Action Mapping for Human ---
            movement = 0 # no-op
            space_held = False
            shift_held = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    terminated = False

            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Run at 30 FPS

        env.close()
    except pygame.error as e:
        print(f"Caught Pygame error: {e}")
        print("This is expected if you are running in a headless environment.")
        print("To run with a display, comment out the 'os.environ' line at the top of the script.")