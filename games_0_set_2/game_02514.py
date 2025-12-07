
# Generated: 2025-08-27T20:38:09.200912
# Source Brief: brief_02514.md
# Brief Index: 2514

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Hold Space and move to form a word. "
        "Release Space to submit. Press Shift to cancel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find and form words on the grid by dragging letters. "
        "Score points by creating longer words before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_TOP_LEFT_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_TOP_LEFT_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 500

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINE = (40, 50, 60)
    COLOR_LETTER = (220, 220, 240)
    COLOR_CURSOR = (255, 200, 0, 150)
    COLOR_SELECTION_BG = (80, 120, 180)
    COLOR_SELECTION_PATH = (255, 220, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_BAR_FULL = (0, 200, 100)
    COLOR_TIMER_BAR_EMPTY = (220, 50, 50)
    PARTICLE_COLORS = [(255, 220, 100), (255, 180, 50), (255, 255, 180)]

    # Game Data
    LETTER_FREQUENCIES = "EEEEEEEEEEEEAAAAAAAAAIIIIIIIIIOOOOOOOONNNNNNRRRRRRTTTTTTLLLLSSSSUUUUDDDDGGGBBCCMMPPFFHHVVWWYYKJXQZ"
    WORD_LIST = {
        'cat', 'dog', 'sun', 'run', 'big', 'red', 'bed', 'egg', 'fun', 'get', 'hat', 'jet', 'log', 'map', 'net',
        'pen', 'rat', 'sit', 'ten', 'vet', 'win', 'yes', 'zip', 'ace', 'act', 'add', 'age', 'ago', 'aid', 'aim',
        'air', 'ale', 'all', 'and', 'any', 'ape', 'apt', 'arc', 'are', 'ark', 'arm', 'art', 'ash', 'ask', 'ate',
        'awe', 'axe', 'bad', 'bag', 'ban', 'bat', 'bee', 'beg', 'bet', 'bid', 'bin', 'bit', 'boa', 'bog', 'boo',
        'bot', 'bow', 'box', 'boy', 'bud', 'bug', 'bum', 'bun', 'bus', 'but', 'buy', 'bye', 'cab', 'cad', 'cam',
        'can', 'cap', 'car', 'cob', 'cod', 'cog', 'con', 'coo', 'cop', 'cot', 'cow', 'coy', 'cry', 'cub', 'cud',
        'cup', 'cur', 'cut', 'dad', 'dam', 'dan', 'day', 'den', 'dew', 'did', 'die', 'dig', 'dim', 'din', 'dip',
        'doe', 'don', 'dot', 'dry', 'dub', 'dud', 'due', 'dug', 'dun', 'duo', 'dye', 'ear', 'eat', 'eel', 'eft',
        'ego', 'eke', 'elf', 'elk', 'elm', 'emu', 'end', 'eon', 'era', 'ere', 'erg', 'err', 'eta', 'eve', 'ewe',
        'eye', 'fad', 'fag', 'fan', 'far', 'fat', 'fed', 'fee', 'fen', 'few', 'fey', 'fez', 'fib', 'fig', 'fin',
        'fir', 'fit', 'fix', 'flu', 'fly', 'fob', 'foe', 'fog', 'foh', 'fon', 'fop', 'for', 'fox', 'fry', 'fud',
        'fug', 'fun', 'fur', 'gab', 'gad', 'gag', 'gal', 'gam', 'gan', 'gap', 'gar', 'gas', 'gat', 'gay', 'gee',
        'gel', 'gem', 'get', 'gig', 'gin', 'gip', 'git', 'gnu', 'goa', 'gob', 'god', 'goo', 'got', 'gum', 'gun',
        'gut', 'guy', 'gym', 'gyp', 'had', 'hag', 'hah', 'ham', 'has', 'hat', 'hay', 'heh', 'hem', 'hen', 'her',
        'hes', 'hew', 'hex', 'hey', 'hic', 'hid', 'hie', 'him', 'hin', 'hip', 'his', 'hit', 'hmm', 'hob', 'hod',
        'hoe', 'hog', 'hon', 'hop', 'hot', 'how', 'hub', 'hue', 'hug', 'huh', 'hum', 'hun', 'hut', 'ice', 'icy',
        'igg', 'ilk', 'ill', 'imp', 'ink', 'inn', 'ion', 'ire', 'irk', 'ism', 'its', 'ivy', 'jab', 'jag', 'jam',
        'jar', 'jaw', 'jay', 'jet', 'jew', 'jib', 'jig', 'job', 'joe', 'jog', 'jot', 'jow', 'joy', 'jug', 'jun',
        'jus', 'jut', 'kef', 'keg', 'ken', 'kep', 'kex', 'key', 'khi', 'kid', 'kif', 'kin', 'kip', 'kir', 'kis',
        'kit', 'koa', 'koi', 'kop', 'kor', 'kos', 'kue', 'kyu', 'lab', 'lac', 'lad', 'lag', 'lam', 'lap', 'lar',
        'las', 'lat', 'lav', 'law', 'lax', 'lay', 'lea', 'led', 'lee', 'leg', 'lek', 'let', 'leu', 'lev', 'lex',
        'ley', 'lib', 'lid', 'lie', 'lin', 'lip', 'lis', 'lit', 'lob', 'log', 'loo', 'lop', 'lot', 'low', 'lox',
        'lug', 'lum', 'luv', 'lux', 'lye', 'mad', 'mae', 'mag', 'man', 'map', 'mar', 'mas', 'mat', 'maw', 'max',
        'may', 'med', 'meg', 'mel', 'mem', 'men', 'met', 'mew', 'mho', 'mib', 'mic', 'mid', 'mig', 'mil', 'mim',
        'mir', 'mis', 'mix', 'moa', 'mob', 'moc', 'mod', 'mog', 'moi', 'mol', 'mom', 'mon', 'moo', 'mop', 'mor',
        'mos', 'mot', 'mow', 'mud', 'mug', 'mum', 'mun', 'mus', 'mut', 'myc', 'nab', 'nae', 'nag', 'nah', 'nam',
        'nan', 'nap', 'naw', 'nay', 'neb', 'nee', 'neg', 'net', 'new', 'nib', 'nil', 'nim', 'nip', 'nit', 'nix',
        'nob', 'nod', 'nog', 'noh', 'nom', 'noo', 'nor', 'nos', 'not', 'now', 'nth', 'nub', 'nun', 'nus', 'nut',
        'oaf', 'oak', 'oar', 'oat', 'oba', 'obe', 'obi', 'obo', 'obs', 'oca', 'och', 'oda', 'odd', 'ode', 'ods',
        'oes', 'off', 'oft', 'ohm', 'oho', 'ohs', 'oik', 'oil', 'oka', 'oke', 'old', 'ole', 'oma', 'omen', 'one',
        'ono', 'ons', 'ooh', 'oot', 'ope', 'ops', 'opt', 'ora', 'orb', 'orc', 'ore', 'orf', 'org', 'ors', 'ort',
        'ose', 'oud', 'our', 'out', 'ova', 'owe', 'owl', 'own', 'owt', 'oxo', 'oxy', 'pac', 'pad', 'pah', 'pal',
        'pam', 'pan', 'pap', 'par', 'pas', 'pat', 'paw', 'pax', 'pay', 'pea', 'pec', 'ped', 'pee', 'peg', 'peh',
        'pen', 'pep', 'per', 'pes', 'pet', 'pew', 'phi', 'pho', 'pht', 'pia', 'pic', 'pie', 'pig', 'pin', 'pip',
        'pis', 'pit', 'piu', 'pix', 'ply', 'pod', 'poh', 'poi', 'pol', 'pom', 'poo', 'pop', 'pot', 'pow', 'pox',
        'pro', 'pry', 'psi', 'pst', 'pub', 'pud', 'pug', 'pul', 'pun', 'pup', 'pur', 'pus', 'put', 'pya', 'pye',
        'pyx', 'qat', 'qua', 'rad', 'rag', 'rah', 'rai', 'raj', 'ram', 'ran', 'rap', 'ras', 'rat', 'raw', 'rax',
        'ray', 'reb', 'rec', 'red', 'ree', 'ref', 'reg', 'rei', 'rem', 'rep', 'res', 'ret', 'rev', 'rex', 'rho',
        'ria', 'rib', 'rid', 'rif', 'rig', 'rim', 'rin', 'rip', 'rob', 'roc', 'rod', 'roe', 'rom', 'rot', 'row',
        'rub', 'rue', 'rug', 'rum', 'run', 'rut', 'rye', 'sab', 'sac', 'sad', 'sae', 'sag', 'sal', 'sap', 'sar',
        'sat', 'sau', 'saw', 'sax', 'say', 'sea', 'sec', 'see', 'seg', 'sei', 'sel', 'sen', 'ser', 'set', 'sew',
        'sex', 'sha', 'she', 'shh', 'sho', 'shy', 'sib', 'sic', 'sig', 'sim', 'sin', 'sip', 'sir', 'sis', 'sit',
        'six', 'ska', 'ski', 'sky', 'sly', 'sob', 'sod', 'sol', 'som', 'son', 'sop', 'sos', 'sot', 'sou', 'sow',
        'sox', 'soy', 'spa', 'spy', 'sri', 'sty', 'sub', 'sue', 'suk', 'sum', 'sun', 'sup', 'suq', 'syn', 'tab',
        'tad', 'tae', 'tag', 'taj', 'tak', 'tam', 'tan', 'tao', 'tap', 'tar', 'tas', 'tat', 'tau', 'tav', 'taw',
        'tax', 'tea', 'tec', 'ted', 'tee', 'teg', 'tel', 'ten', 'tes', 'tet', 'tew', 'the', 'tho', 'thy', 'tic',
        'tie', 'til', 'tin', 'tip', 'tis', 'tit', 'tod', 'toe', 'tog', 'tom', 'ton', 'too', 'top', 'tor', 'tot',
        'tow', 'toy', 'try', 'tsk', 'tub', 'tug', 'tui', 'tun', 'tup', 'tut', 'tux', 'twa', 'two', 'tye', 'udo',
        'ugh', 'uke', 'ulu', 'umm', 'ump', 'uns', 'upo', 'ups', 'urb', 'urd', 'ure', 'urn', 'urp', 'use', 'uta',
        'ute', 'uts', 'vac', 'van', 'var', 'vas', 'vat', 'vau', 'vav', 'vaw', 'vee', 'veg', 'vet', 'vex', 'via',
        'vid', 'vie', 'vig', 'vim', 'vin', 'vis', 'voe', 'vog', 'vow', 'vox', 'vug', 'vum', 'wab', 'wad', 'wae',
        'wag', 'wan', 'wap', 'war', 'was', 'wat', 'waw', 'wax', 'way', 'web', 'wed', 'wee', 'wen', 'wet', 'wha',
        'who', 'why', 'wig', 'win', 'wis', 'wit', 'wiz', 'woe', 'wog', 'wok', 'won', 'woo', 'wop', 'wos', 'wot',
        'wow', 'wry', 'wud', 'wye', 'wyn', 'xis', 'yag', 'yah', 'yak', 'yam', 'yap', 'yar', 'yaw', 'yay', 'yea',
        'yeh', 'yen', 'yep', 'yes', 'yet', 'yew', 'yid', 'yin', 'yip', 'yob', 'yod', 'yok', 'yom', 'yon', 'you',
        'yow', 'yuk', 'yum', 'yup', 'zag', 'zap', 'zas', 'zax', 'zed', 'zee', 'zek', 'zen', 'zep', 'zig', 'zin',
        'zip', 'zit', 'zoa', 'zoo', 'zuz', 'word', 'game', 'play', 'code', 'grid', 'time', 'score', 'find', 'make',
        'list', 'good', 'best', 'fast', 'slow', 'long', 'work', 'test', 'line', 'cell', 'path', 'move', 'drag',
        'drop', 'fall', 'fill', 'clear', 'start', 'stop', 'over', 'win', 'lose', 'deep', 'mind', 'gym', 'agent'
    }

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
        self.font_letter = pygame.font.SysFont("dejavusansmono", 30, bold=True)
        self.font_ui = pygame.font.SysFont("dejavusans", 24)
        self.font_msg = pygame.font.SysFont("dejavusans", 48, bold=True)
        
        self.grid = None
        self.cursor_pos = None
        self.is_selecting = None
        self.selection_path = None
        self.last_space_held = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.win = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.is_selecting = False
        self.selection_path = []
        self.last_space_held = False
        self.particles = []
        
        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._update_game_logic(movement, space_held, shift_held)
        
        # Word submission logic
        space_pressed = space_held and not self.last_space_held
        space_released = not space_held and self.last_space_held
        
        if space_pressed and not self.is_selecting:
            self.is_selecting = True
            self.selection_path = [list(self.cursor_pos)]
            # sfx: select_start

        elif space_released and self.is_selecting:
            reward += self._submit_word()
            self.is_selecting = False
            self.selection_path = []
        
        if shift_held and self.is_selecting:
            self.is_selecting = False
            self.selection_path = []
            # sfx: cancel_selection
            
        self.last_space_held = space_held
        
        # Update timer and check for termination
        self.time_remaining -= 1 / self.FPS
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100 # Win bonus
            else:
                reward -= 10 # Time out penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_logic(self, movement, space_held, shift_held):
        # --- Handle cursor movement ---
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # --- Handle word selection path ---
        if self.is_selecting and self.cursor_pos != prev_cursor_pos:
            if self.cursor_pos not in self.selection_path:
                last_pos = self.selection_path[-1]
                if self._is_adjacent(self.cursor_pos, last_pos):
                    self.selection_path.append(list(self.cursor_pos))
                    # sfx: select_extend
            elif len(self.selection_path) > 1 and self.cursor_pos == self.selection_path[-2]:
                # Allow backtracking
                self.selection_path.pop()
                # sfx: select_backtrack
        
        # --- Update particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['radius'] += p['growth']
            p['life'] -= 1

    def _submit_word(self):
        word = "".join([self.grid[y][x] for x, y in self.selection_path]).lower()
        
        if len(word) >= 3 and word in self.WORD_LIST:
            # Valid word
            self.score += len(word) ** 2
            
            for x, y in self.selection_path:
                self.grid[y][x] = None
                self._create_particles(x, y)
            
            self._refill_grid()
            # sfx: word_success
            return len(word) + 0.1
        else:
            # Invalid word
            # sfx: word_fail
            return -0.1

    def _create_particles(self, grid_x, grid_y):
        px, py = self._grid_to_pixel(grid_x, grid_y, center=True)
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': random.choice(self.PARTICLE_COLORS)
            })

    def _refill_grid(self):
        for x in range(self.GRID_SIZE):
            empty_count = 0
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y][x] is None:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[y + empty_count][x] = self.grid[y][x]
                    self.grid[y][x] = None
            
            for y in range(empty_count):
                self.grid[y][x] = self.np_random.choice(list(self.LETTER_FREQUENCIES))

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
        elif self.time_remaining <= 0:
            self.game_over = True
            self.win = False
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
        
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_remaining}

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_x = self.GRID_TOP_LEFT_X + i * self.CELL_SIZE
            start_y = self.GRID_TOP_LEFT_Y
            end_y = self.GRID_TOP_LEFT_Y + self.GRID_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (start_x, start_y), (start_x, end_y), 1)

            start_x = self.GRID_TOP_LEFT_X
            start_y = self.GRID_TOP_LEFT_Y + i * self.CELL_SIZE
            end_x = self.GRID_TOP_LEFT_X + self.GRID_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (start_x, start_y), (end_x, start_y), 1)

        # Draw selection path
        if self.is_selecting and len(self.selection_path) > 1:
            points = [self._grid_to_pixel(x, y, center=True) for x, y in self.selection_path]
            pygame.draw.lines(self.screen, self.COLOR_SELECTION_PATH, False, points, 5)
        
        # Draw letters and selection highlights
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                px, py = self._grid_to_pixel(x, y)
                
                # Draw selection background
                if [x, y] in self.selection_path:
                    pygame.draw.rect(self.screen, self.COLOR_SELECTION_BG, (px, py, self.CELL_SIZE, self.CELL_SIZE))

                # Draw letters
                letter = self.grid[y][x]
                if letter:
                    text_surf = self.font_letter.render(letter, True, self.COLOR_LETTER)
                    text_rect = text_surf.get_rect(center=(px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2))
                    self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(*self.cursor_pos)
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, (cursor_px, cursor_py))

        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], alpha))


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = max(0, self.time_remaining / self.GAME_DURATION_SECONDS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        # Interpolate color from green to red
        timer_color = (
            self.COLOR_TIMER_BAR_EMPTY[0] + (self.COLOR_TIMER_BAR_FULL[0] - self.COLOR_TIMER_BAR_EMPTY[0]) * timer_ratio,
            self.COLOR_TIMER_BAR_EMPTY[1] + (self.COLOR_TIMER_BAR_FULL[1] - self.COLOR_TIMER_BAR_EMPTY[1]) * timer_ratio,
            self.COLOR_TIMER_BAR_EMPTY[2] + (self.COLOR_TIMER_BAR_FULL[2] - self.COLOR_TIMER_BAR_EMPTY[2]) * timer_ratio,
        )

        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, bar_width * timer_ratio, bar_height))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = "YOU WIN!" if self.win else "TIME'S UP!"
            msg_surf = self.font_msg.render(msg_text, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _generate_grid(self):
        self.grid = [
            [self.np_random.choice(list(self.LETTER_FREQUENCIES)) for _ in range(self.GRID_SIZE)]
            for _ in range(self.GRID_SIZE)
        ]

    def _is_adjacent(self, pos1, pos2):
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) == 1
    
    def _grid_to_pixel(self, grid_x, grid_y, center=False):
        px = self.GRID_TOP_LEFT_X + grid_x * self.CELL_SIZE
        py = self.GRID_TOP_LEFT_Y + grid_y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return px, py

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Word Grid Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Key down events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1

            # Key up events
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0
        
        # Continuous key state for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        else:
            movement = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)

    env.close()