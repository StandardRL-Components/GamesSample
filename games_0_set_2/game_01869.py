
# Generated: 2025-08-28T02:58:22.215417
# Source Brief: brief_01869.md
# Brief Index: 1869

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a letter. "
        "Press Shift to submit the current word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Weave words by connecting adjacent letters on the grid. "
        "Find 5 words before the 60-second timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_DIM = 6
    TIME_LIMIT = 60.0
    WORDS_TO_WIN = 5
    MAX_STEPS = int(TIME_LIMIT * 30) + 1 # 30 FPS

    # --- Colors ---
    COLOR_BG = (25, 28, 32)
    COLOR_GRID_BG = (45, 50, 56)
    COLOR_GRID_LINE = (65, 72, 82)
    COLOR_LETTER_DEFAULT = (200, 205, 215)
    COLOR_LETTER_SELECTED = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PATH = (255, 215, 0, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_VALID_WORD = (100, 255, 100)
    COLOR_INVALID_WORD = (255, 100, 100)
    COLOR_TIMER_HIGH = (100, 220, 100)
    COLOR_TIMER_MID = (255, 180, 0)
    COLOR_TIMER_LOW = (255, 80, 80)

    # --- Word List (no external files) ---
    WORD_LIST = set(['AT', 'ON', 'IN', 'IS', 'IT', 'GO', 'DO', 'IF', 'ME', 'MY', 'NO', 'SO', 'TO', 'UP', 'US', 'WE', 'BE', 'BY', 'HE', 'OR', 'AN', 'ACE', 'ACT', 'ADD', 'AGO', 'AID', 'AIM', 'AIR', 'ALE', 'ALL', 'AND', 'ANT', 'ANY', 'APE', 'APT', 'ARC', 'ARE', 'ARM', 'ART', 'ASH', 'ASK', 'AWE', 'AXE', 'BAD', 'BAG', 'BAN', 'BAT', 'BED', 'BEE', 'BEG', 'BET', 'BID', 'BIG', 'BIN', 'BIT', 'BOX', 'BOY', 'BUG', 'BUN', 'BUS', 'BUT', 'BUY', 'BYE', 'CAB', 'CAN', 'CAP', 'CAR', 'CAT', 'COD', 'COG', 'CON', 'COP', 'COT', 'COW', 'CRY', 'CUB', 'CUP', 'CUT', 'DAD', 'DAY', 'DEN', 'DEW', 'DID', 'DIE', 'DIG', 'DIM', 'DIP', 'DOG', 'DON', 'DOT', 'DRY', 'EAR', 'EAT', 'EGG', 'EGO', 'END', 'EON', 'ERA', 'EVE', 'EYE', 'FAD', 'FAN', 'FAR', 'FAT', 'FED', 'FEE', 'FEW', 'FIB', 'FIG', 'FIN', 'FIR', 'FIT', 'FIX', 'FLY', 'FOG', 'FOR', 'FOX', 'FRY', 'FUN', 'FUR', 'GAP', 'GAS', 'GEL', 'GEM', 'GET', 'GIN', 'GOD', 'GOT', 'GUM', 'GUN', 'GUT', 'GUY', 'GYM', 'HAD', 'HAS', 'HAT', 'HAY', 'HEM', 'HEN', 'HER', 'HEX', 'HEY', 'HID', 'HIM', 'HIP', 'HIS', 'HIT', 'HOG', 'HOP', 'HOT', 'HOW', 'HUG', 'HUM', 'HUT', 'ICE', 'ICY', 'ILL', 'INK', 'INN', 'ION', 'JAB', 'JAM', 'JAR', 'JAW', 'JAY', 'JET', 'JIG', 'JOB', 'JOG', 'JOT', 'JOY', 'JUG', 'KIN', 'KIT', 'LAB', 'LAD', 'LAP', 'LAW', 'LAY', 'LED', 'LEG', 'LET', 'LID', 'LIE', 'LIP', 'LIT', 'LOG', 'LOT', 'LOW', 'MAN', 'MAP', 'MAT', 'MAY', 'MEN', 'MET', 'MUD', 'MUG', 'MUM', 'NAB', 'NAG', 'NAP', 'NET', 'NEW', 'NIB', 'NIL', 'NIP', 'NOD', 'NOR', 'NOT', 'NOW', 'NUN', 'NUT', 'OAK', 'OAR', 'OAT', 'ODD', 'ODE', 'OFF', 'OIL', 'OLD', 'ONE', 'OPT', 'ORB', 'ORE', 'OUR', 'OUT', 'OWE', 'OWL', 'OWN', 'PAD', 'PAL', 'PAN', 'PAR', 'PAT', 'PAW', 'PAY', 'PEA', 'PEG', 'PEN', 'PEP', 'PER', 'PET', 'PIE', 'PIG', 'PIN', 'PIT', 'PLY', 'POD', 'POP', 'POT', 'PRO', 'PRY', 'PUB', 'PUG', 'PUN', 'PUP', 'PUT', 'RAD', 'RAG', 'RAM', 'RAN', 'RAP', 'RAT', 'RAW', 'RAY', 'RED', 'REP', 'RIB', 'RID', 'RIG', 'RIM', 'RIP', 'ROB', 'ROD', 'ROE', 'ROT', 'ROW', 'RUB', 'RUE', 'RUG', 'RUM', 'RUN', 'RUT', 'RYE', 'SAD', 'SAG', 'SAP', 'SAT', 'SAW', 'SAY', 'SEE', 'SET', 'SEW', 'SHE', 'SHY', 'SIN', 'SIP', 'SIR', 'SIT', 'SKI', 'SKY', 'SLY', 'SOB', 'SOD', 'SON', 'SOW', 'SPY', 'STY', 'SUB', 'SUE', 'SUM', 'SUN', 'TAB', 'TAD', 'TAG', 'TAN', 'TAP', 'TAR', 'TEA', 'TEE', 'TEN', 'THE', 'TIE', 'TIN', 'TIP', 'TOE', 'TOM', 'TON', 'TOP', 'TOW', 'TOY', 'TRY', 'TUB', 'TUG', 'USE', 'VAN', 'VAT', 'VET', 'VIE', 'VOW', 'WAD', 'WAG', 'WAR', 'WAS', 'WAX', 'WAY', 'WEB', 'WED', 'WET', 'WHO', 'WHY', 'WIG', 'WIN', 'WIT', 'WON', 'YAK', 'YAM', 'YAP', 'YAW', 'YEA', 'YEN', 'YEP', 'YES', 'YET', 'YOU', 'ZAP', 'ZEN', 'ZIG', 'ZIP', 'WORD', 'GAME', 'PLAY', 'CODE', 'TEST', 'GRID', 'CELL', 'LINE', 'PATH', 'LINK', 'FIND', 'MAKE', 'GOOD', 'BEST', 'FAST', 'SLOW', 'QUIT', 'HELP', 'TIME', 'LEFT', 'RIGHT', 'DOWN', 'MOVE', 'JUMP', 'PYTHON', 'AGENT', 'REWARD', 'SCORE', 'STATE', 'ACTION', 'WEAVE', 'LETTER'])
    LETTER_FREQUENCIES = {
        'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7, 'S': 6.3, 'H': 6.1, 'R': 6.0,
        'D': 4.3, 'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4, 'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0,
        'P': 1.9, 'B': 1.5, 'V': 1.0, 'K': 0.8, 'J': 0.2, 'X': 0.2, 'Q': 0.1, 'Z': 0.1
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
        
        self.font_large = pygame.font.Font(None, 56)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.current_word = ""
        self.current_path = []
        self.submitted_words = set()
        self.timer = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_messages = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIME_LIMIT
        
        self.grid = self._generate_grid()
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.current_word = ""
        self.current_path = []
        self.submitted_words = set()
        
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_messages = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        if not self.game_over:
            self.timer -= 1.0 / 30.0 # Assuming 30 FPS
            self.steps += 1
            
            # Handle movement
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            if dx != 0 or dy != 0:
                self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_DIM
                self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_DIM

            # Handle letter selection
            if space_pressed:
                reward += self._select_letter()
            
            # Handle word submission
            if shift_pressed:
                reward += self._submit_word()
        
        # Update feedback animations
        self._update_feedback()
        
        # Check termination conditions
        terminated = (self.timer <= 0 or 
                      len(self.submitted_words) >= self.WORDS_TO_WIN or 
                      self.steps >= self.MAX_STEPS)
        
        if terminated and not self.game_over:
            self.game_over = True
            if len(self.submitted_words) >= self.WORDS_TO_WIN:
                reward += 10 # Win bonus
                self._add_feedback(f"YOU WIN!", self.COLOR_VALID_WORD, 120, 80)
            else:
                self._add_feedback(f"TIME'S UP!", self.COLOR_INVALID_WORD, 120, 80)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _select_letter(self):
        cx, cy = self.cursor_pos
        
        if self.current_path and (cx, cy) == self.current_path[-1]:
            return 0.0
        
        is_valid_selection = False
        if not self.current_path:
            is_valid_selection = True # First letter
        else:
            last_x, last_y = self.current_path[-1]
            if abs(cx - last_x) <= 1 and abs(cy - last_y) <= 1 and (cx, cy) not in self.current_path:
                is_valid_selection = True
        
        if is_valid_selection:
            self.current_path.append((cx, cy))
            self.current_word += self.grid[cy][cx]
            return 0.1 # Small reward for extending the word
        else:
            self.current_word = ""
            self.current_path = []
            return 0.0

    def _submit_word(self):
        reward = 0.0
        word = self.current_word
        
        if len(word) > 1 and word in self.WORD_LIST and word not in self.submitted_words:
            word_score = 1.0 + 0.5 * len(word)
            reward += word_score
            self.score += word_score
            self.submitted_words.add(word)
            # Sound effect placeholder: pygame.mixer.Sound('valid_word.wav').play()
            self._add_feedback(f"+{word_score:.1f}", self.COLOR_VALID_WORD, 60)
        elif len(word) > 0:
            reward -= 1.0
            # Sound effect placeholder: pygame.mixer.Sound('invalid_word.wav').play()
            self._add_feedback("Invalid", self.COLOR_INVALID_WORD, 60)
            
        self.current_word = ""
        self.current_path = []
        return reward

    def _generate_grid(self):
        letters, weights = zip(*self.LETTER_FREQUENCIES.items())
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        grid_letters = self.np_random.choice(
            letters, 
            size=self.GRID_DIM * self.GRID_DIM, 
            p=probabilities
        )
        return grid_letters.reshape((self.GRID_DIM, self.GRID_DIM))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_size = 300
        cell_size = grid_size / self.GRID_DIM
        grid_x = (self.SCREEN_WIDTH - grid_size) / 2 - 100
        grid_y = (self.SCREEN_HEIGHT - grid_size) / 2
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (grid_x, grid_y, grid_size, grid_size), border_radius=10)
        
        if len(self.current_path) > 1:
            points = []
            for (px, py) in self.current_path:
                center_x = int(grid_x + (px + 0.5) * cell_size)
                center_y = int(grid_y + (py + 0.5) * cell_size)
                points.append((center_x, center_y))
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, points, 5)

        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                rect = pygame.Rect(grid_x + x * cell_size, grid_y + y * cell_size, cell_size, cell_size)
                center_x, center_y = int(rect.centerx), int(rect.centery)
                
                if (x, y) in self.current_path:
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(cell_size * 0.4), self.COLOR_LETTER_SELECTED)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(cell_size * 0.4), self.COLOR_LETTER_SELECTED)
                    letter_color = self.COLOR_GRID_BG
                else:
                    letter_color = self.COLOR_LETTER_DEFAULT

                letter = self.grid[y][x]
                text_surf = self.font_large.render(letter, True, letter_color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)
        
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(grid_x + cursor_x * cell_size, grid_y + cursor_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=5)

    def _render_ui(self):
        word_text = self.current_word if self.current_word else "_"
        text_surf = self.font_large.render(word_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2 - 100, 40))
        self.screen.blit(text_surf, text_rect)

        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 30))
        self.screen.blit(score_surf, score_rect)
        
        time_ratio = max(0, self.timer / self.TIME_LIMIT)
        if time_ratio > 0.5: timer_color = self.COLOR_TIMER_HIGH
        elif time_ratio > 0.2: timer_color = self.COLOR_TIMER_MID
        else: timer_color = self.COLOR_TIMER_LOW
            
        timer_text = f"{max(0, self.timer):.1f}"
        timer_surf = self.font_medium.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_surf, timer_rect)
        
        found_title_surf = self.font_medium.render("Found Words", True, self.COLOR_UI_TEXT)
        self.screen.blit(found_title_surf, (self.SCREEN_WIDTH - 200, 70))
        
        y_offset = 110
        for i, word in enumerate(list(self.submitted_words)[:10]):
            word_surf = self.font_small.render(word, True, self.COLOR_VALID_WORD)
            self.screen.blit(word_surf, (self.SCREEN_WIDTH - 180, y_offset + i * 22))

        for msg in self.feedback_messages:
            alpha = int(255 * (msg['timer'] / msg['max_timer']))
            color = msg['color']
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            font = pygame.font.Font(None, msg['size'])
            text_surf = font.render(msg['text'], True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=msg['pos'])
            temp_surf.blit(text_surf, text_rect)
            self.screen.blit(temp_surf, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "found_words": len(self.submitted_words),
        }
        
    def _add_feedback(self, text, color, duration, size=48):
        pos = (self.SCREEN_WIDTH / 2 - 100, 80)
        self.feedback_messages.append({
            "text": text, "color": color, "timer": duration,
            "max_timer": duration, "pos": pos, "size": size
        })
    
    def _update_feedback(self):
        for msg in self.feedback_messages:
            msg['timer'] -= 1
        self.feedback_messages = [m for m in self.feedback_messages if m['timer'] > 0]

    def close(self):
        pygame.font.quit()
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Weave")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    print(env.game_description)

    while not done:
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30)

    env.close()
    pygame.quit()