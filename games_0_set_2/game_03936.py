
# Generated: 2025-08-28T00:54:13.116174
# Source Brief: brief_03936.md
# Brief Index: 3936

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame


# A curated list of common English words for the game
WORD_LIST = {
    'cat', 'dog', 'sun', 'run', 'fun', 'big', 'red', 'bed', 'pen', 'ten', 'ant', 'art', 'eat', 'tea', 'sea', 'see', 'ape', 'ace', 'act', 'add', 'ago', 'air', 'ale', 'all', 'and', 'any', 'arm', 'ask', 'ate', 'awe', 'axe', 'bad', 'bag', 'ban', 'bat', 'bay', 'bee', 'beg', 'bet', 'bid', 'bin', 'bit', 'bow', 'box', 'boy', 'bud', 'bug', 'bun', 'bus', 'but', 'buy', 'bye', 'cab', 'cap', 'car', 'cop', 'cot', 'cow', 'cry', 'cub', 'cup', 'cut', 'dad', 'day', 'den', 'did', 'die', 'dig', 'dim', 'dip', 'dot', 'dry', 'dug', 'due', 'ear', 'egg', 'ego', 'elf', 'end', 'era', 'eye', 'fan', 'far', 'fat', 'fed', 'fee', 'few', 'fig', 'fin', 'fit', 'fix', 'fly', 'foe', 'fog', 'for', 'fox', 'fry', 'fun', 'fur', 'gap', 'gas', 'gem', 'get', 'gig', 'gin', 'god', 'got', 'gum', 'gun', 'gut', 'guy', 'gym', 'had', 'has', 'hat', 'hay', 'hem', 'hen', 'her', 'hey', 'hid', 'him', 'hip', 'his', 'hit', 'hog', 'hop', 'hot', 'how', 'hub', 'hug', 'hum', 'hut', 'ice', 'icy', 'ill', 'ink', 'inn', 'ion', 'its', 'jam', 'jar', 'jaw', 'jay', 'jet', 'jig', 'job', 'jog', 'joy', 'jug', 'jut', 'keg', 'key', 'kid', 'kin', 'kit', 'lab', 'lad', 'lag', 'lap', 'law', 'lay', 'led', 'leg', 'let', 'lid', 'lie', 'lip', 'lit', 'log', 'lot', 'low', 'mad', 'man', 'map', 'mat', 'may', 'men', 'met', 'mix', 'mob', 'mom', 'mop', 'mud', 'mug', 'mum', 'nap', 'net', 'new', 'nib', 'nil', 'nip', 'nit', 'nob', 'nod', 'nor', 'not', 'now', 'nun', 'nut', 'oar', 'oat', 'odd', 'off', 'oil', 'old', 'one', 'orb', 'ore', 'our', 'out', 'owe', 'owl', 'own', 'pad', 'pal', 'pan', 'pap', 'par', 'pat', 'paw', 'pay', 'pea', 'peg', 'pen', 'pep', 'per', 'pet', 'pie', 'pig', 'pin', 'pip', 'pit', 'ply', 'pod', 'pop', 'pot', 'pro', 'pry', 'pub', 'pug', 'pun', 'pup', 'pus', 'put', 'rad', 'rag', 'ram', 'ran', 'rap', 'rat', 'raw', 'ray', 'red', 'rep', 'rib', 'rid', 'rig', 'rim', 'rip', 'rob', 'rod', 'rot', 'row', 'rub', 'rue', 'rug', 'rum', 'run', 'rye', 'sad', 'sag', 'sap', 'sat', 'saw', 'say', 'sea', 'see', 'set', 'sew', 'sex', 'she', 'shy', 'sin', 'sip', 'sir', 'sit', 'six', 'ski', 'sky', 'sly', 'sob', 'sod', 'son', 'sow', 'soy', 'spa', 'spy', 'sty', 'sub', 'sue', 'sum', 'sun', 'sup', 'tab', 'tad', 'tag', 'tan', 'tap', 'tar', 'tat', 'tax', 'tea', 'ted', 'tee', 'ten', 'the', 'thy', 'tic', 'tie', 'tin', 'tip', 'toe', 'tog', 'tom', 'ton', 'too', 'top', 'tor', 'tow', 'toy', 'try', 'tub', 'tug', 'tun', 'two', 'use', 'van', 'vat', 'vet', 'via', 'vie', 'vow', 'wad', 'wag', 'wan', 'war', 'was', 'wat', 'wax', 'way', 'web', 'wed', 'wee', 'wet', 'who', 'why', 'wig', 'win', 'wit', 'woe', 'won', 'woo', 'wow', 'wry', 'yak', 'yam', 'yap', 'yaw', 'yay', 'yea', 'yen', 'yep', 'yes', 'yet', 'you', 'zip', 'zoo',
    'word', 'game', 'play', 'code', 'grid', 'cell', 'find', 'make', 'list', 'test', 'time', 'fast', 'slow', 'good', 'best', 'more', 'less', 'path', 'move', 'left', 'down', 'push', 'pull', 'fire', 'work', 'love', 'hate', 'live', 'dead', 'star', 'moon', 'blue', 'pink', 'gold', 'five', 'four', 'tree', 'many', 'long', 'short', 'letter', 'puzzle', 'search', 'create', 'player', 'action', 'reward', 'visual', 'render', 'python', 'expert', 'design', 'arcade', 'gamer', 'score', 'point', 'timer', 'clock', 'reset', 'frame', 'space', 'shift', 'bonus', 'valid', 'flash', 'bright', 'clean', 'style', 'color', 'select', 'submit', 'winner', 'loser'
}
VOWELS = "AEIOU"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the cursor. Press space to select adjacent letters. Press shift to submit your word."
    game_description = "Race against the clock to find and form words from the letter grid. Score 500 points to win before time runs out!"
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.grid_size = 8
        self.cell_size = 40
        self.grid_pixels = self.grid_size * self.cell_size
        self.grid_offset_x = (self.width - self.grid_pixels) // 2
        self.grid_offset_y = (self.height - self.grid_pixels) // 2 + 20

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_grid = pygame.font.Font(None, 36)
        self.font_ui = pygame.font.Font(None, 32)
        self.font_feedback = pygame.font.Font(None, 28)
        self.font_gameover = pygame.font.Font(None, 72)

        self.COLOR_BG = (15, 23, 42)
        self.COLOR_GRID = (30, 41, 59)
        self.COLOR_LETTER = (226, 232, 240)
        self.COLOR_CURSOR = (250, 204, 21, 100)
        self.COLOR_SELECT = (34, 197, 94)
        self.COLOR_UI = (241, 245, 249)
        self.COLOR_VALID = (255, 255, 255)
        self.COLOR_INVALID = (239, 68, 68)

        self.max_steps = 1800 # 60 seconds at 30fps
        self.win_score = 500
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.current_word = ""
        self.timer = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_outcome = ""
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.word_feedback = None

        self.reset()
        self.validate_implementation()

    def _get_random_letter(self):
        if self.np_random.random() < 0.4: # 40% chance of vowel
            return self.np_random.choice(list(VOWELS))
        else:
            return self.np_random.choice(list(CONSONANTS))

    def _generate_grid(self):
        grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Guarantee at least two 3-letter words
        guaranteed_words = self.np_random.choice([w for w in WORD_LIST if len(w) == 3], 2, replace=False)
        
        for word in guaranteed_words:
            placed = False
            for _ in range(20): # 20 attempts to place a word
                path = []
                start_x, start_y = self.np_random.integers(0, self.grid_size, size=2)
                
                if grid[start_y][start_x] != '':
                    continue

                path.append([start_x, start_y])
                curr_x, curr_y = start_x, start_y

                possible = True
                for _ in range(len(word) - 1):
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = curr_x + dx, curr_y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and grid[ny][nx] == '' and [nx, ny] not in path:
                            neighbors.append([nx, ny])
                    
                    if not neighbors:
                        possible = False
                        break
                    
                    curr_x, curr_y = self.np_random.choice(neighbors)
                    path.append([curr_x, curr_y])

                if possible:
                    for i, (px, py) in enumerate(path):
                        grid[py][px] = word[i].upper()
                    placed = True
                    break
            
            if not placed: # Failsafe if placement is too hard
                for char in word:
                    while True:
                        rx, ry = self.np_random.integers(0, self.grid_size, size=2)
                        if grid[ry][rx] == '':
                            grid[ry][rx] = char.upper()
                            break

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] == '':
                    grid[r][c] = self._get_random_letter()
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = 60.0
        self.game_over = False
        self.game_outcome = ""
        
        self.grid = self._generate_grid()
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        
        self.selected_path = []
        self.current_word = ""
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        self.word_feedback = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.grid_size - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.grid_size - 1, self.cursor_pos[0] + 1)

        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            if list(self.cursor_pos) not in self.selected_path:
                is_adjacent = False
                if not self.selected_path:
                    is_adjacent = True
                else:
                    last_x, last_y = self.selected_path[-1]
                    if abs(x - last_x) <= 1 and abs(y - last_y) <= 1 and (x, y) != (last_x, last_y):
                        is_adjacent = True
                
                if is_adjacent and len(self.current_word) < 8:
                    self.selected_path.append(list(self.cursor_pos))
                    self.current_word += self.grid[y][x]
                    reward += 0.1
                    # sfx: letter select sound

        if shift_held and not self.prev_shift_held:
            if self.current_word:
                is_valid = len(self.current_word) >= 3 and self.current_word.lower() in WORD_LIST
                
                if is_valid:
                    word_len = len(self.current_word)
                    word_score = word_len * 10
                    self.score += word_score
                    reward += word_score
                    self.word_feedback = ("VALID!", self.COLOR_VALID, 30)
                    # sfx: valid word sound
                    
                    for (px, py) in self.selected_path:
                        self.grid[py][px] = self._get_random_letter()
                        # Create particles
                        for _ in range(10):
                            self.particles.append(self._create_particle(px, py))
                else:
                    reward -= 1.0
                    self.word_feedback = ("INVALID", self.COLOR_INVALID, 30)
                    # sfx: invalid word sound

                self.current_word = ""
                self.selected_path = []

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Update Game State ---
        self.timer = max(0, self.timer - 1.0 / 30.0)
        self.steps += 1
        
        self._update_particles()
        if self.word_feedback:
            _, _, lifetime = self.word_feedback
            if lifetime > 0:
                self.word_feedback = (self.word_feedback[0], self.word_feedback[1], lifetime - 1)
            else:
                self.word_feedback = None

        # --- Check Termination ---
        terminated = False
        if self.score >= self.win_score:
            terminated = True
            self.game_over = True
            self.game_outcome = "YOU WIN!"
            reward += 100
        elif self.timer <= 0:
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME'S UP!"
            reward -= 100
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME'S UP!"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle(self, grid_x, grid_y):
        center_x = self.grid_offset_x + grid_x * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + grid_y * self.cell_size + self.cell_size // 2
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 2 + 1
        return {
            "x": center_x, "y": center_y,
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "lifetime": self.np_random.integers(20, 40),
            "color": random.choice([(250, 204, 21), (234, 179, 8), (217, 119, 6)])
        }

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifetime"] -= 1
            if p["lifetime"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and letters
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                is_selected = [c, r] in self.selected_path
                if is_selected:
                    pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, border_radius=4)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=4)
                
                letter_surf = self.font_grid.render(self.grid[r][c], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=rect.center)
                self.screen.blit(letter_surf, letter_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.gfxdraw.box(self.screen, cursor_rect, self.COLOR_CURSOR)

        # Draw particles
        for p in self.particles:
            radius = int(p["lifetime"] / 6)
            if radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), radius, p["color"])
                pygame.gfxdraw.aacircle(self.screen, int(p["x"]), int(p["y"]), radius, p["color"])

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (20, 10))

        # Timer
        timer_text = f"TIME: {max(0, math.ceil(self.timer))}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        timer_rect = timer_surf.get_rect(topright=(self.width - 20, 10))
        self.screen.blit(timer_surf, timer_rect)
        
        # Current Word Display
        word_bg_rect = pygame.Rect(0, self.height - 40, self.width, 40)
        pygame.draw.rect(self.screen, self.COLOR_GRID, word_bg_rect)
        
        display_word = self.current_word if self.current_word else " "
        word_surf = self.font_ui.render(display_word, True, self.COLOR_LETTER)
        word_rect = word_surf.get_rect(center=word_bg_rect.center)
        self.screen.blit(word_surf, word_rect)
        
        # Word Feedback
        if self.word_feedback:
            text, color, lifetime = self.word_feedback
            alpha = int(255 * (lifetime / 30))
            feedback_surf = self.font_feedback.render(text, True, color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(centerx=self.width/2, bottom=word_bg_rect.top - 5)
            self.screen.blit(feedback_surf, feedback_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            gameover_surf = self.font_gameover.render(self.game_outcome, True, self.COLOR_UI)
            gameover_rect = gameover_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(gameover_surf, gameover_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "current_word": self.current_word
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This requires a window to capture key events.
    # The environment itself is headless, but this wrapper creates a display.
    
    try:
        import os
        # The environment is headless, but for human play, we need a display.
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
            del os.environ["SDL_VIDEODRIVER"]

        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Word Finder")
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + "="*30)
        print("MANUAL PLAY INSTRUCTIONS")
        print(env.user_guide)
        print("="*30 + "\n")

        while not terminated:
            # Action defaults
            movement = 0
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            # Render the observation from the environment to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Match the environment's internal FPS
            
            if terminated:
                print("Game Over!")
                print(f"Final Score: {info['score']}")
                pygame.time.wait(3000) # Pause to show final screen
                
    except ImportError:
        print("Pygame not installed, cannot run manual play example.")
    except Exception as e:
        print(f"An error occurred during manual play: {e}")
    finally:
        env.close()