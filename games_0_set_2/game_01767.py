
# Generated: 2025-08-28T02:39:25.526605
# Source Brief: brief_01767.md
# Brief Index: 1767

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Arrow keys to move the worm. Press Space to submit the collected word."

    # Must be a short, user-facing description of the game:
    game_description = "A puzzle game where you guide a worm on a grid to spell words against a move limit."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_HEIGHT = 80
    GRID_WIDTH, GRID_HEIGHT = 16, 8
    CELL_SIZE = 40
    GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    
    MAX_MOVES = 50
    MAX_STEPS = 1000
    WORDS_PER_LEVEL = 5

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 80)
    COLOR_LETTER = (255, 255, 255)
    COLOR_WORM = (50, 255, 50)
    COLOR_WORM_GLOW = (50, 255, 50, 50)
    COLOR_UI_BG = (5, 10, 20)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_FLASH_SUCCESS = (0, 255, 0, 100)
    COLOR_FLASH_FAIL = (255, 0, 0, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("consolas", 32)
            self.font_medium = pygame.font.SysFont("consolas", 24)
            self.font_small = pygame.font.SysFont("consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        self._generate_word_list()
        
        # Initialize state variables
        self.worm_pos = None
        self.grid = None
        self.current_word = None
        self.score = None
        self.moves_remaining = None
        self.level = None
        self.word_length = None
        self.words_spelled_this_level = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.flash_alpha = None
        self.flash_color = None
        self.valid_words = None
        self.valid_prefixes = None
        
        self.reset()
        self.validate_implementation()

    def _generate_word_list(self):
        self.WORDS = {
            3: ["ART", "EAT", "TEA", "RAT", "TAR", "ATE", "EAR", "ERA", "ARE", "CAT", "DOG", "PIG", "RUN", "SUN"],
            4: ["WORD", "GAME", "PLAY", "CODE", "GRID", "WORM", "GLOW", "FOUR", "FIVE", "STAR", "MOON", "RAIN"],
            5: ["AGENT", "SOLVE", "BRAIN", "GYMNASIUM", "LEVEL", "SCORE", "LIGHT", "GREEN", "SPACE", "VALID"],
            6: ["PYTHON", "PUZZLE", "ACTION", "REWARD", "VISUAL", "EFFECT", "LETTER", "FORMAT", "RANDOM"],
            7: ["CONTROL", "QUALITY", "EXPERT", "MINIMAL", "DISPLAY", "SUCCESS", "FAILURE"],
            8: ["GYMNASIUM", "ADVANCE", "FEEDBACK", "RENDER", "COMPLETE"]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.moves_remaining = self.MAX_MOVES
        self.words_spelled_this_level = 0
        self.current_word = ""
        self.particles = []
        self.flash_alpha = 0
        self.flash_color = (0,0,0)

        self._update_level_specifics()
        self._generate_grid()
        
        # Place worm in an empty spot
        empty_cells = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] == '':
                    empty_cells.append((c, r))
        if not empty_cells: # Should not happen with our generation
            self.worm_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        else:
            self.worm_pos = self.np_random.choice(empty_cells)
            self.worm_pos = (self.worm_pos[0], self.worm_pos[1])

        return self._get_observation(), self._get_info()

    def _update_level_specifics(self):
        self.word_length = min(max(self.WORDS.keys()), self.level + 2)
        self.valid_words = self.WORDS.get(self.word_length, [])
        if not self.valid_words:
            self.valid_words = self.WORDS[max(self.WORDS.keys())]

        self.valid_prefixes = {w[:i] for w in self.valid_words for i in range(1, len(w) + 1)}

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        num_target_words = self.np_random.integers(3, 6)
        target_words = self.np_random.choice(self.valid_words, size=num_target_words, replace=False)
        
        required_letters = list("".join(target_words))
        
        all_coords = [(c, r) for c in range(self.GRID_WIDTH) for r in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)
        
        # Place required letters
        for i, letter in enumerate(required_letters):
            if i < len(all_coords):
                c, r = all_coords[i]
                self.grid[r][c] = letter
        
        # Fill remaining with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(len(required_letters), len(all_coords)):
            c, r = all_coords[i]
            self.grid[r][c] = self.np_random.choice(list(alphabet))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        self.steps += 1
        
        # --- Handle Movement ---
        if movement != 0:
            self.moves_remaining -= 1
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            new_x = self.worm_pos[0] + dx
            new_y = self.worm_pos[1] + dy

            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT:
                self.worm_pos = (new_x, new_y)
                letter = self.grid[new_y][new_x]
                if letter != '':
                    # Collect letter
                    self.current_word += letter
                    self.grid[new_y][new_x] = ''
                    # Sound placeholder: sfx_collect_letter.wav
                    
                    # Spawn collection particles
                    start_pos = (self.worm_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, 
                                 self.worm_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2 + self.UI_HEIGHT)
                    for _ in range(5):
                        self.particles.append(self._create_particle(start_pos, self.COLOR_LETTER))

                    # Prefix reward
                    if self.current_word in self.valid_prefixes:
                        reward += 1
                    else:
                        reward += -0.1

        # --- Handle Word Submission ---
        if space_pressed and self.current_word:
            if self.current_word in self.valid_words:
                # Correct word
                reward += 10
                self.score += len(self.current_word) * 10
                self.words_spelled_this_level += 1
                self.flash_color = self.COLOR_FLASH_SUCCESS
                self.flash_alpha = 255
                # Sound placeholder: sfx_word_success.wav

                # Spawn success particles
                for _ in range(50):
                     self.particles.append(self._create_particle((self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), self.COLOR_WORM, burst=True))
                
                if self.words_spelled_this_level >= self.WORDS_PER_LEVEL:
                    # Level up
                    reward += 50
                    self.score += 100
                    self.level += 1
                    self.words_spelled_this_level = 0
                    self._update_level_specifics()
                    # Sound placeholder: sfx_level_up.wav

                self._generate_grid()
                self.current_word = ""

            else:
                # Incorrect word
                reward -= 1
                self.score = max(0, self.score - 10)
                self.flash_color = self.COLOR_FLASH_FAIL
                self.flash_alpha = 255
                self.current_word = ""
                # Sound placeholder: sfx_word_fail.wav
        
        # --- Update Game State ---
        self._update_particles()
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 25)

        terminated = self.moves_remaining <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particle(self, pos, color, burst=False):
        if burst:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        else:
            vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1)]
        return {
            'pos': list(pos),
            'vel': vel,
            'life': self.np_random.integers(20, 40),
            'color': color,
            'size': self.np_random.uniform(2, 5)
        }

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        game_surface = self.screen.subsurface((0, self.UI_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.UI_HEIGHT))
        
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = r * self.CELL_SIZE
            pygame.draw.line(game_surface, self.COLOR_GRID, (0, y), (self.GAME_AREA_WIDTH, y), 1)
        for c in range(self.GRID_WIDTH + 1):
            x = c * self.CELL_SIZE
            pygame.draw.line(game_surface, self.COLOR_GRID, (x, 0), (x, self.GAME_AREA_HEIGHT), 1)

        # Draw letters
        letter_font = pygame.font.Font(None, self.CELL_SIZE)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                letter = self.grid[r][c]
                if letter:
                    text_surf = letter_font.render(letter, True, self.COLOR_LETTER)
                    text_rect = text_surf.get_rect(center=(c * self.CELL_SIZE + self.CELL_SIZE / 2, r * self.CELL_SIZE + self.CELL_SIZE / 2))
                    game_surface.blit(text_surf, text_rect)

        # Draw worm
        wx, wy = self.worm_pos
        center_x = int(wx * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(wy * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # Glow
        glow_radius = int(self.CELL_SIZE * 0.6)
        pygame.gfxdraw.filled_circle(game_surface, center_x, center_y, glow_radius, self.COLOR_WORM_GLOW)
        pygame.gfxdraw.aacircle(game_surface, center_x, center_y, glow_radius, self.COLOR_WORM_GLOW)
        
        # Body
        body_radius = int(self.CELL_SIZE * 0.35)
        pygame.gfxdraw.filled_circle(game_surface, center_x, center_y, body_radius, self.COLOR_WORM)
        pygame.gfxdraw.aacircle(game_surface, center_x, center_y, body_radius, self.COLOR_WORM)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.UI_HEIGHT))
            size = int(p['size'] * (p['life'] / 40.0))
            if size > 0:
                pygame.draw.circle(game_surface, p['color'], pos, size)

        # Draw flash effect
        if self.flash_alpha > 0:
            flash_surf = pygame.Surface(game_surface.get_size(), pygame.SRCALPHA)
            color = self.flash_color[:3] + (int(self.flash_alpha * (self.flash_color[3]/255.0)),)
            flash_surf.fill(color)
            game_surface.blit(flash_surf, (0, 0))

    def _render_ui(self):
        ui_surface = self.screen.subsurface((0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        ui_surface.fill(self.COLOR_UI_BG)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1), 2)

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        ui_surface.blit(score_surf, (15, 10))

        # Moves
        moves_text = f"MOVES: {self.moves_remaining}"
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_UI_TEXT)
        ui_surface.blit(moves_surf, (15, 40))

        # Level
        level_text = f"LEVEL: {self.level} (Find {self.word_length}-letter words)"
        level_surf = self.font_small.render(level_text, True, self.COLOR_UI_TEXT)
        level_rect = level_surf.get_rect(right=self.SCREEN_WIDTH - 15, top=10)
        ui_surface.blit(level_surf, level_rect)

        # Words this level
        words_text = f"WORDS: {self.words_spelled_this_level} / {self.WORDS_PER_LEVEL}"
        words_surf = self.font_small.render(words_text, True, self.COLOR_UI_TEXT)
        words_rect = words_surf.get_rect(right=self.SCREEN_WIDTH - 15, top=35)
        ui_surface.blit(words_surf, words_rect)

        # Current Word
        word_label_surf = self.font_medium.render("WORD:", True, self.COLOR_UI_TEXT)
        word_label_rect = word_label_surf.get_rect(centerx=self.SCREEN_WIDTH / 2 - 50, centery=self.UI_HEIGHT / 2)
        ui_surface.blit(word_label_surf, word_label_rect)
        
        word_surf = self.font_large.render(self.current_word, True, self.COLOR_LETTER)
        word_rect = word_surf.get_rect(midleft=(word_label_rect.right + 10, self.UI_HEIGHT / 2))
        ui_surface.blit(word_surf, word_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "current_word": self.current_word,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, you would need to use pygame.display
    # This is for testing the environment logic headless
    obs, info = env.reset()
    print("Initial state:", info)

    terminated = False
    total_reward = 0
    
    # Simulate a few random steps
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Info={info}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

    env.close()