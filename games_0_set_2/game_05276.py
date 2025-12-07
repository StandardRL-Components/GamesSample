
# Generated: 2025-08-28T04:30:15.010575
# Source Brief: brief_05276.md
# Brief Index: 5276

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import string
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to place the current tile on the grid."
    )

    game_description = (
        "Strategically place letter tiles on an isometric grid to create words and reach a target score before running out of tiles."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.TARGET_SCORE = 100
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_TILE = (30, 144, 255) # Dodger Blue
        self.COLOR_TILE_TEXT = (255, 255, 255)
        self.COLOR_CURSOR_VALID = (0, 255, 127, 100) # Spring Green (alpha)
        self.COLOR_CURSOR_INVALID = (255, 69, 0, 100) # Orangered (alpha)
        self.COLOR_HIGHLIGHT = (255, 215, 0) # Gold
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (40, 50, 60)
        self.COLOR_UI_TILE_BG = (50, 65, 80)

        # --- Isometric Projection ---
        self.TILE_W = 28
        self.TILE_H = 14
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.tile_font = pygame.font.SysFont("Arial", 14, bold=True)
        self.ui_font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.ui_font_small = pygame.font.SysFont("Arial", 16)
        
        # --- Game State Initialization ---
        self.grid = None
        self.tile_bag = None
        self.current_tile = None
        self.score = 0
        self.steps = 0
        self.cursor_pos = (0, 0)
        self.game_over = False
        self.last_space_held = False
        self.newly_formed_coords = []
        self.highlight_timer = 0
        self.last_word_score = 0
        self.last_action_feedback = ""
        self.feedback_timer = 0

        self.LETTER_VALUES = {letter: i + 1 for i, letter in enumerate(string.ascii_uppercase)}
        self.TILE_FREQUENCIES = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9, 
            'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6, 
            'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1
        }
        
        self.reset()
        self.validate_implementation()

    def _create_tile_bag(self):
        bag = []
        for letter, count in self.TILE_FREQUENCIES.items():
            bag.extend([letter] * count)
        self.np_random.shuffle(bag)
        return bag

    def _draw_tile(self):
        if self.tile_bag:
            return self.tile_bag.pop()
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.tile_bag = self._create_tile_bag()
        self.current_tile = self._draw_tile()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.last_space_held = False
        self.newly_formed_coords = []
        self.highlight_timer = 0
        self.last_word_score = 0
        self.last_action_feedback = ""
        self.feedback_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self._update_timers()

        # Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos = (
                (self.cursor_pos[0] + dx) % self.GRID_SIZE,
                (self.cursor_pos[1] + dy) % self.GRID_SIZE
            )

        # Handle tile placement on space press (rising edge)
        if space_held and not self.last_space_held:
            reward += self._attempt_placement()
        self.last_space_held = space_held

        # Check for termination conditions
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.last_action_feedback = "TARGET SCORE REACHED!"
            self.feedback_timer = 90
        elif self.current_tile is None:
            reward -= 50
            terminated = True
            self.last_action_feedback = "OUT OF TILES!"
            self.feedback_timer = 90
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_timers(self):
        if self.highlight_timer > 0:
            self.highlight_timer -= 1
            if self.highlight_timer == 0:
                self.newly_formed_coords = []
                self.last_word_score = 0
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            if self.feedback_timer == 0:
                self.last_action_feedback = ""

    def _attempt_placement(self):
        x, y = self.cursor_pos
        if self.grid[y][x] == '':
            # Place tile
            self.grid[y][x] = self.current_tile
            # Find words
            new_words, word_coords = self._check_for_words(x, y)
            
            if new_words:
                word_score = sum(self._calculate_word_score(word) for word in new_words)
                self.score += word_score
                self.last_word_score = word_score
                self.newly_formed_coords = list(set(coord for coords in word_coords for coord in coords))
                self.highlight_timer = 30 # Frames to highlight
                self.last_action_feedback = f"+{word_score} POINTS!"
                self.feedback_timer = 60
                # SFX: place_tile_success.wav, score_word.wav
            else:
                # Placed in isolation, not forming a word
                self.last_action_feedback = "Placed tile."
                self.feedback_timer = 60
                # SFX: place_tile_success.wav

            self.current_tile = self._draw_tile()
            return 0.1 + self.last_word_score # Reward for placement + word score
        else:
            self.last_action_feedback = "CAN'T PLACE HERE!"
            self.feedback_timer = 60
            # SFX: error.wav
            return -0.1 # Small penalty for invalid move

    def _check_for_words(self, x, y):
        words = []
        all_coords = []
        
        # Check horizontal
        start_x = x
        while start_x > 0 and self.grid[y][start_x - 1] != '':
            start_x -= 1
        
        word = ""
        coords = []
        for i in range(start_x, self.GRID_SIZE):
            if self.grid[y][i] != '':
                word += self.grid[y][i]
                coords.append((i, y))
            else:
                break
        
        if len(word) > 1:
            words.append(word)
            all_coords.append(coords)

        # Check vertical
        start_y = y
        while start_y > 0 and self.grid[start_y - 1][x] != '':
            start_y -= 1
            
        word = ""
        coords = []
        for i in range(start_y, self.GRID_SIZE):
            if self.grid[i][x] != '':
                word += self.grid[i][x]
                coords.append((x, i))
            else:
                break
        
        if len(word) > 1:
            words.append(word)
            all_coords.append(coords)
            
        return words, all_coords

    def _calculate_word_score(self, word):
        return sum(self.LETTER_VALUES.get(char, 0) for char in word)

    def _to_iso(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_W // 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_H // 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and tiles from back to front
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._render_grid_cell(c, r)
        
        # Draw cursor
        self._render_cursor()

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] != '':
                    self._render_tile(c, r, self.grid[r][c])

    def _render_grid_cell(self, x, y):
        screen_x, screen_y = self._to_iso(x, y)
        points = [
            (screen_x, screen_y - self.TILE_H // 2),
            (screen_x + self.TILE_W // 2, screen_y),
            (screen_x, screen_y + self.TILE_H // 2),
            (screen_x - self.TILE_W // 2, screen_y),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_cursor(self):
        x, y = self.cursor_pos
        screen_x, screen_y = self._to_iso(x, y)
        is_valid = self.grid[y][x] == ''
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        points = [
            (screen_x, screen_y - self.TILE_H // 2),
            (screen_x + self.TILE_W // 2, screen_y),
            (screen_x, screen_y + self.TILE_H // 2),
            (screen_x - self.TILE_W // 2, screen_y),
        ]
        
        # Create a temporary surface for the transparent cursor
        temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surface, points, color)
        self.screen.blit(temp_surface, (0, 0))

    def _render_tile(self, x, y, letter):
        screen_x, screen_y = self._to_iso(x, y)
        is_highlighted = self.highlight_timer > 0 and (x, y) in self.newly_formed_coords
        
        base_color = self.COLOR_HIGHLIGHT if is_highlighted else self.COLOR_TILE
        top_color = tuple(min(255, c + 30) for c in base_color)
        side_color = tuple(max(0, c - 30) for c in base_color)

        # Draw 3D-effect tile
        top_points = [
            (screen_x, screen_y - self.TILE_H // 2),
            (screen_x + self.TILE_W // 2, screen_y),
            (screen_x, screen_y + self.TILE_H // 2),
            (screen_x - self.TILE_W // 2, screen_y),
        ]
        left_side_points = [
            (screen_x - self.TILE_W // 2, screen_y),
            (screen_x, screen_y + self.TILE_H // 2),
            (screen_x, screen_y + self.TILE_H // 2 + 5),
            (screen_x - self.TILE_W // 2, screen_y + 5),
        ]
        right_side_points = [
            (screen_x + self.TILE_W // 2, screen_y),
            (screen_x, screen_y + self.TILE_H // 2),
            (screen_x, screen_y + self.TILE_H // 2 + 5),
            (screen_x + self.TILE_W // 2, screen_y + 5),
        ]
        
        pygame.draw.polygon(self.screen, side_color, left_side_points)
        pygame.draw.polygon(self.screen, side_color, right_side_points)
        pygame.draw.polygon(self.screen, top_color, top_points)

        # Draw letter
        text_surf = self.tile_font.render(letter, True, self.COLOR_TILE_TEXT)
        text_rect = text_surf.get_rect(center=(screen_x, screen_y))
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.ui_font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        score_label = self.ui_font_small.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_label, (20, 10))
        self.screen.blit(score_text, (20, 30))
        
        # --- Target Score ---
        target_text = self.ui_font_small.render(f"TARGET: {self.TARGET_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (20, 60))

        # --- Current Tile Display ---
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.WIDTH - 130, self.HEIGHT - 70, 110, 60), border_radius=5)
        label_surf = self.ui_font_small.render("TILES LEFT", True, self.COLOR_UI_TEXT)
        self.screen.blit(label_surf, (self.WIDTH - 118, self.HEIGHT - 65))
        
        # Tile stack effect
        if self.tile_bag:
            for i in range(min(5, len(self.tile_bag)//5)):
                pygame.draw.rect(self.screen, self.COLOR_UI_TILE_BG, (self.WIDTH - 90 + i, self.HEIGHT - 45 - i, 40, 30), border_radius=3)
        
        tiles_left_text = self.ui_font_large.render(str(len(self.tile_bag)), True, self.COLOR_UI_TEXT)
        self.screen.blit(tiles_left_text, (self.WIDTH - 85, self.HEIGHT - 42))

        # --- Next Tile Display ---
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (20, self.HEIGHT - 70, 60, 60), border_radius=5)
        label_surf = self.ui_font_small.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(label_surf, (33, self.HEIGHT - 65))
        
        if self.current_tile:
            pygame.draw.rect(self.screen, self.COLOR_UI_TILE_BG, (30, self.HEIGHT - 45, 40, 30), border_radius=3)
            tile_text = self.ui_font_large.render(self.current_tile, True, self.COLOR_UI_TEXT)
            self.screen.blit(tile_text, (39, self.HEIGHT - 45))

        # --- Feedback Text ---
        if self.feedback_timer > 0:
            alpha = min(255, self.feedback_timer * 5)
            feedback_surf = self.ui_font_large.render(self.last_action_feedback, True, self.COLOR_HIGHLIGHT)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.WIDTH // 2, 40))
            self.screen.blit(feedback_surf, feedback_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tiles_remaining": len(self.tile_bag) + (1 if self.current_tile else 0),
            "cursor_pos": self.cursor_pos
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Word Grid")
    
    running = True
    clock = pygame.time.Clock()

    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the env to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30)

    env.close()