
# Generated: 2025-08-28T06:02:16.182941
# Source Brief: brief_02789.md
# Brief Index: 2789

        
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
        "Controls: Use arrow keys to move the selector. Press space to select a gem group."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Select groups of 2 or more adjacent, same-colored gems to collect them. "
        "Collect 20 gems within 15 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.grid_width = 8
        self.grid_height = 8
        self.num_gem_types = 5
        self.moves_limit = 15
        self.win_score = 20
        self.min_match_size = 2

        # --- Visuals ---
        self.cell_size = 40
        self.grid_margin_x = (self.screen_width - self.grid_width * self.cell_size) // 2
        self.grid_margin_y = (self.screen_height - self.grid_height * self.cell_size) // 2

        self.colors = {
            "bg": (20, 30, 40),
            "grid_line": (50, 60, 70),
            "cursor": (255, 255, 255),
            "text": (230, 230, 230),
            "win_text": (100, 255, 100),
            "lose_text": (255, 100, 100),
            "overlay": (0, 0, 0, 180),
        }
        self.gem_colors = {
            1: (255, 80, 80),   # Red
            2: (80, 255, 80),   # Green
            3: (80, 150, 255),  # Blue
            4: (255, 255, 80),  # Yellow
            5: (200, 80, 255),  # Purple
        }

        self.font_main = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win_status = False
        self.steps = 0
        self.last_match_info = {}

        # --- Initialization ---
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.moves_limit
        self.game_over = False
        self.win_status = False
        self.steps = 0
        self.cursor_pos = [self.grid_width // 2, self.grid_height // 2]
        self._generate_grid()
        self.last_match_info = {}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_press, _ = action
        reward = 0
        self.steps += 1
        self.last_match_info = {}

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.grid_width
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.grid_height

        # --- Handle Action (Space Press) ---
        if space_press == 1:
            self.moves_left -= 1
            
            match_group = self._find_match_group(self.cursor_pos[1], self.cursor_pos[0])
            
            if len(match_group) >= self.min_match_size:
                # Sound: gem_match.wav
                num_matched = len(match_group)
                self.score += num_matched
                reward += num_matched  # +1 per gem

                if num_matched >= 4:
                    # Sound: bonus_match.wav
                    reward += 5  # Bonus for large match

                gem_type = self.grid[self.cursor_pos[1], self.cursor_pos[0]]
                self.last_match_info = {
                    "positions": match_group,
                    "color": self.gem_colors[gem_type],
                    "size": num_matched,
                    "frame": self.steps
                }

                for r, c in match_group:
                    self.grid[r, c] = 0  # Mark as empty

                self._apply_gravity_and_refill()
            else:
                # Sound: invalid_move.wav
                reward -= 0.5 # Small penalty for a wasted move

        # --- Check Termination ---
        terminated = False
        if self.score >= self.win_score:
            self.game_over = True
            self.win_status = True
            terminated = True
            reward += 50
            # Sound: win_game.wav
        elif self.moves_left <= 0:
            self.game_over = True
            self.win_status = False
            terminated = True
            reward -= 50
            # Sound: lose_game.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.num_gem_types + 1, size=(self.grid_height, self.grid_width))
            if self._has_possible_moves():
                break

    def _has_possible_moves(self):
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                gem_type = self.grid[r, c]
                if gem_type == 0: continue
                # Check right
                if c < self.grid_width - 1 and self.grid[r, c + 1] == gem_type:
                    return True
                # Check down
                if r < self.grid_height - 1 and self.grid[r + 1, c] == gem_type:
                    return True
        return False

    def _find_match_group(self, start_r, start_c):
        if self.grid[start_r, start_c] == 0:
            return []

        target_type = self.grid[start_r, start_c]
        q = [(start_r, start_c)]
        visited = set(q)
        match_group = []

        while q:
            r, c = q.pop(0)
            match_group.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_type:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return match_group

    def _apply_gravity_and_refill(self):
        for c in range(self.grid_width):
            write_idx = self.grid_height - 1
            for r in range(self.grid_height - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if write_idx != r:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_idx -= 1
            
            for r in range(write_idx, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.num_gem_types + 1)

    def _get_observation(self):
        self.screen.fill(self.colors["bg"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid and Gems ---
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                x = self.grid_margin_x + c * self.cell_size
                y = self.grid_margin_y + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                pygame.draw.rect(self.screen, self.colors["grid_line"], rect, 1)

                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(x, y, gem_type)
        
        # --- Draw Match Effects ---
        if self.last_match_info:
            color = self.last_match_info["color"]
            for r, c in self.last_match_info["positions"]:
                x = self.grid_margin_x + c * self.cell_size + self.cell_size // 2
                y = self.grid_margin_y + r * self.cell_size + self.cell_size // 2
                for i in range(5):
                    angle = i * (2 * math.pi / 5) + self.steps * 0.5
                    end_x = x + math.cos(angle) * 15
                    end_y = y + math.sin(angle) * 15
                    pygame.draw.line(self.screen, (255, 255, 200), (x, y), (int(end_x), int(end_y)), 2)

        # --- Draw Cursor ---
        cursor_x = self.grid_margin_x + self.cursor_pos[0] * self.cell_size
        cursor_y = self.grid_margin_y + self.cursor_pos[1] * self.cell_size
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        alpha = 100 + pulse * 155
        cursor_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.colors["cursor"], alpha), (0, 0, self.cell_size, self.cell_size), 4, border_radius=8)
        self.screen.blit(cursor_surface, (cursor_x, cursor_y))

    def _draw_gem(self, x, y, gem_type):
        color = self.gem_colors[gem_type]
        darker_color = tuple(max(0, val - 50) for val in color)
        rect = pygame.Rect(x + 4, y + 4, self.cell_size - 8, self.cell_size - 8)
        
        pygame.gfxdraw.box(self.screen, rect, color)
        pygame.gfxdraw.rectangle(self.screen, rect, darker_color)
        
        highlight_rect = pygame.Rect(x + 6, y + 6, self.cell_size - 20, self.cell_size - 20)
        pygame.gfxdraw.box(self.screen, highlight_rect, (255, 255, 255, 50))

    def _render_ui(self):
        score_text = f"Gems: {self.score} / {self.win_score}"
        moves_text = f"Moves: {self.moves_left}"
        self._draw_text(score_text, (20, 20), self.font_main, self.colors["text"])
        
        moves_surf = self.font_main.render(moves_text, True, self.colors["text"])
        self._draw_text(moves_text, (self.screen_width - moves_surf.get_width() - 20, 20), self.font_main, self.colors["text"])

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill(self.colors["overlay"])
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("YOU WIN!", self.colors["win_text"]) if self.win_status else ("GAME OVER", self.colors["lose_text"])
            
            self._draw_text(msg, (self.screen_width // 2, self.screen_height // 2 - 20), self.font_title, color, center=True)
            self._draw_text("Resetting...", (self.screen_width // 2, self.screen_height // 2 + 30), self.font_main, self.colors["text"], center=True)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Collector")
    
    running = True
    clock = pygame.time.Clock()
    
    last_action_time = 0
    action_delay = 150  # milliseconds

    while running:
        now = pygame.time.get_ticks()
        action = [0, 0, 0] # Default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if now > last_action_time + action_delay:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        last_action_time = now
                    elif event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
                    
                    if any(a != 0 for a in action):
                        obs, reward, terminated, truncated, info = env.step(action)
                        last_action_time = now
                        if terminated:
                            print(f"Game Over! Final Score: {info['score']}, Moves Left: {info['moves_left']}")
                            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                            screen.blit(surf, (0, 0))
                            pygame.display.flip()
                            pygame.time.wait(2000)
                            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(60)

    env.close()