import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:30:08.806021
# Source Brief: brief_00508.md
# Brief Index: 508
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent combines numbers on a grid to reach a target score.

    The agent controls a cursor on a 10x10 grid. It can move the cursor and select
    cells containing numbers. When two numbers are selected, they are combined if their
    sum is 50 or less. The goal is to reach a total combined score of 500 within
    the time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Combine numbers on a grid to reach a target score. Select two numbers to add them together, but their sum cannot exceed 50."
    user_guide = "Use the arrow keys (↑↓←→) to move the cursor. Press space to select/deselect a number. Select two numbers to combine them."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    WIN_SCORE = 500
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    COMBINE_LIMIT = 50
    
    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_LINES = (30, 40, 65)
    COLOR_CURSOR = (255, 255, 0, 150) # Yellow, semi-transparent
    COLOR_SELECTION = (255, 180, 0) # Bright Orange/Yellow
    COLOR_TEXT = (240, 240, 255)
    COLOR_LOW_NUM = np.array([50, 100, 255])  # Blue
    COLOR_HIGH_NUM = np.array([255, 50, 100]) # Red
    COLOR_TIMER_BAR = (50, 200, 255)
    COLOR_FAIL_FLASH = (255, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_cells = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        self.number_spawn_timer = None
        self.particles = None
        self.effects = None

        # --- Grid Layout Calculation ---
        self.grid_area_height = self.SCREEN_HEIGHT - 50
        self.cell_size = min(self.SCREEN_WIDTH // self.GRID_WIDTH, self.grid_area_height // self.GRID_HEIGHT)
        self.grid_pixel_width = self.cell_size * self.GRID_WIDTH
        self.grid_pixel_height = self.cell_size * self.GRID_HEIGHT
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height)

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_cells = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.number_spawn_timer = 0
        
        self.particles = []
        self.effects = [] # For temporary visual effects like flashes

        # Initial number spawns
        for _ in range(5):
            self._spawn_number()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._handle_selection()
        self.prev_space_held = space_held
        
        self._update_game_logic()

        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on conditions other than termination
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_game_logic(self):
        self.steps += 1
        
        # Spawn new numbers periodically
        self.number_spawn_timer += 1
        if self.number_spawn_timer > 30: # Spawn a number every second (at 30fps)
            self.number_spawn_timer = 0
            self._spawn_number()

        # Update particles and effects
        self.particles = [p for p in self.particles if p.update()]
        self.effects = [e for e in self.effects if e.update()]

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_HEIGHT
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_HEIGHT
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_WIDTH
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_WIDTH
        # sfx: cursor_move

    def _handle_selection(self):
        y, x = self.cursor_pos
        cell_coords = (y, x)
        
        if self.grid[y, x] == 0:
            return 0 # Cannot select an empty cell

        if cell_coords in self.selected_cells:
            # Deselect
            self.selected_cells.remove(cell_coords)
            # sfx: deselect
            return 0
        
        if len(self.selected_cells) < 2:
            # Select
            self.selected_cells.append(cell_coords)
            # sfx: select
        
        if len(self.selected_cells) == 2:
            # Attempt combination
            return self._attempt_combination()
        
        return 0

    def _attempt_combination(self):
        (y1, x1), (y2, x2) = self.selected_cells
        num1, num2 = self.grid[y1, x1], self.grid[y2, x2]
        
        total = num1 + num2
        
        if total <= self.COMBINE_LIMIT:
            # sfx: combine_success
            self.grid[y1, x1] = total
            self.grid[y2, x2] = 0
            self.score += total
            
            # Create particle burst effect
            center_x = self.grid_offset_x + x1 * self.cell_size + self.cell_size // 2
            center_y = self.grid_offset_y + y1 * self.cell_size + self.cell_size // 2
            self._create_particles((center_x, center_y), self._get_number_color(total))
            
            # Clear selection
            self.selected_cells = []

            # Calculate reward
            reward = 0.1 # Base reward for any combination
            if total > 25: reward += 1
            if total > 40: reward += 5
            return reward
        else:
            # sfx: combine_fail
            # Add a visual effect for failure
            self.effects.append(FailFlash(self.selected_cells, 10))
            self.selected_cells = []
            return 0

    def _spawn_number(self):
        empty_cells = np.argwhere(self.grid == 0)
        if len(empty_cells) > 0:
            y, x = random.choice(empty_cells)
            self.grid[y, x] = self.np_random.integers(1, 11) # Spawn smaller numbers

    def _check_termination(self):
        # 1. Win condition
        if self.score >= self.WIN_SCORE:
            return True
        # 2. Time ran out
        if self.steps >= self.MAX_STEPS:
            return True
        # 3. Grid is full and no moves are possible
        if np.count_nonzero(self.grid) == self.GRID_WIDTH * self.GRID_HEIGHT:
            if not self._is_move_possible():
                return True
        return False

    def _is_move_possible(self):
        """Checks if any valid combination exists on the grid."""
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                current_val = self.grid[r, c]
                if current_val == 0: continue
                # Check neighbor below
                if r + 1 < self.GRID_HEIGHT:
                    neighbor_val = self.grid[r + 1, c]
                    if neighbor_val > 0 and current_val + neighbor_val <= self.COMBINE_LIMIT:
                        return True
                # Check neighbor to the right
                if c + 1 < self.GRID_WIDTH:
                    neighbor_val = self.grid[r, c + 1]
                    if neighbor_val > 0 and current_val + neighbor_val <= self.COMBINE_LIMIT:
                        return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid()
        self._render_numbers()
        self._render_cursor()
        self._render_effects()
        self._render_particles()

    def _render_grid(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_pixel_height), 1)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_pixel_width, y), 1)

    def _get_number_color(self, value):
        """Interpolates color based on number value."""
        if value == 0:
            return self.COLOR_TEXT
        # Normalize value from 1 to COMBINE_LIMIT
        t = (value - 1) / (self.COMBINE_LIMIT - 1)
        t = max(0, min(1, t)) # Clamp t to [0, 1]
        color = self.COLOR_LOW_NUM * (1 - t) + self.COLOR_HIGH_NUM * t
        return tuple(int(c) for c in color)

    def _render_numbers(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                value = self.grid[r, c]
                if value > 0:
                    color = self._get_number_color(value)
                    rect = pygame.Rect(
                        self.grid_offset_x + c * self.cell_size,
                        self.grid_offset_y + r * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    
                    # Draw selection outline if selected
                    if (r, c) in self.selected_cells:
                        pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 4, border_radius=4)
                    
                    # Draw number text
                    text_surf = self.font_medium.render(str(value), True, color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

    def _render_cursor(self):
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + c * self.cell_size,
            self.grid_offset_y + r * self.cell_size,
            self.cell_size, self.cell_size
        )
        # Use a surface to draw the transparent rectangle
        cursor_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, self.COLOR_CURSOR, cursor_surf.get_rect(), border_radius=4)
        self.screen.blit(cursor_surf, rect.topleft)

    def _render_effects(self):
        for effect in self.effects:
            effect.draw(self)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # --- Score Display ---
        score_text = f"SCORE: {self.score} / {self.WIN_SCORE}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 25))
        self.screen.blit(score_surf, score_rect)

        # --- Timer Bar ---
        time_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        bar_width = self.SCREEN_WIDTH * time_ratio
        bar_rect = pygame.Rect(0, 0, bar_width, 5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, bar_rect)

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "VICTORY!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "selected_cells": self.selected_cells,
        }
        
    def _create_particles(self, pos, color):
        for _ in range(30):
            self.particles.append(Particle(pos, color, self.np_random))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# --- Helper classes for visual effects ---

class Particle:
    def __init__(self, pos, color, rng):
        self.pos = list(pos)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.life = rng.integers(15, 31) # frames
        self.radius = rng.uniform(2, 5)
        self.color = color

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # gravity
        self.life -= 1
        self.radius -= 0.1
        return self.life > 0 and self.radius > 0

    def draw(self, screen):
        # Use gfxdraw for anti-aliased circles
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        radius_int = int(self.radius)
        if radius_int > 0:
            pygame.gfxdraw.aacircle(screen, pos_int[0], pos_int[1], radius_int, self.color)
            pygame.gfxdraw.filled_circle(screen, pos_int[0], pos_int[1], radius_int, self.color)

class FailFlash:
    def __init__(self, cells_to_flash, duration):
        self.cells = list(cells_to_flash)
        self.duration = duration
        self.life = duration

    def update(self):
        self.life -= 1
        return self.life > 0

    def draw(self, env):
        # Fade out the flash
        alpha = int(255 * (self.life / self.duration))
        color = (*env.COLOR_FAIL_FLASH, alpha)
        
        for r, c in self.cells:
            rect = pygame.Rect(
                env.grid_offset_x + c * env.cell_size,
                env.grid_offset_y + r * env.cell_size,
                env.cell_size, env.cell_size
            )
            flash_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            flash_surf.fill(color)
            env.screen.blit(flash_surf, rect.topleft)

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for manual testing and will not be run by the evaluation system.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Setup window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Number Combine Gym Environment")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        
        # Action mapping
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Render the observation to the display window
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human playability

    env.close()