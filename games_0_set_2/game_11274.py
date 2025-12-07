import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:53:01.625014
# Source Brief: brief_01274.md
# Brief Index: 1274
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Cover the grid by placing expanding waves. Achieve 80% coverage to win, "
        "but avoid wave overlaps which cause a critical overload."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a wave at the cursor's location."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_TIME_SECONDS = 120
    MAX_STEPS = MAX_TIME_SECONDS * FPS
    WIN_PERCENTAGE = 0.80

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GRID = (50, 50, 80)
    COLOR_INACTIVE = (40, 80, 120)
    COLOR_COVERED = (200, 200, 80)
    COLOR_WAVE = (0, 255, 150)
    COLOR_OVERLOAD = (255, 50, 50)
    COLOR_CURSOR = (255, 0, 255)
    COLOR_TEXT = (220, 220, 255)
    
    # Cell states
    STATE_INACTIVE = 0
    STATE_COVERED = 1
    STATE_OVERLOAD = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        # Observation space is the pixel data from the screen
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action space: [movement, space_held, shift_held]
        # movement: 0=None, 1=Up, 2=Down, 3=Left, 4=Right
        # space_held: 0=Released, 1=Held
        # shift_held: 0=Released, 1=Held (unused in this game)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables (initialized in reset)
        self.level = 0
        self.won_last_game = False
        self.grid_size = 0
        self.cell_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self.wave_speed_modifier = 0.0
        self.cursor_pos = [0, 0]
        self.grid = []
        self.waves = []
        self.total_cells = 0
        self.covered_cells = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- LEVEL PROGRESSION ---
        if self.won_last_game:
            self.level += 1
        else:
            self.level = 1
        self.won_last_game = False

        # --- DIFFICULTY SCALING ---
        self.grid_size = min(15, 5 + (self.level - 1) * 2)
        # Base speed: 2 cells/sec. Increase by 0.5 cells/sec per level.
        self.wave_speed_modifier = 2.0 + (self.level - 1) * 0.5

        # --- GRID & CELL SIZING ---
        grid_area_size = min(self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 80)
        self.cell_size = grid_area_size / self.grid_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_size * self.cell_size) / 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_size * self.cell_size) / 2
        
        # --- STATE INITIALIZATION ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        self.grid = [[self.STATE_INACTIVE for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.waves = []
        self.total_cells = self.grid_size * self.grid_size
        self.covered_cells = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- UNPACK ACTION ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- ACTION HANDLING ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_size - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_size - 1)

        # 2. Cell Activation
        if space_pressed:
            cx, cy = self.cursor_pos
            if self.grid[cy][cx] == self.STATE_INACTIVE:
                wave_center_x = self.grid_offset_x + cx * self.cell_size + self.cell_size / 2
                wave_center_y = self.grid_offset_y + cy * self.cell_size + self.cell_size / 2
                wave_speed_px = self.wave_speed_modifier * self.cell_size / self.FPS
                
                self.waves.append({
                    "cx": wave_center_x, "cy": wave_center_y,
                    "radius": 0, "speed": wave_speed_px
                })
        
        # --- GAME LOGIC: WAVE PROPAGATION & COLLISION ---
        old_covered_count = self._count_covered_cells()
        
        # Grow all active waves
        for wave in self.waves:
            wave['radius'] += wave['speed']

        # Use an influence map to detect overlaps and new coverage
        influence_map = [[0] * self.grid_size for _ in range(self.grid_size)]
        for wave in self.waves:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    cell_center_x = self.grid_offset_x + c * self.cell_size + self.cell_size / 2
                    cell_center_y = self.grid_offset_y + r * self.cell_size + self.cell_size / 2
                    dist_sq = (cell_center_x - wave['cx'])**2 + (cell_center_y - wave['cy'])**2
                    if dist_sq <= wave['radius']**2:
                        influence_map[r][c] += 1

        num_overloads_this_step = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if influence_map[r][c] > 1 and self.grid[r][c] != self.STATE_OVERLOAD:
                    self.grid[r][c] = self.STATE_OVERLOAD
                    num_overloads_this_step += 1
                elif influence_map[r][c] == 1 and self.grid[r][c] == self.STATE_INACTIVE:
                    self.grid[r][c] = self.STATE_COVERED
        
        if num_overloads_this_step > 0:
            self.game_over = True
            reward += num_overloads_this_step * -5.0

        # --- REWARD CALCULATION ---
        self.covered_cells = self._count_covered_cells()
        newly_covered = self.covered_cells - old_covered_count
        if newly_covered > 0:
            reward += newly_covered * 0.1

        # --- TERMINATION CHECK ---
        terminated = False
        truncated = False
        if self.game_over: # Overload
            terminated = True
            reward = -100.0
        elif self.covered_cells / self.total_cells >= self.WIN_PERCENTAGE:
            terminated = True
            reward = 100.0
            self.won_last_game = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Timeout is a form of termination, not truncation
            reward = -100.0

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's surfarray has dimensions (width, height, channels)
        # but Gymnasium expects (height, width, channels)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        coverage = self.covered_cells / self.total_cells if self.total_cells > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "coverage": coverage,
            "won": coverage >= self.WIN_PERCENTAGE
        }

    def _render_game(self):
        # 1. Draw grid cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                state = self.grid[r][c]
                if state == self.STATE_INACTIVE:
                    color = self.COLOR_INACTIVE
                elif state == self.STATE_COVERED:
                    color = self.COLOR_COVERED
                else: # OVERLOAD
                    color = self.COLOR_OVERLOAD
                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, cell_rect, 1)

        # 2. Draw expanding waves
        for wave in self.waves:
            # Use gfxdraw for anti-aliased, transparent circles
            radius = int(wave['radius'])
            if radius > 0:
                # Transparent fill
                pygame.gfxdraw.filled_circle(
                    self.screen, int(wave['cx']), int(wave['cy']), radius,
                    self.COLOR_WAVE + (60,) # Add alpha
                )
                # Anti-aliased outline
                pygame.gfxdraw.aacircle(
                    self.screen, int(wave['cx']), int(wave['cy']), radius, self.COLOR_WAVE
                )

        # 3. Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Coverage Percentage
        coverage_pct = self.covered_cells / self.total_cells if self.total_cells > 0 else 0
        coverage_text = f"COVERAGE: {coverage_pct:.1%}"
        coverage_surf = self.font_large.render(coverage_text, True, self.COLOR_TEXT)
        self.screen.blit(coverage_surf, (20, 10))

        # Timer
        time_left = max(0, self.MAX_TIME_SECONDS - (self.steps / self.FPS))
        timer_text = f"TIME: {time_left:.1f}s"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))

        # Level
        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (20, 40))

    def _count_covered_cells(self):
        return sum(row.count(self.STATE_COVERED) for row in self.grid)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It is not used by the evaluation environment.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Grid")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Input ---
        movement = 0 # None
        space_held = 0 # Released
        shift_held = 0 # Released (unused)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Convert observation back to a Pygame surface
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        if terminated:
            # Display game over message
            font = pygame.font.SysFont("monospace", 48, bold=True)
            win = info.get("won", False)
            msg = "LEVEL COMPLETE" if win else "OVERLOAD / TIMEOUT"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            text_surf = font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            screen.blit(text_surf, text_rect)
            
            font_small = pygame.font.SysFont("monospace", 24)
            reset_text_surf = font_small.render("Press 'R' to restart", True, GameEnv.COLOR_TEXT)
            reset_text_rect = reset_text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            screen.blit(reset_text_surf, reset_text_rect)

            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()