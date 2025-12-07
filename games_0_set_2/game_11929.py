import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:48:58.744311
# Source Brief: brief_01929.md
# Brief Index: 1929
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Strategically place quantum blocks to build a stable city. Manage resources and "
        "create synergistic connections to reach 100% stability."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to place the selected block. "
        "Press shift to cycle through available block types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (30, 40, 50)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_STABILITY_HIGH = (0, 255, 128)
    COLOR_STABILITY_MID = (255, 255, 0)
    COLOR_STABILITY_LOW = (255, 50, 50)

    BLOCK_TYPES = [
        {
            "name": "Stabilizer",
            "color": (0, 150, 255),
            "cost": 1,
            "base_stability": 2,
        },
        {
            "name": "Conduit",
            "color": (255, 80, 80),
            "cost": 3,
            "base_stability": -4,
        },
        {
            "name": "Generator",
            "color": (0, 220, 120),
            "cost": 2,
            "base_stability": 0,
        },
    ]

    # Synergy: (type1, type2) -> stability_bonus
    SYNERGY_MATRIX = {
        (0, 0): 2, (0, 1): -3, (0, 2): 2,
        (1, 0): -3, (1, 1): 0, (1, 2): -2,
        (2, 0): 2, (2, 1): -2, (2, 2): 1,
    }
    PORTAL_BONUS = 5
    GENERATOR_RESOURCE_BONUS = 3

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)
        
        # Game state variables are initialized in reset()
        self.grid = []
        self.cursor_pos = [0, 0]
        self.stability = 0
        self.resources = []
        self.selected_block_idx = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        # self.validate_implementation() # Commented out as it's not part of the standard API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.stability = 50
        self.resources = [5, 2, 3] 
        self.selected_block_idx = 0
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        reward = 0
        
        # --- Handle Input ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if shift_pressed:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPES)
            # sfx: UI_CYCLE

        if space_pressed:
            reward += self._place_block()
            # sfx: BLOCK_PLACE_SUCCESS or BLOCK_PLACE_FAIL

        self.steps += 1
        
        # --- Update Game State ---
        self._update_particles()
        
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.stability >= 100:
                reward += 100
                self.score += 100
                # sfx: VICTORY
            elif self.stability <= 0:
                reward -= 100
                self.score -= 100
                # sfx: DEFEAT

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # In this game logic, time limit is a termination condition.
        # We'll return terminated=True for both win/loss and time limit.
        is_terminated = self.stability <= 0 or self.stability >= 100 or self.steps >= self.MAX_STEPS
        is_truncated = False # The logic combines termination and truncation.

        return self._get_observation(), reward, is_terminated, is_truncated, self._get_info()

    def _place_block(self):
        cx, cy = self.cursor_pos
        if self.grid[cy][cx] is not None:
            return 0 # Cell occupied

        block_type = self.BLOCK_TYPES[self.selected_block_idx]
        if self.resources[self.selected_block_idx] < block_type["cost"]:
            return 0 # Not enough resources

        self.resources[self.selected_block_idx] -= block_type["cost"]
        self.grid[cy][cx] = self.selected_block_idx
        
        # --- Calculate Stability Change & Reward ---
        delta_stability = 0
        portal_reward = 0

        # Base stability
        delta_stability += block_type["base_stability"]
        
        # Generator resource bonus
        if self.selected_block_idx == 2: # Generator
            self.resources[2] += self.GENERATOR_RESOURCE_BONUS
            
        # Synergy and Portal bonuses
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                neighbor_type_idx = self.grid[ny][nx]
                if neighbor_type_idx is not None:
                    # Synergy
                    synergy_key = tuple(sorted((self.selected_block_idx, neighbor_type_idx)))
                    delta_stability += self.SYNERGY_MATRIX.get(synergy_key, 0)
                    
                    # Portal connection
                    if self.selected_block_idx == neighbor_type_idx:
                        delta_stability += self.PORTAL_BONUS
                        portal_reward += 1.0 # Event-based reward for new portal
                        # sfx: PORTAL_FORM
        
        # Apply stability change and clamp
        self.stability += delta_stability
        self.stability = np.clip(self.stability, 0, 100)
        
        # Create particles
        self._create_particles(cx, cy, block_type["color"])
        
        # Continuous feedback reward
        reward = 0
        if delta_stability > 0: reward += 0.1
        elif delta_stability < 0: reward -= 0.1
        
        return reward + portal_reward

    def _check_termination(self):
        return self.stability <= 0 or self.stability >= 100 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stability": self.stability}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_portals()
        self._draw_blocks()
        self._draw_cursor()
        self._draw_particles()

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_portals(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                block_type_idx = self.grid[y][x]
                if block_type_idx is not None:
                    # Check right neighbor
                    if x + 1 < self.GRID_COLS and self.grid[y][x+1] == block_type_idx:
                        self._draw_animated_line(x, y, x + 1, y, self.BLOCK_TYPES[block_type_idx]["color"])
                    # Check down neighbor
                    if y + 1 < self.GRID_ROWS and self.grid[y+1][x] == block_type_idx:
                        self._draw_animated_line(x, y, x, y + 1, self.BLOCK_TYPES[block_type_idx]["color"])
                        
    def _draw_animated_line(self, x1, y1, x2, y2, color):
        start_pos = (x1 * self.CELL_SIZE + self.CELL_SIZE // 2, y1 * self.CELL_SIZE + self.CELL_SIZE // 2)
        end_pos = (x2 * self.CELL_SIZE + self.CELL_SIZE // 2, y2 * self.CELL_SIZE + self.CELL_SIZE // 2)
        
        # Base line
        pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
        
        # Animated pulse
        pulse_progress = (self.steps % 30) / 30.0
        px = start_pos[0] + (end_pos[0] - start_pos[0]) * pulse_progress
        py = start_pos[1] + (end_pos[1] - start_pos[1]) * pulse_progress
        pulse_color = (min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80))
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 4, pulse_color)

    def _draw_blocks(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                block_type_idx = self.grid[y][x]
                if block_type_idx is not None:
                    color = self.BLOCK_TYPES[block_type_idx]["color"]
                    rect = pygame.Rect(
                        x * self.CELL_SIZE, y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    self._draw_glowing_rect(rect, color)

    def _draw_glowing_rect(self, rect, color, glow_size=10):
        glow_color = tuple(c // 2 for c in color)
        for i in range(glow_size, 0, -2):
            glow_rect = rect.inflate(i, i)
            alpha = 150 * (1 - i / glow_size)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color + (alpha,), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
        
        inner_rect = rect.inflate(-8, -8)
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)
        
        highlight_color = (min(255, color[0]+60), min(255, color[1]+60), min(255, color[2]+60))
        pygame.draw.rect(self.screen, highlight_color, inner_rect, 2, border_radius=3)

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            cx * self.CELL_SIZE, cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Pulsating alpha
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = self.COLOR_CURSOR + (int(alpha),)
        
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        line_width = 3
        pygame.draw.line(s, color, (0, 0), (rect.width, 0), line_width)
        pygame.draw.line(s, color, (0, 0), (0, rect.height), line_width)
        pygame.draw.line(s, color, (rect.width - line_width//2, 0), (rect.width-line_width//2, rect.height), line_width)
        pygame.draw.line(s, color, (0, rect.height-line_width//2), (rect.width, rect.height-line_width//2), line_width)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # --- Stability Bar ---
        bar_y = 10
        bar_height = 20
        bar_width = self.SCREEN_WIDTH - 20
        
        stability_ratio = self.stability / 100.0
        
        # Interpolate color
        if stability_ratio > 0.5:
            interp = (stability_ratio - 0.5) * 2
            bar_color = [int(self.COLOR_STABILITY_MID[i] * (1-interp) + self.COLOR_STABILITY_HIGH[i] * interp) for i in range(3)]
        else:
            interp = stability_ratio * 2
            bar_color = [int(self.COLOR_STABILITY_LOW[i] * (1-interp) + self.COLOR_STABILITY_MID[i] * interp) for i in range(3)]
            
        # Background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, bar_y, bar_width, bar_height), border_radius=5)
        # Foreground
        fill_width = max(0, bar_width * stability_ratio)
        pygame.draw.rect(self.screen, bar_color, (10, bar_y, fill_width, bar_height), border_radius=5)
        # Text
        stability_text = f"STABILITY: {int(self.stability)}%"
        self._draw_text(stability_text, self.font_medium, (self.SCREEN_WIDTH // 2, bar_y + bar_height // 2), centered=True)

        # --- Block Selector ---
        ui_y = self.SCREEN_HEIGHT - 60
        total_ui_width = len(self.BLOCK_TYPES) * 100
        start_x = (self.SCREEN_WIDTH - total_ui_width) // 2
        
        for i, block_type in enumerate(self.BLOCK_TYPES):
            box_x = start_x + i * 100
            
            # Highlight selected
            if i == self.selected_block_idx:
                highlight_rect = pygame.Rect(box_x - 5, ui_y - 5, 90, 50)
                alpha = 128 + 127 * math.sin(self.steps * 0.2)
                s = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_CURSOR + (int(alpha),), s.get_rect(), 2, border_radius=8)
                self.screen.blit(s, highlight_rect.topleft)

            # Block icon
            block_rect = pygame.Rect(box_x, ui_y, 40, 40)
            self._draw_glowing_rect(block_rect, block_type["color"], glow_size=6)
            
            # Resource count
            resource_text = f"x{self.resources[i]}"
            text_color = self.COLOR_TEXT if self.resources[i] >= block_type["cost"] else self.COLOR_STABILITY_LOW
            self._draw_text(resource_text, self.font_medium, (box_x + 65, ui_y + 20), color=text_color, centered=True)
            
        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.stability >= 100:
                msg = "CITY STABLE"
                color = self.COLOR_STABILITY_HIGH
            else:
                msg = "CONNECTION LOST"
                color = self.COLOR_STABILITY_LOW
            self._draw_text(msg, self.font_large, (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 - 20), color=color, centered=True)
            self._draw_text(f"Final Score: {int(self.score)}", self.font_medium, (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2 + 30), centered=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, centered=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if centered:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, color):
        px = grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # lifetime -= 1
        self.particles = [p for p in self.particles if p[4] > 0]

    def _draw_particles(self):
        for px, py, _, _, lifetime, color in self.particles:
            radius = int(lifetime * 0.2)
            if radius > 0:
                alpha = int(255 * (lifetime / 40.0))
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color + (alpha,), (radius, radius), radius)
                self.screen.blit(s, (int(px) - radius, int(py) - radius))


    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Allow running with a display for human play
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum City Builder")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")

        # --- Rendering for Human Player ---
        # The observation is already a rendered frame, we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()