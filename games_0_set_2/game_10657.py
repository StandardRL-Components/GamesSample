import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:46:45.159641
# Source Brief: brief_00657.md
# Brief Index: 657
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Nanobot Containment'.

    In this real-time strategy game, the player controls a cursor to deploy
    nanobots on a grid. The goal is to contain and eradicate a spreading
    polygonal virus before it reaches the central city core.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `action[1]`: Deploy Nanobot (0: released, 1: pressed)
    - `action[2]`: Reserved (no-op)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game screen.

    **Rewards:**
    - **Win:** +100 (all virus cells eradicated)
    - **Loss:** -100 (virus reaches city core)
    - **Virus Destroyed:** +1 per cell
    - **Virus Exists:** -0.1 per cell, per step
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Deploy nanobots to contain and eradicate a spreading virus before it infects the city core."
    user_guide = "Use the arrow keys (↑↓←→) to move the cursor. Press space to deploy a nanobot at the cursor's location."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20

    # --- Colors ---
    COLOR_BG = (10, 15, 25)
    COLOR_GRID = (30, 40, 60)
    COLOR_CORE = (0, 191, 255)
    COLOR_CORE_GLOW = (0, 100, 150)
    COLOR_VIRUS = (255, 20, 50)
    COLOR_VIRUS_GLOW = (180, 0, 30)
    COLOR_NANOBOT = (20, 255, 150)
    COLOR_NANOBOT_GLOW = (10, 180, 110)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # --- Game Parameters ---
    INITIAL_RESOURCES = 100
    NANOBOT_COST = 5
    INITIAL_SPREAD_INTERVAL = 15  # Spread every N steps
    MAX_STEPS = 1000
    CORE_RADIUS_PX = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.resources = 0
        self.cursor_pos = (0, 0)
        self.virus_cells = set()
        self.nanobot_cells = set()
        self.particles = []
        self.spread_interval = 0
        self.spread_timer = 0
        self.core_center_px = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.core_grid_cells = self._get_core_grid_cells()
        self.prev_space_held = False
        
        # self.reset() is called by the wrapper
        # self.validate_implementation() is for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.resources = self.INITIAL_RESOURCES
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        
        self.nanobot_cells.clear()
        self.virus_cells.clear()
        
        # Start virus at a random edge
        edge = self.np_random.integers(4)
        if edge == 0: # top
            start_pos = (self.np_random.integers(self.GRID_COLS), 0)
        elif edge == 1: # bottom
            start_pos = (self.np_random.integers(self.GRID_COLS), self.GRID_ROWS - 1)
        elif edge == 2: # left
            start_pos = (0, self.np_random.integers(self.GRID_ROWS))
        else: # right
            start_pos = (self.GRID_COLS - 1, self.np_random.integers(self.GRID_ROWS))
        self.virus_cells.add(start_pos)

        self.particles.clear()
        
        self.spread_interval = self.INITIAL_SPREAD_INTERVAL
        self.spread_timer = self.spread_interval
        
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Input & Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._move_cursor(movement)
        
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            deploy_reward = self._deploy_nanobot()
            reward += deploy_reward
        self.prev_space_held = space_held

        # --- 2. Update Game Logic ---
        self._update_particles()
        
        self.spread_timer -= 1
        if self.spread_timer <= 0:
            self._spread_virus()
            # Difficulty scaling
            if self.steps > 0 and self.steps % 200 == 0:
                self.spread_interval = max(3, self.spread_interval - 1)
            self.spread_timer = self.spread_interval

        # --- 3. Calculate Continuous Reward ---
        reward -= 0.1 * len(self.virus_cells)
        self.score += reward

        # --- 4. Check Termination Conditions ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward
        self.game_over = terminated

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        x = np.clip(x, 0, self.GRID_COLS - 1)
        y = np.clip(y, 0, self.GRID_ROWS - 1)
        self.cursor_pos = (int(x), int(y))

    def _deploy_nanobot(self):
        reward = 0
        target_cell = self.cursor_pos
        
        is_empty = target_cell not in self.virus_cells and target_cell not in self.nanobot_cells
        
        if self.resources >= self.NANOBOT_COST and is_empty:
            self.resources -= self.NANOBOT_COST
            self.nanobot_cells.add(target_cell)
            # sfx: nanobot_deploy.wav
            self._create_particles(target_cell, self.COLOR_NANOBOT, 20, "deploy")

            destroyed_this_turn = set()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (target_cell[0] + dx, target_cell[1] + dy)
                if check_pos in self.virus_cells:
                    destroyed_this_turn.add(check_pos)
            
            for virus_pos in destroyed_this_turn:
                self.virus_cells.remove(virus_pos)
                reward += 1.0
                # sfx: virus_destroy.wav
                self._create_particles(virus_pos, self.COLOR_VIRUS, 30, "destroy")
        
        return reward

    def _spread_virus(self):
        newly_infected = set()
        candidates = set()
        
        for vx, vy in self.virus_cells:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    neighbor = (nx, ny)
                    if neighbor not in self.virus_cells and neighbor not in self.nanobot_cells:
                        candidates.add(neighbor)

        for cx, cy in candidates:
            # Infection chance increases with more adjacent virus cells
            virus_neighbors = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (cx + dx, cy + dy) in self.virus_cells:
                    virus_neighbors += 1
            
            infection_prob = 0.25 * virus_neighbors
            if self.np_random.random() < infection_prob:
                newly_infected.add((cx, cy))
                # sfx: virus_spread.wav

        self.virus_cells.update(newly_infected)

    def _check_termination(self):
        # Loss condition: Virus reaches core
        if not self.virus_cells.isdisjoint(self.core_grid_cells):
            self.win_message = "CORE INFECTED"
            return True, -100.0

        # Win condition: Virus eradicated
        if not self.virus_cells and self.steps > 0:
            self.win_message = "VIRUS ERADICATED"
            return True, 100.0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "virus_count": len(self.virus_cells),
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw city core with glow
        glow_radius = int(self.CORE_RADIUS_PX + 10 + 5 * math.sin(self.steps * 0.05))
        pygame.gfxdraw.filled_circle(self.screen, *self.core_center_px, glow_radius, (*self.COLOR_CORE_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, *self.core_center_px, self.CORE_RADIUS_PX, self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.screen, *self.core_center_px, self.CORE_RADIUS_PX, self.COLOR_CORE)

        # Draw virus cells with pulsating glow
        for x, y in self.virus_cells:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pulse = 1 + 0.15 * math.sin(self.steps * 0.2 + x + y)
            glow_rect = rect.inflate(self.CELL_SIZE * 0.6 * pulse, self.CELL_SIZE * 0.6 * pulse)
            
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*self.COLOR_VIRUS_GLOW, 100), shape_surf.get_rect(), border_radius=4)
            self.screen.blit(shape_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_VIRUS, rect.inflate(-4, -4), border_radius=2)

        # Draw nanobots
        for x, y in self.nanobot_cells:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            glow_rect = rect.inflate(self.CELL_SIZE * 0.4, self.CELL_SIZE * 0.4)
            
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*self.COLOR_NANOBOT_GLOW, 120), shape_surf.get_rect(), border_radius=4)
            self.screen.blit(shape_surf, glow_rect.topleft)
            
            pygame.draw.rect(self.screen, self.COLOR_NANOBOT, rect.inflate(-4, -4), border_radius=2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor
        cx, cy = self.cursor_pos
        px, py = cx * self.CELL_SIZE + self.CELL_SIZE // 2, cy * self.CELL_SIZE + self.CELL_SIZE // 2
        size = self.CELL_SIZE // 2
        line_len = size // 2
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py - size), (px - size + line_len, py - size), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py - size), (px - size, py - size + line_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px + size, py - size), (px + size - line_len, py - size), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px + size, py - size), (px + size, py - size + line_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py + size), (px - size + line_len, py + size), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py + size), (px - size, py + size - line_len), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px + size, py + size), (px + size - line_len, py + size), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px + size, py + size), (px + size, py + size - line_len), 2)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color):
            shadow = font.render(text, True, shadow_color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            content = font.render(text, True, color)
            self.screen.blit(content, pos)

        # Resources
        res_text = f"RESOURCES: {self.resources}"
        draw_text(res_text, (10, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Virus Coverage
        total_cells = self.GRID_COLS * self.GRID_ROWS
        coverage = (len(self.virus_cells) / total_cells) * 100
        cov_text = f"VIRUS: {coverage:.1f}%"
        text_width = self.font_main.size(cov_text)[0]
        draw_text(cov_text, (self.SCREEN_WIDTH - text_width - 10, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Game Over Message
        if self.game_over:
            text_width, text_height = self.font_large.size(self.win_message)
            pos = ((self.SCREEN_WIDTH - text_width) // 2, (self.SCREEN_HEIGHT - text_height) // 2)
            draw_text(self.win_message, pos, self.font_large, self.COLOR_CURSOR, self.COLOR_TEXT_SHADOW)
            
    def _get_core_grid_cells(self):
        core_cells = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                cell_center_x = (c + 0.5) * self.CELL_SIZE
                cell_center_y = (r + 0.5) * self.CELL_SIZE
                dist_sq = (cell_center_x - self.core_center_px[0])**2 + (cell_center_y - self.core_center_px[1])**2
                if dist_sq < self.CORE_RADIUS_PX**2:
                    core_cells.add((c, r))
        return core_cells

    def _create_particles(self, grid_pos, color, count, p_type):
        px, py = (grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            if p_type == "deploy":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(15, 25)
            elif p_type == "destroy":
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(20, 40)
            
            self.particles.append(Particle(px, py, vel, color, life, self.np_random))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, x, y, vel, color, life, np_random):
        self.np_random = np_random
        self.pos = [x, y]
        self.vel = vel
        self.color = color
        self.life = life
        self.max_life = life
        self.radius = self.np_random.integers(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95 # friction
        self.vel[1] *= 0.95
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            current_radius = int(self.radius * (self.life / self.max_life))
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos[0]), int(self.pos[1]), current_radius, (*self.color, alpha)
                )

if __name__ == "__main__":
    # Example of how to use the environment
    # Set the video driver to a real one for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    # Use Pygame for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Nanobot Containment")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Human Input Mapping ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is the rendered screen, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Final Info: {info}")

    # Keep the window open for a few seconds to see the result
    pygame.time.wait(3000)
    env.close()