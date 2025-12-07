import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:03:25.313718
# Source Brief: brief_02077.md
# Brief Index: 2077
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
    A Gymnasium environment for a symbol-matching puzzle game.

    The player controls a cursor on a grid of falling symbols. The goal is to
    select groups of 3 or more matching adjacent symbols to clear them,
    triggering chain reactions for points. Clearing symbols has a chance to
    spawn a bomb. If a bomb is included in a match, the game ends.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `action[1]`: Select (0: released, 1: held)
    - `action[2]`: Unused (0: released, 1: held)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 per symbol cleared.
    - +1.0 per chain reaction link.
    - +100 for winning (reaching 1000 score).
    - -50 for losing (activating a bomb).

    **Termination:**
    - Score >= 1000 (win).
    - A bomb is part of a match (loss).
    - Episode length > 1000 steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match groups of 3 or more adjacent symbols to clear them from the grid and score points. "
        "Clearing symbols may create bombs; matching a bomb will end the game."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to select and clear a "
        "highlighted group of matching symbols."
    )
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 40, 25
    CELL_SIZE = 16
    SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    
    # Symbol types and colors
    SYMBOLS = {
        0: {"name": "empty", "color": (0, 0, 0)},
        1: {"name": "circle", "color": (255, 80, 80)},  # Red
        2: {"name": "square", "color": (80, 255, 80)},  # Green
        3: {"name": "triangle", "color": (80, 120, 255)}, # Blue
        4: {"name": "bomb", "color": (50, 50, 50)}    # Black
    }
    
    # Game parameters
    FALL_INTERVAL = 5       # Ticks between symbol falls
    MIN_MATCH_SIZE = 3
    BOMB_SPAWN_CHANCE = 0.10
    WIN_SCORE = 1000
    MAX_STEPS = 1000
    CHAIN_RESOLVE_DELAY = 10 # Frames to wait for symbols to fall after a match

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.particles = []
        self.is_chaining = False
        self.chain_timer = 0
        self.last_space_held = False

        # self.reset() # This is handled by the environment runner
        # self.validate_implementation() # This is for internal testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.particles = []
        self.is_chaining = False
        self.chain_timer = 0
        self.last_space_held = False
        
        # Pre-fill the board partially
        for y in range(self.GRID_HEIGHT // 2, self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self.grid[x, y] = self.np_random.integers(1, 4)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = bool(space_held)

        # --- Game Logic Update ---
        self._update_particles()
        
        if self.is_chaining:
            self.chain_timer -= 1
            self._apply_gravity()
            if self.chain_timer <= 0:
                reward += self._resolve_chains()
        else:
            self._handle_player_input(movement, space_pressed)
            reward_from_match = self._process_player_match(space_pressed)
            if reward_from_match > 0:
                reward += reward_from_match
            else: # Only update gravity/spawning if no match was made
                self._update_falling_symbols()

        # --- Check for Termination ---
        terminated = self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS or self.game_over
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.score >= self.WIN_SCORE:
            reward += 100 # Win reward
        if terminated and self.game_over and not (self.score >= self.WIN_SCORE):
            reward += -50 # Bomb loss reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_input(self, movement, space_pressed):
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

    def _process_player_match(self, space_pressed):
        if not space_pressed:
            return 0
        
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] == 0: return 0 # Cannot match empty space

        group = self._find_connected_group(cx, cy)
        if len(group) < self.MIN_MATCH_SIZE:
            # Sound: Misfire/error sound
            return 0
        
        # Sound: Match success sound
        return self._initiate_match(group)

    def _initiate_match(self, group):
        reward = 0
        # Check for bombs first
        for x, y in group:
            if self.grid[x, y] == 4: # Bomb
                self.game_over = True
                self._create_explosion(x, y)
                # Sound: Large explosion
                return 0

        # Process a successful match
        reward += 1.0 # Chain reaction reward
        for x, y in group:
            reward += 0.1 # Per-symbol reward
            self._create_match_particles(x, y, self.SYMBOLS[self.grid[x, y]]['color'])
            
            # 10% chance to spawn a bomb
            if self.np_random.random() < self.BOMB_SPAWN_CHANCE:
                self.grid[x, y] = 4
            else:
                self.grid[x, y] = 0
        
        self.score += len(group)
        self.is_chaining = True
        self.chain_timer = self.CHAIN_RESOLVE_DELAY
        return reward

    def _resolve_chains(self):
        reward = 0
        all_new_matches = []
        
        # Find all matches on the board
        visited = np.zeros_like(self.grid, dtype=bool)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] != 0 and not visited[x, y]:
                    group = self._find_connected_group(x, y)
                    for gx, gy in group:
                        visited[gx, gy] = True
                    if len(group) >= self.MIN_MATCH_SIZE:
                        all_new_matches.append(group)
        
        if not all_new_matches:
            self.is_chaining = False
            return 0

        # Process all found matches simultaneously
        for group in all_new_matches:
            # Sound: Chain link sound
            reward += self._initiate_match(group)
            if self.game_over: break # Bomb ended the chain
        
        return reward

    def _update_falling_symbols(self):
        self.fall_timer += 1
        if self.fall_timer >= self.FALL_INTERVAL:
            self.fall_timer = 0
            self._apply_gravity()
            self._spawn_new_symbols()

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_y = -1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0 and empty_y == -1:
                    empty_y = y
                elif self.grid[x, y] != 0 and empty_y != -1:
                    self.grid[x, empty_y] = self.grid[x, y]
                    self.grid[x, y] = 0
                    empty_y -= 1

    def _spawn_new_symbols(self):
        for x in range(self.GRID_WIDTH):
            if self.grid[x, 0] == 0:
                self.grid[x, 0] = self.np_random.integers(1, 4) # Types 1, 2, 3

    def _find_connected_group(self, start_x, start_y):
        target_symbol = self.grid[start_x, start_y]
        if target_symbol == 0:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[nx, ny] == target_symbol:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group
    
    # --- Particle and Effects ---
    def _create_match_particles(self, grid_x, grid_y, color):
        cx = (grid_x + 0.5) * self.CELL_SIZE
        cy = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(5):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                "pos": [cx, cy],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 20, "max_life": 20,
                "color": color, "type": "spark"
            })

    def _create_explosion(self, grid_x, grid_y):
        cx = (grid_x + 0.5) * self.CELL_SIZE
        cy = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(50):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            self.particles.append({
                "pos": [cx, cy],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 40, "max_life": 40,
                "color": (255, 150, 50), "type": "spark"
            })
        self.particles.append({
            "pos": [cx, cy], "radius": 0, "max_radius": self.CELL_SIZE * 3,
            "life": 20, "max_life": 20,
            "color": (255, 255, 255), "type": "shockwave"
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            if p["type"] == "spark":
                p["pos"][0] += p["vel"][0]
                p["pos"][1] += p["vel"][1]
                p["vel"][1] += 0.1 # Gravity on particles
            elif p["type"] == "shockwave":
                p["radius"] = p["max_radius"] * (1 - p["life"] / p["max_life"])
    
    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_symbols()
        self._render_cursor_and_highlight()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_symbols(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                symbol_id = self.grid[x, y]
                if symbol_id == 0: continue
                
                s_info = self.SYMBOLS[symbol_id]
                color = s_info["color"]
                cx = int(rect[0] + self.CELL_SIZE / 2)
                cy = int(rect[1] + self.CELL_SIZE / 2)
                radius = int(self.CELL_SIZE * 0.4)

                if s_info["name"] == "circle":
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                elif s_info["name"] == "square":
                    s_rect = (rect[0] + 2, rect[1] + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                    pygame.draw.rect(self.screen, color, s_rect, border_radius=2)
                elif s_info["name"] == "triangle":
                    points = [
                        (cx, cy - radius),
                        (cx - radius, cy + radius * 0.7),
                        (cx + radius, cy + radius * 0.7)
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, points, color)
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                elif s_info["name"] == "bomb":
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                    # Skull
                    skull_color = (200, 200, 200)
                    pygame.draw.rect(self.screen, skull_color, (cx - 3, cy - 2, 6, 6), border_radius=1)
                    pygame.draw.circle(self.screen, skull_color, (cx, cy + 3), 3)
                    pygame.draw.rect(self.screen, self.COLOR_BG, (cx-2, cy-1, 1, 2))
                    pygame.draw.rect(self.screen, self.COLOR_BG, (cx+1, cy-1, 1, 2))

    def _render_cursor_and_highlight(self):
        # Highlight potential match
        cx, cy = self.cursor_pos
        if self.grid[cx, cy] != 0:
            group = self._find_connected_group(cx, cy)
            if len(group) >= self.MIN_MATCH_SIZE:
                highlight_color = self.SYMBOLS[self.grid[cx, cy]]['color']
                for hx, hy in group:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE))
                    s.set_alpha(80)
                    s.fill(highlight_color)
                    self.screen.blit(s, (hx * self.CELL_SIZE, hy * self.CELL_SIZE))
        
        # Draw cursor
        cursor_rect = (
            self.cursor_pos[0] * self.CELL_SIZE, 
            self.cursor_pos[1] * self.CELL_SIZE, 
            self.CELL_SIZE, 
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_particles(self):
        for p in self.particles:
            if p["type"] == "spark":
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = p["color"] # color is already a tuple (r,g,b)
                s = pygame.Surface((6, 6), pygame.SRCALPHA)
                size = int(3 * (p["life"] / p["max_life"]))
                pygame.draw.circle(s, color + (alpha,), (3,3), max(1, size))
                self.screen.blit(s, (int(p["pos"][0]-3), int(p["pos"][1]-3)))

            elif p["type"] == "shockwave":
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = p["color"]
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color + (alpha,))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        if self.game_over:
            status = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_text = self.font.render(status, True, self.COLOR_CURSOR)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 10))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "is_chaining": self.is_chaining}
    
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local development.
    # To run it, you may need to unset the SDL_VIDEODRIVER dummy variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Puzzle")
    clock = pygame.time.Clock()
    
    running = True
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
        
        action = [movement, space, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for smooth play

    env.close()