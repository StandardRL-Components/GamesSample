import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:19:22.923527
# Source Brief: brief_03029.md
# Brief Index: 3029
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, dx, dy, radius, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.initial_life = life
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            current_radius = int(self.radius * (self.life / self.initial_life))
            if current_radius > 0:
                # Create a temporary surface for transparency
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(self.x - current_radius), int(self.y - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Clear a path through a dense forest by cutting trees and growing vines to reach the golden target. "
        "Manage your terra-energy to fuel your actions and score points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to grow a vine on an empty tile, "
        "and press shift to cut down trees or remove existing vines."
    )
    auto_advance = True
    
    # Tile type constants
    T_EMPTY = 0
    T_TREE = 1
    T_WATER = 2
    T_TARGET = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = 20
        self.WIDTH = self.GRID_W * self.TILE_SIZE # 640
        self.HEIGHT = self.GRID_H * self.TILE_SIZE # 400
        self.MAX_STEPS = 2000
        self.INITIAL_ENERGY = 30
        self.VINE_COST = 2
        self.CUT_COST = 1

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 25, 20)
        self.COLOR_EARTH = (76, 57, 40)
        self.COLOR_TREE = (24, 80, 50)
        self.COLOR_TREE_TRUNK = (50, 30, 20)
        self.COLOR_WATER = (60, 120, 180)
        self.COLOR_TARGET = (255, 215, 0)
        self.COLOR_VINE = (80, 220, 100)
        self.COLOR_CURSOR = (0, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.target_pos = None
        self.terra_energy = 0
        self.vines = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.unlocked_milestones = None
        
        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # this is for debugging, not needed in final version

    def _generate_level(self):
        grid = np.full((self.GRID_W, self.GRID_H), self.T_TREE, dtype=int)
        start_pos = (1, self.GRID_H // 2)
        
        stack = deque([start_pos])
        visited = {start_pos}
        path_tiles = [start_pos]
        
        grid[start_pos] = self.T_EMPTY

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors)
                # Carve path
                grid[(cx + nx) // 2, (cy + ny) // 2] = self.T_EMPTY
                grid[nx, ny] = self.T_EMPTY
                visited.add((nx,ny))
                path_tiles.append((nx,ny))
                stack.append((nx,ny))
            else:
                stack.pop()

        # Set target at the furthest point from start
        self.target_pos = max(path_tiles, key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1]))
        grid[self.target_pos] = self.T_TARGET

        # Place some water patches
        for _ in range(30):
            wx, wy = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            if grid[wx, wy] == self.T_TREE:
                grid[wx, wy] = self.T_WATER

        return grid, start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid, start_pos = self._generate_level()
        self.cursor_pos = list(start_pos)
        self.terra_energy = self.INITIAL_ENERGY
        
        self.vines = set()
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        self.unlocked_milestones = set()
        
        return self._get_observation(), self._get_info()

    def _create_particles(self, x, y, count, color, life_range, speed_range, radius_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            life = self.np_random.integers(*life_range)
            radius = self.np_random.uniform(*radius_range)
            self.particles.append(Particle(
                x, y, color, life,
                math.cos(angle) * speed, math.sin(angle) * speed,
                radius
            ))

    def _handle_grow_vine(self):
        reward = 0
        cx, cy = self.cursor_pos
        
        if self.terra_energy < self.VINE_COST: return 0 # sfx: error_buzz
        if self.grid[cx, cy] != self.T_EMPTY: return 0 # sfx: error_buzz

        # Find adjacent vine or starting point to connect to
        connection_found = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                # Check if neighbor has a vine
                if any((nx,ny) in vine_pair for vine_pair in self.vines):
                    new_vine = tuple(sorted(((cx, cy), (nx, ny))))
                    if new_vine not in self.vines:
                        self.vines.add(new_vine)
                        connection_found = True
                        break
        
        if connection_found:
            self.terra_energy -= self.VINE_COST
            reward -= 0.1 # Action cost
            # sfx: vine_grow
            px = (cx + 0.5) * self.TILE_SIZE
            py = (cy + 0.5) * self.TILE_SIZE
            self._create_particles(px, py, 15, self.COLOR_VINE, (10, 20), (0.5, 1.5), (1, 3))
        
        return reward

    def _handle_cut(self):
        reward = 0
        cx, cy = self.cursor_pos
        
        if self.terra_energy < self.CUT_COST: return 0 # sfx: error_buzz

        px = (cx + 0.5) * self.TILE_SIZE
        py = (cy + 0.5) * self.TILE_SIZE
        
        # Case 1: Cut a tree
        if self.grid[cx, cy] == self.T_TREE:
            self.terra_energy -= self.CUT_COST
            self.grid[cx, cy] = self.T_EMPTY
            self.score += 1
            reward += 1.0 - 0.1 # Clear reward + action cost
            # sfx: tree_fall
            self._create_particles(px, py, 40, self.COLOR_TREE_TRUNK, (20, 40), (1, 3), (2, 5))
            
            # Milestone check
            if self.score > 0 and self.score % 5 == 0 and self.score not in self.unlocked_milestones:
                reward += 5.0
                self.terra_energy += 15 # Ability unlocked: bonus energy
                self.unlocked_milestones.add(self.score)
                # sfx: power_up
                self._create_particles(self.WIDTH/2, 30, 50, self.COLOR_TARGET, (30, 60), (2, 4), (3, 6))

        # Case 2: Cut a vine
        else:
            vines_at_cursor = {v for v in self.vines if (cx, cy) in v}
            if vines_at_cursor:
                self.terra_energy -= self.CUT_COST
                self.vines -= vines_at_cursor
                reward -= 0.1 # Action cost
                # sfx: vine_cut
                self._create_particles(px, py, 10, self.COLOR_VINE, (10, 20), (1, 2), (1, 2))
        
        return reward

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False
        
        # --- Handle Input and Actions ---
        # 1. Movement
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_H - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_W - 1: self.cursor_pos[0] += 1
        
        # 2. Terraforming actions (on button press)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed:
            reward += self._handle_grow_vine()
        
        if shift_pressed:
            reward += self._handle_cut()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        
        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            
        # --- Check Termination Conditions ---
        if tuple(self.cursor_pos) == self.target_pos:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: victory_fanfare

        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.terra_energy}

    def _render_game(self):
        ts = self.TILE_SIZE
        # Draw grid tiles
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = pygame.Rect(x * ts, y * ts, ts, ts)
                tile_type = self.grid[x, y]
                
                if tile_type == self.T_EMPTY:
                    pygame.draw.rect(self.screen, self.COLOR_EARTH, rect)
                elif tile_type == self.T_TREE:
                    pygame.draw.rect(self.screen, self.COLOR_TREE, rect)
                    pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, (x*ts+ts*0.4, y*ts+ts*0.4, ts*0.2, ts*0.2))
                elif tile_type == self.T_WATER:
                    pygame.draw.rect(self.screen, self.COLOR_WATER, rect)
                elif tile_type == self.T_TARGET:
                    # Glowing target
                    glow_radius = int(ts * 0.8)
                    pygame.gfxdraw.filled_circle(self.screen, x * ts + ts // 2, y * ts + ts // 2, glow_radius, self.COLOR_TARGET + (50,))
                    pygame.gfxdraw.filled_circle(self.screen, x * ts + ts // 2, y * ts + ts // 2, int(ts * 0.4), self.COLOR_TARGET)

        # Draw vines
        for p1, p2 in self.vines:
            start_pos = (p1[0] * ts + ts // 2, p1[1] * ts + ts // 2)
            end_pos = (p2[0] * ts + ts // 2, p2[1] * ts + ts // 2)
            pygame.draw.line(self.screen, self.COLOR_VINE, start_pos, end_pos, 3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * ts, cy * ts, ts, ts)
        # Glow effect
        glow_surf = pygame.Surface((ts*2, ts*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_CURSOR + (60,), (ts/2, ts/2, ts, ts), border_radius=5)
        self.screen.blit(glow_surf, ((cx-0.5)*ts, (cy-0.5)*ts))
        # Main cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((10, 10, 10, 180))
        self.screen.blit(ui_panel, (0, 0))

        # Energy display
        energy_text = self.font_ui.render(f"TERRA-ENERGY: {self.terra_energy}", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (10, 10))

        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (250, 10))
        
        # Steps display
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (420, 10))

        # Mini-map
        map_w, map_h = 64, 40 # 2px per tile
        map_surf = pygame.Surface((map_w, map_h))
        map_surf.fill(self.COLOR_BG)
        map_surf.set_alpha(200)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                color = self.COLOR_EARTH
                if self.grid[x, y] == self.T_TREE: color = self.COLOR_TREE
                elif self.grid[x, y] == self.T_WATER: color = self.COLOR_WATER
                elif self.grid[x, y] == self.T_TARGET: color = self.COLOR_TARGET
                pygame.draw.rect(map_surf, color, (x*2, y*2, 2, 2))
        # Player on minimap
        pygame.draw.rect(map_surf, self.COLOR_CURSOR, (self.cursor_pos[0]*2, self.cursor_pos[1]*2, 2, 2))
        self.screen.blit(map_surf, (self.WIDTH - map_w - 10, 50))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH - map_w - 11, 49, map_w+2, map_h+2), 1)

        if self.game_over:
            msg = "TARGET REACHED!" if tuple(self.cursor_pos) == self.target_pos else "TIME UP"
            msg_text = self.font_msg.render(msg, True, self.COLOR_TARGET)
            text_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    # Remap keys for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }
    
    # Pygame window for display
    # We need to unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Terraform Explorer")
    
    running = True
    while running:
        # Default action is no-op
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Energy: {info['energy']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.metadata["render_fps"])

    env.close()