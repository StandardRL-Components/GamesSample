
# Generated: 2025-08-28T05:21:13.582522
# Source Brief: brief_02597.md
# Brief Index: 2597

        
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

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the labyrinth. "
        "Reach the green exit before the time runs out. Avoid unseen traps."
    )

    game_description = (
        "A top-down horror survival game. Navigate a dark, procedurally generated "
        "labyrinth to find the exit. Your vision is limited, and deadly traps are "
        "hidden in the darkness. Each step is a choice between safety and the unknown."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.TRAP_DENSITY = 0.4  # % of dead ends that become traps

        # --- Colors ---
        self.COLOR_BG = (10, 10, 15)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_FLOOR_LIT = (70, 70, 80)
        self.COLOR_FLOOR_REVEALED = (25, 25, 35)
        self.COLOR_PLAYER = (220, 220, 255)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TRAP_FLASH = (255, 20, 20)
        self.COLOR_TEXT = (200, 200, 200)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.trap_locations = None
        self.revealed_tiles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.triggered_trap_pos = None
        self.light_radius = 4
        
        # This seed is for the environment's random number generator
        self.np_random = None

        # --- Pre-rendered surfaces for effects ---
        self._player_light_surface = self._create_light_surface(self.light_radius * self.CELL_SIZE, self.COLOR_PLAYER)
        self._exit_light_surface = self._create_light_surface(int(self.CELL_SIZE * 2.5), self.COLOR_EXIT)
        self._trap_light_surface = self._create_light_surface(self.CELL_SIZE * 2, self.COLOR_TRAP_FLASH)

        self.validate_implementation()
        self.reset()

    def _create_light_surface(self, radius, color):
        """Creates a circular, fading light surface for blending."""
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -2):
            alpha = 255 * (1 - (i / radius))**2
            alpha = max(0, min(255, int(alpha * 0.4))) # Adjust intensity
            pygame.gfxdraw.filled_circle(surface, radius, radius, i, (*color, alpha))
        return surface

    def _generate_maze_and_objects(self):
        """Generates the maze, finds start/exit, and places traps."""
        # 1. Initialize grid with walls
        self.maze = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.uint8)
        
        # 2. Use recursive backtracking (DFS) to carve paths
        stack = deque()
        start_cell = (1, 1)
        self.player_pos = list(start_cell)
        self.maze[start_cell] = 0
        stack.append(start_cell)

        while stack:
            current_x, current_y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = current_x + dx, current_y + dy
                if 0 < nx < self.GRID_WIDTH -1 and 0 < ny < self.GRID_HEIGHT -1 and self.maze[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_x, next_y = self.np_random.choice(neighbors, axis=0)
                wall_x, wall_y = (current_x + next_x) // 2, (current_y + next_y) // 2
                self.maze[wall_x, wall_y] = 0
                self.maze[next_x, next_y] = 0
                stack.append((next_x, next_y))
            else:
                stack.pop()

        # 3. Find the furthest point from the start for the exit using BFS
        queue = deque([(start_cell, 0)])
        visited = {start_cell}
        furthest_cell, max_dist = start_cell, 0
        
        path_cells = np.argwhere(self.maze == 0)
        adj = {tuple(cell): [] for cell in path_cells}
        for x, y in path_cells:
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                if self.maze[x+dx, y+dy] == 0:
                    adj[(x,y)].append((x+dx, y+dy))

        while queue:
            (x, y), dist = queue.popleft()
            if dist > max_dist:
                max_dist = dist
                furthest_cell = (x, y)
            for neighbor in adj.get((x,y), []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        self.exit_pos = furthest_cell

        # 4. Identify dead ends and place traps
        self.trap_locations = set()
        dead_ends = []
        for x, y in path_cells:
            if (x, y) != start_cell and (x, y) != self.exit_pos:
                if len(adj.get((x,y), [])) == 1:
                    dead_ends.append((x,y))
        
        num_traps = int(len(dead_ends) * self.TRAP_DENSITY)
        trap_indices = self.np_random.choice(len(dead_ends), num_traps, replace=False)
        for i in trap_indices:
            self.trap_locations.add(dead_ends[i])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze_and_objects()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.triggered_trap_pos = None
        
        self.revealed_tiles = set()
        self._update_revealed_tiles()
        
        return self._get_observation(), self._get_info()

    def _update_revealed_tiles(self):
        px, py = self.player_pos
        for x in range(px - self.light_radius, px + self.light_radius + 1):
            for y in range(py - self.light_radius, py + self.light_radius + 1):
                if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist <= self.light_radius:
                        self.revealed_tiles.add((x, y))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = -0.1  # Small penalty for each step

        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            next_x, next_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if self.maze[next_x, next_y] == 0:  # Check for wall collision
                self.player_pos = [next_x, next_y]
                self._update_revealed_tiles()
        
        # --- Check for game events ---
        player_pos_tuple = tuple(self.player_pos)
        
        if player_pos_tuple == self.exit_pos:
            reward += 10.0
            self.score += reward
            self.game_over = True
            self.win = True
            # Sound: victory fanfare
        elif player_pos_tuple in self.trap_locations:
            reward -= 10.0
            self.score += reward
            self.game_over = True
            self.triggered_trap_pos = player_pos_tuple
            # Sound: trap spring, player scream
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            # Sound: timeout buzzer

        if movement != 0:
            self.score += reward

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Maze with Fog of War ---
        px, py = self.player_pos
        for x, y in self.revealed_tiles:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            dist_from_player = math.sqrt((x - px)**2 + (y - py)**2)
            
            color = self.COLOR_WALL if self.maze[x, y] == 1 else self.COLOR_FLOOR_REVEALED
            
            if dist_from_player <= self.light_radius:
                if self.maze[x, y] == 0:
                    color = self.COLOR_FLOOR_LIT

            pygame.draw.rect(self.screen, color, rect)

        # --- Draw Exit ---
        if tuple(self.exit_pos) in self.revealed_tiles:
            ex, ey = self.exit_pos
            rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
            
            # Exit glow effect
            light_pos = (rect.centerx - self._exit_light_surface.get_width() // 2, 
                         rect.centery - self._exit_light_surface.get_height() // 2)
            self.screen.blit(self._exit_light_surface, light_pos, special_flags=pygame.BLEND_RGBA_ADD)

        # --- Draw Player and Light ---
        player_center_x = int((self.player_pos[0] + 0.5) * self.CELL_SIZE)
        player_center_y = int((self.player_pos[1] + 0.5) * self.CELL_SIZE)
        
        # Player light effect
        light_pos = (player_center_x - self._player_light_surface.get_width() // 2, 
                     player_center_y - self._player_light_surface.get_height() // 2)
        self.screen.blit(self._player_light_surface, light_pos, special_flags=pygame.BLEND_RGBA_ADD)

        # Player circle
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, self.CELL_SIZE // 3, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, self.CELL_SIZE // 3, self.COLOR_PLAYER)

        # --- Draw Triggered Trap Effect ---
        if self.triggered_trap_pos:
            tx, ty = self.triggered_trap_pos
            trap_center_x = int((tx + 0.5) * self.CELL_SIZE)
            trap_center_y = int((ty + 0.5) * self.CELL_SIZE)
            
            # Pulsating red glow
            pulse = abs(math.sin(self.steps * 0.5))
            radius = int(self.CELL_SIZE * (0.5 + pulse * 0.5))
            alpha = int(100 + pulse * 100)
            pygame.gfxdraw.filled_circle(self.screen, trap_center_x, trap_center_y, radius, (*self.COLOR_TRAP_FLASH, alpha))


    def _render_ui(self):
        # --- Draw Timer/Steps ---
        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        text_surface = self.font_small.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # --- Draw Score ---
        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

        # --- Draw Game Over/Win Message ---
        if self.game_over:
            if self.win:
                msg = "EXIT REACHED"
                color = self.COLOR_EXIT
            else:
                msg = "YOU DIED"
                color = self.COLOR_TRAP_FLASH
            
            end_text_surface = self.font_large.render(msg, True, color)
            end_text_rect = end_text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surface, end_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Labyrinth Horror")
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print("      LABYRINTH HORROR")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")

        if not terminated:
            keys = pygame.key.get_pressed()
            moved = False
            if keys[pygame.K_UP]:
                action[0] = 1
                moved = True
            elif keys[pygame.K_DOWN]:
                action[0] = 2
                moved = True
            elif keys[pygame.K_LEFT]:
                action[0] = 3
                moved = True
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
                moved = True
            
            if moved: # Only step if a key was pressed
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(15) # Limit player input speed

    env.close()