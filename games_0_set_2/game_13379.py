import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:08:49.615629
# Source Brief: brief_03379.md
# Brief Index: 3379
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated maze, collect all the keys, and reach the exit before time runs out. "
        "Build momentum by moving in the same direction to travel faster."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your character through the maze."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 31  # Must be odd
    MAZE_HEIGHT = 19 # Must be odd
    TILE_SIZE = min(SCREEN_WIDTH // (MAZE_WIDTH + 1), SCREEN_HEIGHT // (MAZE_HEIGHT + 1))
    OFFSET_X = (SCREEN_WIDTH - MAZE_WIDTH * TILE_SIZE) // 2
    OFFSET_Y = (SCREEN_HEIGHT - MAZE_HEIGHT * TILE_SIZE) // 2

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 50, 90)
    COLOR_PATH = (25, 30, 45)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 80, 80, 60)
    COLOR_KEY = (255, 220, 50)
    COLOR_KEY_GLOW = (255, 220, 50, 80)
    COLOR_EXIT = (80, 255, 120)
    COLOR_EXIT_GLOW = (80, 255, 120, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIME_BAR_GOOD = (80, 255, 120)
    COLOR_TIME_BAR_BAD = (255, 80, 80)

    # Game Parameters
    NUM_KEYS = 5
    INITIAL_TIME = 120.0
    MAX_STEPS = 1000
    MAX_MOMENTUM = 3
    TRAIL_LENGTH = 10

    # Reward Structure
    REWARD_WIN = 100.0
    REWARD_LOSS_TIME = -50.0
    REWARD_KEY = 2.0
    REWARD_STEP = -0.1

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
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)

        self.maze = None
        self.player_pos = None
        self.key_locations = None
        self.exit_pos = None
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.keys_collected = None
        self.momentum = None
        self.last_move_direction = None
        self.player_trail = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.time_remaining = self.INITIAL_TIME
        self.keys_collected = 0
        self.momentum = 1
        self.last_move_direction = 0 # 0 for none
        self.player_trail = deque(maxlen=self.TRAIL_LENGTH)
        
        self._generate_maze_and_entities()
        self.player_trail.append(self.player_pos)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        movement = action[0]
        reward = self.REWARD_STEP
        self.steps += 1

        # --- Handle Movement & Collision ---
        if movement != 0:
            if movement == self.last_move_direction:
                self.momentum = min(self.MAX_MOMENTUM, self.momentum + 1)
            else:
                self.momentum = 1
            self.last_move_direction = movement
            
            # # Sound placeholder:
            # if self.momentum > 1: print("SFX: whoosh")
            # else: print("SFX: step")

            # Move one tile at a time to check for collisions
            current_pos = list(self.player_pos)
            for _ in range(self.momentum):
                next_pos = list(current_pos)
                if movement == 1: next_pos[1] -= 1  # Up
                elif movement == 2: next_pos[1] += 1 # Down
                elif movement == 3: next_pos[0] -= 1 # Left
                elif movement == 4: next_pos[0] += 1 # Right

                if self.maze[next_pos[1]][next_pos[0]] == 1: # Wall collision
                    self.momentum = 1 # Reset momentum on hit
                    # # Sound placeholder: print("SFX: thump")
                    break
                else:
                    current_pos = next_pos
            
            self.player_pos = tuple(current_pos)
            self.player_trail.append(self.player_pos)

        # --- Handle Key Collection ---
        if self.player_pos in self.key_locations:
            self.key_locations.remove(self.player_pos)
            self.keys_collected += 1
            reward += self.REWARD_KEY
            # # Sound placeholder: print("SFX: key_pickup")

        # --- Update Time ---
        time_penalty_reduction = 0.02 * self.keys_collected
        time_penalty_per_step = (1.0 / self.INITIAL_TIME) * 10 * (1.0 - time_penalty_reduction)
        self.time_remaining -= time_penalty_per_step * 10 # Scale up penalty to be noticeable

        # --- Check for Termination ---
        terminated = False
        if self.player_pos == self.exit_pos and self.keys_collected == self.NUM_KEYS:
            reward += self.REWARD_WIN
            terminated = True
            # # Sound placeholder: print("SFX: victory_jingle")
        elif self.time_remaining <= 0:
            reward += self.REWARD_LOSS_TIME
            terminated = True
            # # Sound placeholder: print("SFX: loss_buzzer")
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        self.score += reward
        
        obs = self._get_observation()
        info = self._get_info()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_maze()
        self._render_keys_and_exit()
        self._render_trail()
        self._render_player()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "keys": self.keys_collected}

    def close(self):
        pygame.quit()

    # --- Helper Methods ---

    def _generate_maze_and_entities(self):
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        start_x, start_y = (1, 1)
        self.maze[start_y, start_x] = 0
        
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH - 1 and 0 < ny < self.MAZE_HEIGHT - 1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                visited.add((nx, ny))
                self.maze[ny, nx] = 0
                self.maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        self.player_pos = (start_x, start_y)
        
        path_cells = np.argwhere(self.maze == 0)
        path_cells = [tuple(cell[::-1]) for cell in path_cells]

        # Find furthest cell for exit
        self.exit_pos = self._find_furthest_cell(self.player_pos)
        
        # Find dead ends for keys
        dead_ends = self._find_dead_ends()
        random.shuffle(dead_ends)
        
        # Place keys, ensuring they are not at start or exit
        potential_key_locs = [loc for loc in dead_ends if loc != self.player_pos and loc != self.exit_pos]
        
        # If not enough dead ends, use other distant path cells
        if len(potential_key_locs) < self.NUM_KEYS:
            other_locs = [loc for loc in path_cells if loc not in potential_key_locs and loc != self.player_pos and loc != self.exit_pos]
            random.shuffle(other_locs)
            potential_key_locs.extend(other_locs)

        self.key_locations = potential_key_locs[:self.NUM_KEYS]

    def _find_furthest_cell(self, start_node):
        q = deque([(start_node, 0)])
        visited = {start_node}
        furthest_node, max_dist = start_node, 0

        while q:
            (cx, cy), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                furthest_node = (cx, cy)
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if self.maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        return furthest_node

    def _find_dead_ends(self):
        dead_ends = []
        for y in range(1, self.MAZE_HEIGHT - 1):
            for x in range(1, self.MAZE_WIDTH - 1):
                if self.maze[y, x] == 0:
                    neighbors = 0
                    if self.maze[y-1, x] == 0: neighbors += 1
                    if self.maze[y+1, x] == 0: neighbors += 1
                    if self.maze[y, x-1] == 0: neighbors += 1
                    if self.maze[y, x+1] == 0: neighbors += 1
                    if neighbors == 1:
                        dead_ends.append((x, y))
        return dead_ends

    # --- Rendering Methods ---

    def _render_maze(self):
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                rect = pygame.Rect(self.OFFSET_X + x * self.TILE_SIZE,
                                   self.OFFSET_Y + y * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

    def _render_keys_and_exit(self):
        # Exit
        ex, ey = self.exit_pos
        exit_center_x = int(self.OFFSET_X + ex * self.TILE_SIZE + self.TILE_SIZE / 2)
        exit_center_y = int(self.OFFSET_Y + ey * self.TILE_SIZE + self.TILE_SIZE / 2)
        glow_radius = int(self.TILE_SIZE * 0.7)
        main_radius = int(self.TILE_SIZE * 0.35)
        
        color = self.COLOR_EXIT if self.keys_collected == self.NUM_KEYS else self.COLOR_WALL
        glow_color = self.COLOR_EXIT_GLOW if self.keys_collected == self.NUM_KEYS else (100,100,100,50)
        
        pygame.gfxdraw.filled_circle(self.screen, exit_center_x, exit_center_y, glow_radius, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, exit_center_x, exit_center_y, main_radius, color)
        pygame.gfxdraw.aacircle(self.screen, exit_center_x, exit_center_y, main_radius, color)

        # Keys
        for kx, ky in self.key_locations:
            key_center_x = int(self.OFFSET_X + kx * self.TILE_SIZE + self.TILE_SIZE / 2)
            key_center_y = int(self.OFFSET_Y + ky * self.TILE_SIZE + self.TILE_SIZE / 2)
            pygame.gfxdraw.filled_circle(self.screen, key_center_x, key_center_y, int(self.TILE_SIZE * 0.5), self.COLOR_KEY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, key_center_x, key_center_y, int(self.TILE_SIZE * 0.25), self.COLOR_KEY)
            pygame.gfxdraw.aacircle(self.screen, key_center_x, key_center_y, int(self.TILE_SIZE * 0.25), self.COLOR_KEY)

    def _render_trail(self):
        trail_positions = list(self.player_trail)
        for i, pos in enumerate(trail_positions):
            alpha = int(150 * (i / len(trail_positions)))
            color = (*self.COLOR_PLAYER[:3], alpha)
            radius = int(self.TILE_SIZE * 0.3 * (i / len(trail_positions)))
            px = int(self.OFFSET_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
            py = int(self.OFFSET_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)

    def _render_player(self):
        px, py = self.player_pos
        player_center_x = int(self.OFFSET_X + px * self.TILE_SIZE + self.TILE_SIZE / 2)
        player_center_y = int(self.OFFSET_Y + py * self.TILE_SIZE + self.TILE_SIZE / 2)
        
        glow_size = int(self.TILE_SIZE * 0.8)
        player_size = int(self.TILE_SIZE * 0.6)
        
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = (player_center_x, player_center_y)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=3)
        self.screen.blit(glow_surf, glow_rect)

        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = (player_center_x, player_center_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
    def _render_ui(self):
        # Time Bar
        time_ratio = max(0, self.time_remaining / self.INITIAL_TIME)
        bar_width = self.SCREEN_WIDTH * 0.5
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 10
        
        current_color = [
            int(c1 * time_ratio + c2 * (1 - time_ratio))
            for c1, c2 in zip(self.COLOR_TIME_BAR_GOOD, self.COLOR_TIME_BAR_BAD)
        ]

        pygame.draw.rect(self.screen, self.COLOR_WALL, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, current_color, (bar_x, bar_y, bar_width * time_ratio, bar_height), border_radius=4)
        
        time_text = self.font_small.render(f"TIME: {int(self.time_remaining)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (bar_x + bar_width + 10, bar_y))

        # Keys collected
        key_text = self.font_large.render(f"KEYS: {self.keys_collected} / {self.NUM_KEYS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(key_text, (self.OFFSET_X, self.SCREEN_HEIGHT - 35))
        
        # Momentum indicator
        momentum_text = self.font_small.render(f"MOMENTUM: x{self.momentum}", True, self.COLOR_UI_TEXT)
        text_rect = momentum_text.get_rect(topright=(self.SCREEN_WIDTH - self.OFFSET_X, self.SCREEN_HEIGHT - 30))
        self.screen.blit(momentum_text, text_rect)

if __name__ == '__main__':
    # --- Example usage and interactive play ---
    # This block is not part of the Gymnasium environment but is useful for testing.
    # We re-enable the display for interactive mode.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Interactive Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    print("--------------------------\n")

    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        manual_action = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                manual_action = True
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Env Reset ---")
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_n:
                    action[0] = 0

        if manual_action:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']:.2f}, Final Keys: {info['keys']}")
                obs, info = env.reset()
                total_reward = 0
        
        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(15) # Control interactive play speed

    env.close()