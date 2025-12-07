
# Generated: 2025-08-27T20:11:17.391170
# Source Brief: brief_02378.md
# Brief Index: 2378

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Collect all the yellow gems while avoiding the red mines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated maze puzzle. Collect 15 gems to win, "
        "but you only have 500 moves. Hitting a mine ends the game immediately."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.GEMS_TO_COLLECT = 15
        self.MAX_MOVES = 500
        self.SAFE_ZONE_STEPS = 50

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_EXPLOSION = (255, 100, 0)
        self.COLOR_TEXT = (240, 240, 240)
        
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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)
            
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.player_pos = None
        self.gem_locations = None
        self.mine_locations = None
        
        self.steps = 0
        self.moves_remaining = 0
        self.gems_collected = 0
        self.score = 0
        self.terminated = False
        self.explosion_state = None  # (pos, timer)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.moves_remaining = self.MAX_MOVES
        self.terminated = False
        self.explosion_state = None
        
        self._generate_maze_and_entities()

        return self._get_observation(), self._get_info()

    def _generate_maze_and_entities(self):
        # 1. Initialize grid with walls
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        
        # 2. Use Randomized DFS to carve paths
        # We operate on a meta-grid to ensure walls between paths
        stack = deque()
        start_cell = (1, 1)
        self.grid[start_cell] = 0
        stack.append(start_cell)

        while stack:
            cy, cx = stack[-1]
            neighbors = []
            for dy, dx in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 < ny < self.GRID_HEIGHT -1 and 0 < nx < self.GRID_WIDTH -1 and self.grid[ny, nx] == 1:
                    neighbors.append((ny, nx))
            
            if neighbors:
                ny, nx = self.np_random.choice(neighbors, axis=0)
                # Carve path to neighbor
                self.grid[ny, nx] = 0
                self.grid[(cy + ny) // 2, (cx + nx) // 2] = 0
                stack.append((ny, nx))
            else:
                stack.pop()
        
        # 3. Get all valid path locations
        path_locations = list(zip(*np.where(self.grid == 0)))
        self.np_random.shuffle(path_locations)

        # 4. Place player
        self.player_pos = path_locations.pop()

        # 5. Determine safe zone around player using BFS
        q = deque([(self.player_pos, 0)])
        visited = {self.player_pos}
        safe_zone = {self.player_pos}
        
        while q:
            (y, x), dist = q.popleft()
            if dist >= self.SAFE_ZONE_STEPS:
                continue
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and self.grid[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    safe_zone.add((ny, nx))
                    q.append(((ny, nx), dist + 1))

        # 6. Place gems and mines
        potential_mine_locations = [loc for loc in path_locations if loc not in safe_zone]
        
        # Ensure we have enough spots for gems and mines
        available_spots = [loc for loc in path_locations if loc != self.player_pos]
        self.np_random.shuffle(available_spots)
        
        num_mines = self.np_random.integers(10, 21) # Random number of mines
        self.gem_locations = set(available_spots[:self.GEMS_TO_COLLECT])
        
        mine_placement_candidates = [loc for loc in available_spots if loc not in self.gem_locations]
        self.mine_locations = set(mine_placement_candidates[:num_mines])


    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()
            
        reward = 0
        
        # Handle explosion animation countdown
        if self.explosion_state:
            pos, timer = self.explosion_state
            timer -= 1
            if timer <= 0:
                self.explosion_state = None
                self.terminated = True
            else:
                self.explosion_state = (pos, timer)
            # Return after processing animation frame, no other logic
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1 (unused)
        # shift_held = action[2] == 1 (unused)

        # Apply action if not a no-op
        if movement != 0:
            self.moves_remaining -= 1
            reward += 0.1  # Survival reward

            py, px = self.player_pos
            dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(movement, (0, 0))
            ny, nx = py + dy, px + dx

            # Check for wall collision
            if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and self.grid[ny, nx] == 0:
                self.player_pos = (ny, nx)

        # Check for events at the new position
        if self.player_pos in self.gem_locations:
            # sfx: gem collect
            self.gem_locations.remove(self.player_pos)
            self.gems_collected += 1
            reward += 10.0
            if self.gems_collected >= self.GEMS_TO_COLLECT:
                # sfx: victory fanfare
                reward += 50.0
                self.terminated = True

        if self.player_pos in self.mine_locations:
            # sfx: explosion
            reward -= 50.0
            self.explosion_state = (self.player_pos, 3) # 3-frame explosion
            self.mine_locations.remove(self.player_pos) # prevent re-triggering

        # Check for termination by running out of moves
        if self.moves_remaining <= 0 and not self.terminated:
            # sfx: failure sound
            self.terminated = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
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
        # Draw walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_WALL,
                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    )
        
        # Draw mines
        for y, x in self.mine_locations:
            points = [
                (x * self.CELL_SIZE + self.CELL_SIZE / 2, y * self.CELL_SIZE + 2),
                ((x + 1) * self.CELL_SIZE - 2, (y + 1) * self.CELL_SIZE - 2),
                (x * self.CELL_SIZE + 2, (y + 1) * self.CELL_SIZE - 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINE)

        # Draw gems
        for y, x in self.gem_locations:
            center_x = int(x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(y * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = int(self.CELL_SIZE / 2.5)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GEM)

        # Draw player
        py, px = self.player_pos
        player_rect = pygame.Rect(
            px * self.CELL_SIZE + 3, 
            py * self.CELL_SIZE + 3, 
            self.CELL_SIZE - 6, 
            self.CELL_SIZE - 6
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw explosion
        if self.explosion_state:
            (y, x), timer = self.explosion_state
            center_x = int(x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(y * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = int((4 - timer) * self.CELL_SIZE / 2)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_EXPLOSION)

    def _render_ui(self):
        # Gem counter
        gem_text = self.font_ui.render(f"GEMS: {self.gems_collected}/{self.GEMS_TO_COLLECT}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 5))

        # Moves remaining
        moves_text = self.font_ui.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 10, 5))
        self.screen.blit(moves_text, moves_rect)
        
        # Game over message
        if self.terminated and not self.explosion_state:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.gems_collected >= self.GEMS_TO_COLLECT:
                msg = "YOU WIN!"
                color = self.COLOR_GEM
            else:
                msg = "GAME OVER"
                color = self.COLOR_MINE
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_remaining": self.moves_remaining,
        }

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    while True:
        action = np.array([0, 0, 0])  # Default to no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
        
        # If an action was chosen, step the environment
        if action[0] != 0 or terminated:
             obs, reward, terminated, truncated, info = env.step(action)
             print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate