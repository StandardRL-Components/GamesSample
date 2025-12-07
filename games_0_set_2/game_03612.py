
# Generated: 2025-08-27T23:53:01.552252
# Source Brief: brief_03612.md
# Brief Index: 3612

        
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
        "Controls: Use arrow keys to navigate the maze. Reach the green exit tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield to reach the exit. Each step costs points, but reaching the exit gives a large bonus. Hitting a mine ends the game. Take calculated risks for a higher score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 500

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_WALL = (80, 80, 100)
    COLOR_PATH = (30, 30, 45)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_EXIT = (50, 255, 150)
    COLOR_MINE = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # Calculate grid rendering properties
        self.tile_size = min((self.SCREEN_HEIGHT - 80) // self.GRID_HEIGHT, (self.SCREEN_WIDTH - 80) // self.GRID_WIDTH)
        self.grid_render_width = self.GRID_WIDTH * self.tile_size
        self.grid_render_height = self.GRID_HEIGHT * self.tile_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_height) // 2
        
        # Game state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.difficulty_level = 0
        self.player_pos = (0, 0)
        self.last_player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.mine_positions = set()
        self.visited_tiles = set()
        self.walls = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT, 4), dtype=bool) # N, E, S, W
        
        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        self.walls.fill(True)
        visited = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True

        while stack:
            x, y = stack[-1]
            neighbors = []
            # Check North
            if y > 0 and not visited[x, y - 1]: neighbors.append((x, y - 1, 0, 2)) # Pos, WallIdx_curr, WallIdx_neighbor
            # Check East
            if x < self.GRID_WIDTH - 1 and not visited[x + 1, y]: neighbors.append((x + 1, y, 1, 3))
            # Check South
            if y < self.GRID_HEIGHT - 1 and not visited[x, y + 1]: neighbors.append((x, y + 1, 2, 0))
            # Check West
            if x > 0 and not visited[x - 1, y]: neighbors.append((x - 1, y, 3, 1))

            if neighbors:
                nx, ny, wall_idx, neighbor_wall_idx = self.np_random.choice([tuple(n) for n in neighbors])
                self.walls[x, y, wall_idx] = False
                self.walls[nx, ny, neighbor_wall_idx] = False
                visited[nx, ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_entities(self):
        num_mines = 10 + 2 * self.difficulty_level
        self.player_pos = (0, 0)
        self.exit_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)
        
        possible_mine_locs = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        possible_mine_locs.discard(self.player_pos)
        possible_mine_locs.discard(self.exit_pos)
        # Avoid placing mines right next to the start
        possible_mine_locs.discard((1,0))
        possible_mine_locs.discard((0,1))
        
        self.mine_positions = set(random.sample(sorted(list(possible_mine_locs)), k=min(num_mines, len(possible_mine_locs))))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        if self.win: # Increase difficulty on successful completion
            self.difficulty_level += 1
        else: # Reset difficulty on failure
            self.difficulty_level = 0
            
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self._generate_maze()
        self._place_entities()
        
        self.last_player_pos = self.player_pos
        self.visited_tiles = {self.player_pos}
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = -0.1 # Cost per step
        
        self.last_player_pos = self.player_pos
        px, py = self.player_pos
        
        if movement == 1 and not self.walls[px, py, 0]: # Up
            py -= 1
        elif movement == 2 and not self.walls[px, py, 2]: # Down
            py += 1
        elif movement == 3 and not self.walls[px, py, 3]: # Left
            px -= 1
        elif movement == 4 and not self.walls[px, py, 1]: # Right
            px += 1
            
        self.player_pos = (max(0, min(self.GRID_WIDTH - 1, px)), max(0, min(self.GRID_HEIGHT - 1, py)))
        self.visited_tiles.add(self.player_pos)
        
        # Reward for moving closer to exit
        dist_before = self._manhattan_distance(self.last_player_pos, self.exit_pos)
        dist_after = self._manhattan_distance(self.player_pos, self.exit_pos)
        if dist_after > dist_before:
            reward -= 2.0
            
        # Reward for avoiding nearby mines
        if self.player_pos != self.last_player_pos:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    neighbor = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                    if neighbor in self.mine_positions:
                        reward += 5.0
                        
        self.steps += 1
        terminated = False

        if self.player_pos in self.mine_positions:
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.win = False
        elif self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = False
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _grid_to_pixel(self, x, y):
        px = self.grid_offset_x + x * self.tile_size
        py = self.grid_offset_y + y * self.tile_size
        return px, py

    def _render_text(self, text, x, y, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _render_maze(self):
        # Draw visited path
        for (x, y) in self.visited_tiles:
            px, py = self._grid_to_pixel(x, y)
            pygame.draw.rect(self.screen, self.COLOR_PATH, (px, py, self.tile_size, self.tile_size))
            
        # Draw walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                px, py = self._grid_to_pixel(x, y)
                if self.walls[x, y, 0]: # North
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.tile_size, py), 2)
                if self.walls[x, y, 1]: # East
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.tile_size, py), (px + self.tile_size, py + self.tile_size), 2)
                if self.walls[x, y, 2]: # South
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.tile_size), (px + self.tile_size, py + self.tile_size), 2)
                if self.walls[x, y, 3]: # West
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.tile_size), 2)
    
    def _render_entities(self):
        # Draw Exit
        ex, ey = self._grid_to_pixel(self.exit_pos[0], self.exit_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex + 4, ey + 4, self.tile_size - 8, self.tile_size - 8))
        
        # Draw Player
        px, py = self._grid_to_pixel(self.player_pos[0], self.player_pos[1])
        center_x = px + self.tile_size // 2
        center_y = py + self.tile_size // 2
        radius = self.tile_size // 3
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        
        # Draw Mines if game over from a mine
        if self.game_over and not self.win:
            for (mx, my) in self.mine_positions:
                mpx, mpy = self._grid_to_pixel(mx, my)
                center_x = mpx + self.tile_size // 2
                center_y = mpy + self.tile_size // 2
                half_size = self.tile_size // 4
                points = [
                    (center_x, center_y - half_size),
                    (center_x - half_size, center_y + half_size),
                    (center_x + half_size, center_y + half_size)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINE)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINE)

    def _render_ui(self):
        self._render_text(f"SCORE: {self.score:.1f}", 20, 15, self.font_ui)
        self._render_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", self.SCREEN_WIDTH - 180, 15, self.font_ui)
        
        if self.game_over:
            if self.win:
                message = "YOU WIN!"
                color = self.COLOR_EXIT
            else:
                message = "GAME OVER"
                color = self.COLOR_MINE
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surf = self.font_game_over.render(message, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_maze()
        self._render_entities()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "difficulty": self.difficulty_level,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = obs.shape[1], obs.shape[0]
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Minefield Maze")
    
    terminated = False
    running = True
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")
    
    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue
                
                # Map keys to actions
                key_map = {
                    pygame.K_UP: 1,
                    pygame.K_DOWN: 2,
                    pygame.K_LEFT: 3,
                    pygame.K_RIGHT: 4,
                }
                if event.key in key_map:
                    action[0] = key_map[event.key]
                
                # Since auto_advance is False, we step on each key press
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()