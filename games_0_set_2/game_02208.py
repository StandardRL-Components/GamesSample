
# Generated: 2025-08-27T19:37:21.677545
# Source Brief: brief_02208.md
# Brief Index: 2208

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. Your goal is to reach the glowing white exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A chilling horror maze. Navigate in the dark with limited visibility, avoid lurking enemies, and find the exit before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 31  # Must be odd
    MAZE_HEIGHT = 19 # Must be odd
    CELL_SIZE = 20
    
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLAYER = (100, 150, 255)
    COLOR_PLAYER_GLOW = (50, 75, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (128, 25, 25)
    COLOR_EXIT = (255, 255, 255)
    COLOR_EXIT_GLOW = (150, 150, 150)
    COLOR_UI_TEXT = (220, 220, 220)

    MAX_STEPS = 1000
    NUM_ENEMIES = 7
    VISIBILITY_RADIUS = 3.5

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.maze = None
        self.player_pos = None
        self.enemies = None
        self.exit_pos = None
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()
    
    def _generate_maze(self):
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        stack = []
        
        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.MAZE_WIDTH -1 and 0 < ny < self.MAZE_HEIGHT -1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice([tuple(n) for n in neighbors])
                wall_x, wall_y = cx + (nx - cx) // 2, cy + (ny - cy) // 2
                maze[wall_y, wall_x] = 0
                maze[ny, nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_furthest_point(self, start_pos):
        q = deque([(start_pos, 0)])
        visited = {start_pos}
        farthest_node, max_dist = start_pos, 0

        while q:
            (cx, cy), dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_node = (cx, cy)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        return farthest_node
    
    def _find_enemy_paths(self):
        paths = []
        possible_loops = []
        for y in range(1, self.MAZE_HEIGHT - 2):
            for x in range(1, self.MAZE_WIDTH - 2):
                if self.maze[y,x]==0 and self.maze[y+1,x]==0 and self.maze[y,x+1]==0 and self.maze[y+1,x+1]==0:
                    dist_to_player = abs(x - self.player_pos[0]) + abs(y - self.player_pos[1])
                    if dist_to_player > 8: # Don't spawn enemies too close
                        possible_loops.append((x, y))

        self.np_random.shuffle(possible_loops)
        
        for i in range(min(self.NUM_ENEMIES, len(possible_loops))):
            x, y = possible_loops[i]
            path = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
            start_index = self.np_random.integers(0, len(path))
            direction = self.np_random.choice([-1, 1])
            paths.append({"path": path, "index": start_index, "direction": direction, "pos": path[start_index]})
        return paths

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = (1, 1)
        self.maze = self._generate_maze()
        self.exit_pos = self._find_furthest_point(self.player_pos)
        self.enemies = self._find_enemy_paths()
        
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False
        self.win = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01 # Small cost for each step
        terminated = False

        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        if self.maze[py, px] == 0:
            self.player_pos = (px, py)

        # --- Enemy Movement ---
        for enemy in self.enemies:
            enemy["index"] = (enemy["index"] + enemy["direction"]) % len(enemy["path"])
            enemy["pos"] = enemy["path"][enemy["index"]]

        self.steps += 1
        
        # --- Check Termination Conditions ---
        if self.player_pos == self.exit_pos:
            reward = 10.0
            terminated = True
            self.game_over = True
            self.win = True
            # sfx: win_sound
        
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                reward = -10.0
                terminated = True
                self.game_over = True
                self.win = False
                # sfx: player_death
                break
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.cumulative_reward += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_glow_circle(self, surface, pos, radius, color, glow_color):
        # Draw outer glow
        for i in range(radius, 0, -2):
            alpha = int(100 * (1 - i / radius))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], i + radius, (*glow_color, alpha))
        
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _render_flickering_triangle(self, surface, pos, size, color, glow_color):
        flicker = (math.sin(self.steps * 0.5) + 1) / 2 # Varies between 0 and 1
        
        # Glow
        glow_alpha = int(50 + 100 * flicker)
        glow_size = int(size * 1.8)
        points_glow = [
            (pos[0], pos[1] - glow_size),
            (pos[0] - glow_size, pos[1] + glow_size),
            (pos[0] + glow_size, pos[1] + glow_size)
        ]
        pygame.gfxdraw.filled_trigon(surface, *points_glow[0], *points_glow[1], *points_glow[2], (*glow_color, glow_alpha))

        # Main shape
        main_color = tuple(int(c * (0.8 + 0.2 * flicker)) for c in color)
        points = [
            (pos[0], pos[1] - size),
            (pos[0] - size, pos[1] + size),
            (pos[0] + size, pos[1] + size)
        ]
        pygame.gfxdraw.aapolygon(surface, points, main_color)
        pygame.gfxdraw.filled_polygon(surface, points, main_color)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        
        player_gx, player_gy = self.player_pos
        
        # Render maze, exit, and enemies within visibility
        for gy in range(self.MAZE_HEIGHT):
            for gx in range(self.MAZE_WIDTH):
                dist_sq = (gx - player_gx)**2 + (gy - player_gy)**2
                if dist_sq > self.VISIBILITY_RADIUS**2:
                    continue

                screen_x = center_x + (gx - player_gx) * self.CELL_SIZE
                screen_y = center_y + (gy - player_gy) * self.CELL_SIZE
                
                rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
                
                if self.maze[gy, gx] == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else: # Path
                    if (gx, gy) == self.exit_pos:
                        pos = (rect.centerx, rect.centery)
                        self._render_glow_circle(self.screen, pos, self.CELL_SIZE // 3, self.COLOR_EXIT, self.COLOR_EXIT_GLOW)
        
        # Render enemies within visibility
        for enemy in self.enemies:
            gx, gy = enemy["pos"]
            dist_sq = (gx - player_gx)**2 + (gy - player_gy)**2
            if dist_sq <= self.VISIBILITY_RADIUS**2:
                screen_x = center_x + (gx - player_gx) * self.CELL_SIZE
                screen_y = center_y + (gy - player_gy) * self.CELL_SIZE
                pos = (screen_x + self.CELL_SIZE // 2, screen_y + self.CELL_SIZE // 2)
                self._render_flickering_triangle(self.screen, pos, self.CELL_SIZE // 4, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
        
        # Render Player (always in center)
        player_screen_pos = (center_x + self.CELL_SIZE // 2, center_y + self.CELL_SIZE // 2)
        self._render_glow_circle(self.screen, player_screen_pos, self.CELL_SIZE // 3, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Render UI
        time_left = self.MAX_STEPS - self.steps
        timer_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU ESCAPED" if self.win else "YOU DIED"
            color = (180, 255, 180) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.cumulative_reward,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a window
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'dummy' for headless

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Horror")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if action[0] != 0: # Only step if a move is attempted
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Convert numpy array back to a surface for display
            display_obs = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(display_obs)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}")
                print("Press 'R' to reset.")
        
        clock.tick(10) # Limit frame rate for human play

    env.close()