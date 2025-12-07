
# Generated: 2025-08-27T23:59:51.882773
# Source Brief: brief_03652.md
# Brief Index: 3652

        
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

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move one step at a time. "
        "Evade the ghost and reach the glowing exit."
    )

    game_description = (
        "Evade a relentless ghost in a procedurally generated haunted mansion. "
        "Navigate the dark, flickering hallways to find the exit before you're caught. "
        "Each stage gets longer and the ghost gets faster."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 40
        self.MAX_STAGES = 3
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_WALL = (40, 35, 50)
        self.COLOR_WALL_TOP = (60, 55, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GHOST = (200, 220, 255)
        self.COLOR_EXIT = (255, 255, 200)
        self.COLOR_TEXT = (230, 230, 230)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 24)
            self.font_small = pygame.font.SysFont("sans", 16)
        
        # Initialize state variables
        self.stage = 1
        self.player_pos = np.array([0, 0])
        self.ghost_pos = np.array([0.0, 0.0])
        self.exit_pos = np.array([0, 0])
        self.maze = np.array([[]])
        self.distance_grid = np.array([[]])
        self.ghost_speed = 0.0
        self.particles = []
        self.last_dist_to_ghost = 0.0
        
        # Initialize state
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def _setup_stage(self):
        """Initializes the maze, player, ghost, and exit for the current stage."""
        # Stage-dependent properties
        maze_dims = {
            1: (15, 11),
            2: (21, 15),
            3: (27, 21),
        }
        ghost_speeds = {1: 1.2, 2: 1.22, 3: 1.24}
        
        self.maze_w, self.maze_h = maze_dims.get(self.stage, (27, 21))
        self.ghost_speed = ghost_speeds.get(self.stage, 1.24)
        
        # Generate maze
        self.maze = self._generate_maze(self.maze_w, self.maze_h)
        
        # Place entities
        self.player_pos = np.array([1, 1])
        self.exit_pos = np.array([self.maze_w - 2, self.maze_h - 2])
        
        # Place ghost at the furthest point from the player
        ghost_start_pos = self._find_furthest_point(self.player_pos)
        self.ghost_pos = ghost_start_pos.astype(float) * self.TILE_SIZE + self.TILE_SIZE / 2
        
        # Pre-calculate path distances to the exit for the UI
        self.distance_grid = self._calculate_path_distances(self.exit_pos)
        
        # Reset dynamic elements
        self.particles = []
        self.last_dist_to_ghost = self._get_ghost_distance()

    def _generate_maze(self, width, height):
        """Generates a maze using recursive backtracking. 1=wall, 0=path."""
        maze = np.ones((height, width), dtype=np.int8)
        stack = [(1, 1)]
        maze[1, 1] = 0
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = self.np_random.choice(neighbors, axis=0)
                maze[ny, nx] = 0
                maze[cy + dy, cx + dx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _bfs(self, start_pos):
        """Helper for pathfinding, returns distances from start."""
        q = deque([(start_pos, 0)])
        visited = {tuple(start_pos)}
        distances = {tuple(start_pos): 0}
        
        while q:
            (x, y), dist = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_w and 0 <= ny < self.maze_h and self.maze[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    distances[(nx, ny)] = dist + 1
                    q.append(((nx, ny), dist + 1))
        return distances

    def _find_furthest_point(self, start_pos):
        """Finds the furthest reachable tile from a starting position."""
        distances = self._bfs(tuple(start_pos))
        furthest_point = max(distances, key=distances.get)
        return np.array(furthest_point)

    def _calculate_path_distances(self, exit_pos):
        """Creates a grid of path distances to the exit for UI display."""
        distances = self._bfs(tuple(exit_pos))
        grid = np.full((self.maze_h, self.maze_w), -1, dtype=int)
        for (x, y), dist in distances.items():
            grid[y, x] = dist
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and "stage" in options:
            self.stage = max(1, min(options["stage"], self.MAX_STAGES))
        else:
            self.stage = 1
            
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        # --- Player Movement ---
        new_px, new_py = self.player_pos[0] + dx, self.player_pos[1] + dy
        if self.maze[new_py, new_px] == 0: # Check for wall collision
            self.player_pos = np.array([new_px, new_py])
            # sfx: player_step.wav

        # --- Ghost Movement ---
        player_center = self.player_pos.astype(float) * self.TILE_SIZE + self.TILE_SIZE / 2
        ghost_dx = player_center[0] - self.ghost_pos[0]
        ghost_dy = player_center[1] - self.ghost_pos[1]
        dist_to_player = math.hypot(ghost_dx, ghost_dy)

        if dist_to_player > 1:
            self.ghost_pos[0] += (ghost_dx / dist_to_player) * self.ghost_speed * (self.TILE_SIZE / 10)
            self.ghost_pos[1] += (ghost_dy / dist_to_player) * self.ghost_speed * (self.TILE_SIZE / 10)
            # sfx: ghost_move.wav (low hum)

        # --- Particle Effects ---
        if self.steps % 2 == 0:
            self.particles.append({
                "pos": self.ghost_pos.copy(),
                "life": 30,
                "size": self.TILE_SIZE * 0.4
            })
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["size"] *= 0.98

        # --- Reward Calculation ---
        reward = 0
        current_dist_to_ghost = self._get_ghost_distance()
        if current_dist_to_ghost > self.last_dist_to_ghost:
            reward += 0.1
        elif current_dist_to_ghost < self.last_dist_to_ghost:
            reward -= 0.2
        self.last_dist_to_ghost = current_dist_to_ghost

        # --- Termination Check ---
        terminated = False
        # 1. Caught by ghost
        if current_dist_to_ghost < self.TILE_SIZE * 0.6:
            terminated = True
            self.game_over = True
            reward = -100.0
            # sfx: player_caught.wav
        
        # 2. Reached exit
        if np.array_equal(self.player_pos, self.exit_pos):
            terminated = True
            self.game_over = True
            reward = 100.0
            # sfx: stage_clear.wav
            if self.stage < self.MAX_STAGES:
                # This would be where you transition to the next stage in a non-gym context
                pass 
        
        # 3. Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_ghost_distance(self):
        return np.linalg.norm(self.player_pos * self.TILE_SIZE - self.ghost_pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera offset to center player
        cam_x = self.WIDTH / 2 - (self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        cam_y = self.HEIGHT / 2 - (self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)

        # Render visible maze
        start_col = max(0, int(-cam_x / self.TILE_SIZE))
        end_col = min(self.maze_w, int((-cam_x + self.WIDTH) / self.TILE_SIZE) + 1)
        start_row = max(0, int(-cam_y / self.TILE_SIZE))
        end_row = min(self.maze_h, int((-cam_y + self.HEIGHT) / self.TILE_SIZE) + 1)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(cam_x + x * self.TILE_SIZE, cam_y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    # Add a 3D effect
                    pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (rect.x, rect.y, self.TILE_SIZE, 5))

        # Render Exit
        exit_screen_x = int(cam_x + self.exit_pos[0] * self.TILE_SIZE)
        exit_screen_y = int(cam_y + self.exit_pos[1] * self.TILE_SIZE)
        self._draw_glow(
            (exit_screen_x + self.TILE_SIZE // 2, exit_screen_y + self.TILE_SIZE // 2),
            self.TILE_SIZE * 0.4, self.COLOR_EXIT, 10
        )
        
        # Render Ghost Particles
        for p in self.particles:
            px, py = int(cam_x + p["pos"][0]), int(cam_y + p["pos"][1])
            alpha = int(255 * (p["life"] / 30))
            color = (*self.COLOR_GHOST, alpha)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (px - p["size"], py - p["size"]))

        # Render Ghost
        ghost_screen_x = int(cam_x + self.ghost_pos[0])
        ghost_screen_y = int(cam_y + self.ghost_pos[1])
        ghost_size = self.TILE_SIZE * 0.8
        ghost_surface = pygame.Surface((ghost_size, ghost_size), pygame.SRCALPHA)
        
        # Ghost body with flicker
        flicker = (math.sin(self.steps * 0.5) + 1) / 2
        alpha = 100 + 50 * flicker
        pygame.draw.ellipse(ghost_surface, (*self.COLOR_GHOST, int(alpha)), (0, 0, ghost_size, ghost_size))
        
        # Ghost "eyes"
        eye_size = ghost_size * 0.1
        eye_y = ghost_size * 0.4
        pygame.draw.circle(ghost_surface, (0,0,0,alpha), (ghost_size*0.3, eye_y), eye_size)
        pygame.draw.circle(ghost_surface, (0,0,0,alpha), (ghost_size*0.7, eye_y), eye_size)
        self.screen.blit(ghost_surface, (ghost_screen_x - ghost_size/2, ghost_screen_y - ghost_size/2))

        # Render Player
        player_screen_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self._draw_glow(player_screen_pos, self.TILE_SIZE * 0.3, self.COLOR_PLAYER, 10)

        # Render flickering light overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        radius = 150 + 30 * (math.sin(self.steps * 0.1) + 1) / 2
        pygame.draw.circle(overlay, (0, 0, 0, 0), player_screen_pos, int(radius))
        self.screen.blit(overlay, (0, 0))

    def _draw_glow(self, pos, radius, color, num_layers):
        """Draws a glowing circle effect."""
        for i in range(num_layers, 0, -1):
            alpha = int(150 * (1 - i / num_layers)**2)
            current_radius = int(radius + i * (radius * 0.5 / num_layers))
            s = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (current_radius, current_radius), current_radius)
            self.screen.blit(s, (pos[0] - current_radius, pos[1] - current_radius))
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

    def _render_ui(self):
        # Stage text
        stage_text = self.font_large.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Distance to exit
        px, py = self.player_pos
        dist_to_exit = self.distance_grid[py, px]
        if dist_to_exit != -1:
            dist_text = self.font_large.render(f"Exit in: {dist_to_exit}m", True, self.COLOR_TEXT)
            self.screen.blit(dist_text, (self.WIDTH - dist_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "player_pos": self.player_pos,
            "dist_to_exit": self.distance_grid[self.player_pos[1], self.player_pos[0]],
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Create a real window to display the game
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Haunted Mansion Evade")
    
    done = False
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Only step if a key was pressed, because auto_advance is False
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Terminated: {terminated}, Score: {info['score']:.2f}")

        # Render to the real screen
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        real_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if done:
            print("Game Over!")
            # Wait a bit before closing
            pygame.time.wait(2000)

        clock.tick(10) # Limit manual play speed

    env.close()