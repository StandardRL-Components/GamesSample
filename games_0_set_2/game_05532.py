
# Generated: 2025-08-28T05:18:26.299493
# Source Brief: brief_05532.md
# Brief Index: 5532

        
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
        "Controls: Arrow keys to move. Evade the ghosts and reach the green exit portal before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated haunted maze, evading spectral foes to reach the exit within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Maze and rendering properties
        self.maze_dim = (31, 19) # Must be odd numbers for maze generation
        self.cell_size = min(self.screen_width // self.maze_dim[0], self.screen_height // self.maze_dim[1])
        self.grid_offset_x = (self.screen_width - self.maze_dim[0] * self.cell_size) // 2
        self.grid_offset_y = (self.screen_height - self.maze_dim[1] * self.cell_size) // 2
        
        # Game constants
        self.max_steps = 1000
        self.num_ghosts = 3
        
        # Colors
        self.color_bg = (20, 20, 30)
        self.color_wall = (50, 50, 70)
        self.color_player = (255, 255, 255)
        self.color_exit = (0, 255, 128)
        self.ghost_colors = [(255, 80, 80), (80, 120, 255), (80, 255, 150)]
        self.color_text = (220, 220, 220)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.ghosts = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)
        stack = deque()
        
        start_x, start_y = (self.np_random.integers(0, width // 2) * 2 + 1, self.np_random.integers(0, height // 2) * 2 + 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                mx, my = (x + nx) // 2, (y + ny) // 2
                maze[ny, nx] = 0
                maze[my, mx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.particles.clear()

        self.maze = self._generate_maze(self.maze_dim[0], self.maze_dim[1])
        
        self.player_pos = np.array([1, 1], dtype=np.int32)
        self.exit_pos = np.array([self.maze_dim[0] - 2, self.maze_dim[1] - 2], dtype=np.int32)
        
        self.ghosts.clear()
        valid_spawn_points = np.argwhere(self.maze == 0)
        for i in range(self.num_ghosts):
            while True:
                spawn_idx = self.np_random.integers(0, len(valid_spawn_points))
                pos = valid_spawn_points[spawn_idx][::-1] # a[y,x] -> (x,y)
                if np.linalg.norm(pos - self.player_pos) > 5: # Spawn away from player
                    break
            
            ghost = {
                "pos": pos.astype(np.int32),
                "color": self.ghost_colors[i % len(self.ghost_colors)],
                "direction": self.np_random.choice([[0,1], [0,-1], [1,0], [-1,0]], axis=0),
                "trail": deque(maxlen=10)
            }
            self.ghosts.append(ghost)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        dist_before = np.linalg.norm(self.player_pos - self.exit_pos)
        
        self._update_player(movement)
        self._update_ghosts()
        self._update_particles()
        
        dist_after = np.linalg.norm(self.player_pos - self.exit_pos)
        
        # Calculate reward
        reward = -0.1  # Step penalty
        if dist_after < dist_before:
            reward += 0.5
        elif dist_after > dist_before:
            reward -= 0.5

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.game_outcome == "WIN":
                time_bonus = (self.max_steps - self.steps) * 0.1
                reward = 10.0 + time_bonus
                self.score += 1
            elif self.game_outcome == "LOSE_GHOST":
                reward = -10.0
            elif self.game_outcome == "LOSE_TIME":
                reward = -5.0 # Smaller penalty for timeout

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        direction_map = {
            1: np.array([0, -1]),  # Up
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0]),   # Right
        }
        if movement in direction_map:
            direction = direction_map[movement]
            next_pos = self.player_pos + direction
            if self.maze[next_pos[1], next_pos[0]] == 0:
                self.player_pos = next_pos

    def _update_ghosts(self):
        for ghost in self.ghosts:
            ghost['trail'].append(ghost['pos'].copy())
            
            # Check for valid new directions
            valid_moves = []
            for d in [[0,1], [0,-1], [1,0], [-1,0]]:
                next_pos_check = ghost['pos'] + d
                if self.maze[next_pos_check[1], next_pos_check[0]] == 0:
                    valid_moves.append(d)
            
            # Decide if changing direction
            is_at_intersection = len(valid_moves) > 2
            next_pos = ghost['pos'] + ghost['direction']
            is_at_wall = self.maze[next_pos[1], next_pos[0]] == 1
            
            if is_at_wall or (is_at_intersection and self.np_random.random() < 0.3): # 30% chance to turn at intersection
                # Exclude turning back unless it's a dead end
                if len(valid_moves) > 1:
                    reverse_dir = [-ghost['direction'][0], -ghost['direction'][1]]
                    possible_moves = [m for m in valid_moves if m != reverse_dir]
                    if not possible_moves: # In case only reverse is possible
                        possible_moves = valid_moves
                else:
                    possible_moves = valid_moves
                
                if possible_moves:
                    move_idx = self.np_random.integers(0, len(possible_moves))
                    ghost['direction'] = np.array(possible_moves[move_idx])

            next_pos = ghost['pos'] + ghost['direction']
            if self.maze[next_pos[1], next_pos[0]] == 0:
                ghost['pos'] = next_pos
                
            # Add particles for movement
            for _ in range(2):
                self.particles.append({
                    "pos": self._grid_to_pixel(ghost['pos']) + self.np_random.random(2) * self.cell_size - self.cell_size/2,
                    "life": self.np_random.integers(10, 20),
                    "max_life": 20,
                    "color": ghost['color']
                })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1

    def _check_termination(self):
        # Check ghost collision
        for ghost in self.ghosts:
            if np.array_equal(self.player_pos, ghost['pos']):
                self.game_over = True
                self.game_outcome = "LOSE_GHOST"
                return True
        
        # Check exit
        if np.array_equal(self.player_pos, self.exit_pos):
            self.game_over = True
            self.game_outcome = "WIN"
            return True
            
        # Check time/steps
        if self.steps >= self.max_steps:
            self.game_over = True
            self.game_outcome = "LOSE_TIME"
            return True
            
        return False

    def _grid_to_pixel(self, grid_pos):
        px = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        py = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return np.array([px, py])

    def _render_glow(self, surface, center, color, max_radius, steps=5):
        for i in range(steps, 0, -1):
            radius = int(max_radius * (i / steps))
            alpha = int(100 * (i / steps)**2)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, (*color, alpha))

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        
        # Render Maze
        wall_size = (self.cell_size, self.cell_size)
        for y in range(self.maze_dim[1]):
            for x in range(self.maze_dim[0]):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(self.grid_offset_x + x * self.cell_size, self.grid_offset_y + y * self.cell_size, *wall_size)
                    pygame.draw.rect(self.screen, self.color_wall, rect)

        # Render Exit
        exit_pixel_pos = self._grid_to_pixel(self.exit_pos)
        self._render_glow(self.screen, exit_pixel_pos, self.color_exit, self.cell_size)
        pygame.gfxdraw.filled_circle(self.screen, int(exit_pixel_pos[0]), int(exit_pixel_pos[1]), self.cell_size // 3, self.color_exit)

        # Render Particles
        for p in self.particles:
            alpha = int(200 * (p['life'] / p['max_life']))
            radius = int(self.cell_size / 8 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], alpha))

        # Render Ghosts
        for ghost in self.ghosts:
            ghost_pixel_pos = self._grid_to_pixel(ghost['pos'])
            color_with_alpha = (*ghost['color'], 180)
            radius = int(self.cell_size * 0.4)
            self._render_glow(self.screen, ghost_pixel_pos, ghost['color'], radius * 2, steps=4)
            pygame.gfxdraw.filled_circle(self.screen, int(ghost_pixel_pos[0]), int(ghost_pixel_pos[1]), radius, color_with_alpha)
            pygame.gfxdraw.aacircle(self.screen, int(ghost_pixel_pos[0]), int(ghost_pixel_pos[1]), radius, color_with_alpha)

        # Render Player
        player_pixel_pos = self._grid_to_pixel(self.player_pos)
        player_radius = int(self.cell_size * 0.4)
        self._render_glow(self.screen, player_pixel_pos, self.color_player, player_radius * 1.5)
        pygame.draw.circle(self.screen, self.color_player, player_pixel_pos, player_radius)

        # Render UI
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.color_text)
        self.screen.blit(score_text, (10, 10))

        time_left = self.max_steps - self.steps
        time_color = (255, 100, 100) if time_left < self.max_steps / 4 else self.color_text
        time_text = self.font_ui.render(f"TIME: {time_left}", True, time_color)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            outcome_text_str = "YOU WIN!" if self.game_outcome == "WIN" else "GAME OVER"
            outcome_color = self.color_exit if self.game_outcome == "WIN" else self.ghost_colors[0]
            
            text_surface = self.font_game_over.render(outcome_text_str, True, outcome_color)
            text_rect = text_surface.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text_surface, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "time_left": self.max_steps - self.steps,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Haunted Maze")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while True:
        # Event handling
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([movement, space_held, shift_held])

            # Only step if an action is taken (since auto_advance is False)
            if not np.array_equal(action, [0, 0, 0]):
                obs, reward, terminated, truncated, info = env.step(action)
                # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit human play speed