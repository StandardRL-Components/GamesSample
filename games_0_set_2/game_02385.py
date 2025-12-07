
# Generated: 2025-08-28T04:38:25.804564
# Source Brief: brief_02385.md
# Brief Index: 2385

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move. Avoid the ghosts and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, avoiding ghostly apparitions, to reach the exit within a time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (60, 60, 80)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_EXIT = (0, 255, 128)
    COLOR_GHOST = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    CELL_SIZE = 40
    MAZE_WIDTH = 31  # Must be odd
    MAZE_HEIGHT = 21 # Must be odd
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 8
    GHOST_RADIUS = 10
    EXIT_RADIUS = 12
    
    BASE_GHOST_SPEED = 1.5
    GHOST_SPEED_INCREMENT = 0.5
    
    TOTAL_LEVELS = 3
    TIME_PER_LEVEL = 60.0
    
    GHOST_PROXIMITY_RADIUS = 150
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.maze = []
        self.player_pos = np.array([0.0, 0.0])
        self.ghosts = []
        self.exit_pos = np.array([0.0, 0.0])
        self.camera_pos = np.array([0.0, 0.0])
        
        self.current_level = 0
        self.time_remaining = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        
        # This will be initialized in reset()
        self.np_random = None

        # self.validate_implementation() # Call this to check your work

    def _generate_maze(self):
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        maze = np.ones((h, w), dtype=np.uint8) # 1 for wall, 0 for path
        
        def is_valid(x, y):
            return 0 <= x < w and 0 <= y < h

        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_valid_path(self, length):
        path_nodes = np.argwhere(self.maze == 0)
        for _ in range(100): # Try 100 times to find a decent path
            start_node_idx = self.np_random.integers(0, len(path_nodes))
            start_node = path_nodes[start_node_idx]
            
            # Simple random walk to create a path
            path = [start_node]
            current = start_node
            for _ in range(length - 1):
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = current[1] + dx, current[0] + dy
                    if 0 <= ny < self.MAZE_HEIGHT and 0 <= nx < self.MAZE_WIDTH and self.maze[ny, nx] == 0:
                        neighbors.append(np.array([ny, nx]))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                path.append(current)
            
            if len(path) > length // 2:
                # Convert grid coords to pixel coords
                pixel_path = [(p[1] * self.CELL_SIZE + self.CELL_SIZE / 2, p[0] * self.CELL_SIZE + self.CELL_SIZE / 2) for p in path]
                return pixel_path
        return [(100, 100), (200, 200)] # Fallback

    def _setup_level(self):
        self.maze = self._generate_maze()
        
        self.player_pos = np.array([1.5 * self.CELL_SIZE, 1.5 * self.CELL_SIZE])
        self.exit_pos = np.array([(self.MAZE_WIDTH - 1.5) * self.CELL_SIZE, (self.MAZE_HEIGHT - 1.5) * self.CELL_SIZE])

        self.ghosts = []
        ghost_speed = self.BASE_GHOST_SPEED + (self.current_level - 1) * self.GHOST_SPEED_INCREMENT
        for _ in range(3):
            path = self._find_valid_path(length=self.np_random.integers(15, 30))
            self.ghosts.append({
                "pos": np.array(path[0], dtype=float),
                "path": path,
                "path_index": 0,
                "path_direction": 1,
                "speed": ghost_speed,
                "flicker": 1.0
            })
        
        self.time_remaining = self.TIME_PER_LEVEL

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # Unpack action
        movement = action[0]
        
        # Update player
        self._update_player(movement)
        
        # Update ghosts
        self._update_ghosts()

        # --- Calculate Reward & Check Termination ---
        reward = -0.01  # Time penalty
        terminated = False
        
        # Ghost proximity penalty
        min_ghost_dist = min(np.linalg.norm(self.player_pos - g['pos']) for g in self.ghosts) if self.ghosts else float('inf')
        if min_ghost_dist < self.GHOST_PROXIMITY_RADIUS:
            reward -= 1.0 * (1.0 - min_ghost_dist / self.GHOST_PROXIMITY_RADIUS)
        
        # Ghost collision
        for ghost in self.ghosts:
            if np.linalg.norm(self.player_pos - ghost['pos']) < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                reward = -100.0
                terminated = True
                self.win_message = "CAUGHT!"
                break
        
        if not terminated:
            # Timeout
            if self.time_remaining <= 0:
                reward = -100.0
                terminated = True
                self.win_message = "TIME'S UP!"
            
            # Reached exit
            elif np.linalg.norm(self.player_pos - self.exit_pos) < self.PLAYER_RADIUS + self.EXIT_RADIUS:
                reward = 10.0
                self.current_level += 1
                if self.current_level > self.TOTAL_LEVELS:
                    reward += 100.0  # Win bonus
                    terminated = True
                    self.win_message = "YOU ESCAPED!"
                else:
                    self._setup_level()
                    # Sound effect placeholder: # sfx_next_level
        
        self.score += reward
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        direction = np.array([0.0, 0.0])
        if movement == 1: direction[1] = -1 # Up
        elif movement == 2: direction[1] = 1 # Down
        elif movement == 3: direction[0] = -1 # Left
        elif movement == 4: direction[0] = 1 # Right
        
        if np.any(direction):
            new_pos = self.player_pos + direction * self.PLAYER_SPEED
            
            # Wall collision detection
            # Check X and Y movement separately to allow sliding along walls
            # Check X
            if self.maze[int(self.player_pos[1] / self.CELL_SIZE), int(new_pos[0] / self.CELL_SIZE)] == 0:
                self.player_pos[0] = new_pos[0]
            # Check Y
            if self.maze[int(new_pos[1] / self.CELL_SIZE), int(self.player_pos[0] / self.CELL_SIZE)] == 0:
                self.player_pos[1] = new_pos[1]

    def _update_ghosts(self):
        for ghost in self.ghosts:
            # Flicker effect
            ghost['flicker'] = 0.75 + (math.sin(self.steps * 0.5 + id(ghost)) * 0.25)
            
            if len(ghost['path']) < 2: continue
            
            target_node_idx = ghost['path_index'] + ghost['path_direction']
            target_pos = np.array(ghost['path'][target_node_idx])
            
            direction_vec = target_pos - ghost['pos']
            distance = np.linalg.norm(direction_vec)
            
            if distance < ghost['speed']:
                ghost['path_index'] = target_node_idx
                if not (0 < ghost['path_index'] < len(ghost['path']) - 1):
                    ghost['path_direction'] *= -1
                ghost['pos'] = target_pos
            else:
                ghost['pos'] += (direction_vec / distance) * ghost['speed']

    def _get_observation(self):
        # Smooth camera follow
        self.camera_pos = self.camera_pos * 0.9 + self.player_pos * 0.1
        
        # Center of screen
        screen_center = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        camera_offset = self.camera_pos - screen_center
        
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render maze
        self._render_maze(camera_offset)
        
        # Render exit
        self._render_exit(camera_offset)
        
        # Render ghosts
        self._render_ghosts(camera_offset)
        
        # Render player
        self._render_player()
        
        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_maze(self, offset):
        start_col = max(0, int(offset[0] / self.CELL_SIZE))
        end_col = min(self.MAZE_WIDTH, int((offset[0] + self.SCREEN_WIDTH) / self.CELL_SIZE) + 1)
        start_row = max(0, int(offset[1] / self.CELL_SIZE))
        end_row = min(self.MAZE_HEIGHT, int((offset[1] + self.SCREEN_HEIGHT) / self.CELL_SIZE) + 1)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        x * self.CELL_SIZE - offset[0],
                        y * self.CELL_SIZE - offset[1],
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
    
    def _render_exit(self, offset):
        pos = self.exit_pos - offset
        # Glow effect
        for i in range(4):
            radius = self.EXIT_RADIUS + i * 4
            alpha = 100 - i * 25
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.EXIT_RADIUS, self.COLOR_EXIT)

    def _render_ghosts(self, offset):
        min_ghost_dist = min(np.linalg.norm(self.player_pos - g['pos']) for g in self.ghosts) if self.ghosts else float('inf')
        
        for ghost in self.ghosts:
            pos = ghost['pos'] - offset
            
            # Proximity glow effect
            dist_to_player = np.linalg.norm(self.player_pos - ghost['pos'])
            if dist_to_player < self.GHOST_PROXIMITY_RADIUS:
                pulse_rad = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                glow_radius = int(self.GHOST_RADIUS * 2.5 * (1 - dist_to_player / self.GHOST_PROXIMITY_RADIUS) * pulse_rad)
                glow_alpha = int(100 * (1 - dist_to_player / self.GHOST_PROXIMITY_RADIUS))
                if glow_radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), glow_radius, (*self.COLOR_GHOST, glow_alpha))
            
            # Main ghost body
            alpha = int(200 * ghost['flicker'])
            color = (*self.COLOR_GHOST, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.GHOST_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.GHOST_RADIUS, color)

    def _render_player(self):
        pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 2, (*self.COLOR_PLAYER, 60))
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Level
        level_text = f"MAZE: {self.current_level}/{self.TOTAL_LEVELS}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(level_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 30))
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_surf = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "time_remaining": self.time_remaining,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset() # Reset for a new game
            terminated = False

    env.close()