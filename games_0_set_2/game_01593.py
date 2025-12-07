
# Generated: 2025-08-28T02:05:35.312539
# Source Brief: brief_01593.md
# Brief Index: 1593

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame


Ghost = namedtuple("Ghost", ["rect", "path", "path_index", "direction"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. "
        "Avoid the red ghosts and find the yellow exit before the time runs out."
    )

    game_description = (
        "Escape a procedurally generated haunted house by avoiding ghosts and reaching the exit within the time limit. "
        "The only light source flickers around you, revealing the maze and the dangers within."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_WALL = (40, 50, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_GHOST = (220, 20, 60)
    COLOR_EXIT = (255, 200, 0)
    COLOR_UI = (200, 200, 220)

    # Game Parameters
    PLAYER_SIZE = 16
    PLAYER_SPEED = 4
    GHOST_SIZE = 18
    INITIAL_GHOST_SPEED = 1.0
    GHOST_SPEED_INCREASE = 0.05
    DIFFICULTY_INTERVAL = 300 # 10 seconds at 30 FPS
    MAX_TIME = 60 * FPS # 60 seconds

    # Maze Generation Parameters
    CELL_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        # State variables are initialized in reset()
        self.player_rect = None
        self.exit_rect = None
        self.ghosts = []
        self.walls = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.current_ghost_speed = self.INITIAL_GHOST_SPEED

        # Light effect
        self.light_radius = 120
        self.light_flicker_amp = 15
        self.light_flicker_speed = 0.2

        self.validate_implementation()

    def _generate_maze(self):
        # Using a simplified Prim's algorithm for maze generation
        grid = [[True for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        walls = []
        
        start_x, start_y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
        grid[start_y][start_x] = False # Mark as path

        # Add walls of the starting cell to the list
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= start_x + dx < self.GRID_WIDTH and 0 <= start_y + dy < self.GRID_HEIGHT:
                walls.append((start_x, start_y, dx, dy))

        while walls:
            wall_x, wall_y, dx, dy = walls.pop(self.np_random.integers(len(walls)))
            
            nx, ny = wall_x + dx, wall_y + dy
            
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and grid[ny][nx]:
                grid[ny][nx] = False # Carve path
                for ndx, ndy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= nx + ndx < self.GRID_WIDTH and 0 <= ny + ndy < self.GRID_HEIGHT:
                        walls.append((nx, ny, ndx, ndy))
        
        # Create wall rects from the grid
        self.walls = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if grid[y][x]:
                    self.walls.append(pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        
        # Create a list of floor cells for entity placement
        floor_cells = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if not grid[y][x]:
                    floor_cells.append((x, y))
        
        return floor_cells

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        self.current_ghost_speed = self.INITIAL_GHOST_SPEED

        floor_cells = self._generate_maze()
        self.np_random.shuffle(floor_cells)

        # Place player
        player_cell = floor_cells.pop()
        self.player_rect = pygame.Rect(
            player_cell[0] * self.CELL_SIZE + (self.CELL_SIZE - self.PLAYER_SIZE) // 2,
            player_cell[1] * self.CELL_SIZE + (self.CELL_SIZE - self.PLAYER_SIZE) // 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )

        # Place exit (far from player)
        best_exit_cell = None
        max_dist = -1
        for cell in floor_cells:
            dist = abs(cell[0] - player_cell[0]) + abs(cell[1] - player_cell[1])
            if dist > max_dist:
                max_dist = dist
                best_exit_cell = cell
        
        floor_cells.remove(best_exit_cell)
        self.exit_rect = pygame.Rect(
            best_exit_cell[0] * self.CELL_SIZE, best_exit_cell[1] * self.CELL_SIZE, 
            self.CELL_SIZE, self.CELL_SIZE
        )

        # Place ghosts
        self.ghosts = []
        num_ghosts = 3
        for _ in range(min(num_ghosts, len(floor_cells))):
            ghost_cell = floor_cells.pop(self.np_random.integers(len(floor_cells)))
            
            rect = pygame.Rect(
                ghost_cell[0] * self.CELL_SIZE + (self.CELL_SIZE - self.GHOST_SIZE) // 2,
                ghost_cell[1] * self.CELL_SIZE + (self.CELL_SIZE - self.GHOST_SIZE) // 2,
                self.GHOST_SIZE, self.GHOST_SIZE
            )
            
            # Create a simple patrol path
            path_type = self.np_random.choice(['horizontal', 'vertical'])
            path = []
            if path_type == 'horizontal':
                path.append(pygame.Vector2(rect.topleft))
                path.append(pygame.Vector2(rect.x + self.CELL_SIZE / 2, rect.y))
            else:
                path.append(pygame.Vector2(rect.topleft))
                path.append(pygame.Vector2(rect.x, rect.y + self.CELL_SIZE / 2))
            
            self.ghosts.append(Ghost(rect=rect, path=path, path_index=1, direction=1))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        
        movement = action[0]
        reward = -0.01  # Small penalty for taking time

        # --- Update Player ---
        dx, dy = 0, 0
        if movement == 1: dy = -self.PLAYER_SPEED
        elif movement == 2: dy = self.PLAYER_SPEED
        elif movement == 3: dx = -self.PLAYER_SPEED
        elif movement == 4: dx = self.PLAYER_SPEED

        # Move and check for wall collisions
        self.player_rect.x += dx
        for wall in self.walls:
            if self.player_rect.colliderect(wall):
                if dx > 0: self.player_rect.right = wall.left
                if dx < 0: self.player_rect.left = wall.right
                break
        
        self.player_rect.y += dy
        for wall in self.walls:
            if self.player_rect.colliderect(wall):
                if dy > 0: self.player_rect.bottom = wall.top
                if dy < 0: self.player_rect.top = wall.bottom
                break
        
        # Boundary checks
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.SCREEN_HEIGHT, self.player_rect.bottom)

        # --- Update Ghosts ---
        for i, ghost in enumerate(self.ghosts):
            target = ghost.path[ghost.path_index]
            current_pos = pygame.Vector2(ghost.rect.center)
            
            move_vec = target - current_pos
            if move_vec.length_squared() < self.current_ghost_speed**2:
                new_path_index = ghost.path_index + ghost.direction
                if not (0 <= new_path_index < len(ghost.path)):
                    new_direction = -ghost.direction
                    new_path_index = ghost.path_index + new_direction
                else:
                    new_direction = ghost.direction
                self.ghosts[i] = ghost._replace(path_index=new_path_index, direction=new_direction)
            else:
                move_vec.scale_to_length(self.current_ghost_speed)
                ghost.rect.center = current_pos + move_vec

        # --- Update Game State ---
        self.steps += 1
        self.time_left -= 1
        
        if self.steps > 0 and self.steps % self.FPS == 0:
            reward += 1 # Reward for each second survived

        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_ghost_speed += self.GHOST_SPEED_INCREASE

        # --- Check for Termination ---
        terminated = False
        if self.player_rect.colliderect(self.exit_rect):
            reward += 100
            terminated = True
            # sound: win_sound.play()
        
        if not terminated:
            for ghost in self.ghosts:
                if self.player_rect.colliderect(ghost.rect):
                    reward -= 10
                    terminated = True
                    # sound: lose_sound.play()
                    break
        
        if not terminated and self.time_left <= 0:
            terminated = True # No specific reward change for timeout

        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        
        # Render exit
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)
        
        # Render ghosts
        for ghost in self.ghosts:
            # Ghost body
            pygame.gfxdraw.filled_circle(self.screen, ghost.rect.centerx, ghost.rect.centery, self.GHOST_SIZE // 2, self.COLOR_GHOST)
            # Ghost trail effect
            for i in range(1, 4):
                offset = i * 4
                color = tuple(max(0, c - 40 * i) for c in self.COLOR_GHOST)
                pygame.gfxdraw.filled_circle(self.screen, ghost.rect.centerx, ghost.rect.centery + offset, max(0, self.GHOST_SIZE // 2 - i * 2), color)

    def _render_light_effect(self):
        # Create a surface for the darkness mask
        darkness = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        darkness.fill((0, 0, 0))

        # Calculate flickering light radius
        flicker = math.sin(self.steps * self.light_flicker_speed) * self.light_flicker_amp
        current_radius = int(self.light_radius + flicker + self.np_random.uniform(-5, 5))

        # Draw soft, layered light circles onto the mask
        for i in range(5):
            radius = int(current_radius * (1 - i * 0.15))
            alpha = 255 - i * 50
            if radius > 0:
                 pygame.gfxdraw.filled_circle(darkness, self.player_rect.centerx, self.player_rect.centery, radius, (alpha, alpha, alpha))
        
        # Blit the darkness mask over the scene with a special blending mode
        self.screen.blit(darkness, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    def _render_ui(self):
        # Render timer
        secs = self.time_left // self.FPS
        mins = secs // 60
        secs %= 60
        timer_text = f"{mins:02}:{secs:02}"
        text_surface = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()

        # Render player on top of game elements but under the light
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        
        self._render_light_effect()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # We need to call reset to initialize everything before getting an observation
        _, _ = self.reset(seed=123)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run the environment in a window
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' on Windows, 'x11' or 'wayland' on Linux

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for visualization
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted House Escape")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    while running:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)

    env.close()