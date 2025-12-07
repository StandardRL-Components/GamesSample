import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:00:01.168405
# Source Brief: brief_00143.md
# Brief Index: 143
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a maze to collect all the crystals before time runs out. "
        "Change your size to pass through tight spaces or break obstacles."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move. "
        "Press space to cycle through small, medium, and large sizes."
    )
    auto_advance = True

    # --- Constants ---
    COLOR_BG = (15, 18, 32)
    COLOR_WALL = (40, 45, 60)
    COLOR_WALL_SIDE = (30, 35, 50)
    COLOR_OBSTACLE = (80, 50, 50)
    COLOR_OBSTACLE_SIDE = (60, 40, 40)
    COLOR_CRYSTAL = (255, 255, 180)
    
    PLAYER_COLORS = [(80, 150, 255), (80, 255, 150), (255, 80, 80)] # Blue, Green, Red
    PLAYER_GLOWS = [(30, 60, 120), (30, 120, 60), (120, 30, 30)]
    PLAYER_SIDES = [(60, 110, 200), (60, 200, 110), (200, 60, 60)]
    
    PLAYER_SIZES = [0.4, 0.6, 0.8] # As a factor of cell size
    
    MAX_LEVELS = 5
    CRYSTALS_PER_LEVEL = 10
    TIME_PER_LEVEL = 60 # seconds
    STEPS_PER_SECOND = 10 # Game logic ticks per second

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_level = pygame.font.SysFont('Consolas', 32, bold=True)
        
        self.render_mode = render_mode
        self.steps_per_level = self.TIME_PER_LEVEL * self.STEPS_PER_SECOND
        
        # Uninitialized variables defined here for clarity
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.time_left = 0
        self.prev_space_held = False
        self.player_grid_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.player_size_idx = 1 # Start as medium
        self.maze = np.array([[]])
        self.cell_size = 0
        self.maze_offset = [0, 0]
        self.crystals = []
        self.obstacles = []
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.prev_space_held = False
        self.particles = []
        
        self._reset_level()
        
        return self._get_observation(), self._get_info()

    def _reset_level(self):
        """Resets state for the start of a new level."""
        self.time_left = self.steps_per_level
        
        # Maze dimensions increase with level
        base_cols, base_rows = 15, 10
        maze_cols = base_cols + (self.level - 1) * 2
        maze_rows = base_rows + (self.level - 1) * 2
        
        self.maze = self._generate_maze(maze_cols, maze_rows)
        
        # Calculate rendering geometry
        self.cell_size = min((self.width - 40) // maze_cols, (self.height - 80) // maze_rows)
        maze_pixel_width = self.cell_size * maze_cols
        maze_pixel_height = self.cell_size * maze_rows
        self.maze_offset = [(self.width - maze_pixel_width) // 2, (self.height - maze_pixel_height) // 2 + 20]
        
        path_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_cells)
        
        # Player start
        start_pos = path_cells.pop()
        self.player_grid_pos = [start_pos[1], start_pos[0]] # grid_pos is [x, y]
        self.player_visual_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_size_idx = 1 # Reset to medium size

        # Place obstacles
        num_obstacles = min(len(path_cells), 2 + (self.level - 1))
        self.obstacles = []
        for _ in range(num_obstacles):
            if not path_cells: break
            ox, oy = path_cells.pop()
            self.obstacles.append([oy, ox])
            self.maze[ox, oy] = 2 # Mark as obstacle

        # Place crystals
        num_crystals = min(len(path_cells), self.CRYSTALS_PER_LEVEL)
        self.crystals = []
        for _ in range(num_crystals):
            if not path_cells: break
            cx, cy = path_cells.pop()
            self.crystals.append([cy, cx])
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        
        # --- Handle Input and State Update ---
        reward = self._handle_input(action)
        self._update_game_state()
        
        # --- Calculate Rewards ---
        # Reward for collecting a crystal is handled in _update_game_state
        
        # --- Check Termination ---
        terminated = False
        if self.time_left <= 0:
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True
        
        # Win condition
        if self.level > self.MAX_LEVELS:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Reward calculation based on distance to nearest crystal ---
        reward = 0
        if self.crystals:
            dist_before = self._get_dist_to_nearest_crystal()

        # --- Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right

        if dx != 0 or dy != 0:
            target_x, target_y = self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy
            
            if self._is_walkable(target_x, target_y):
                self.player_grid_pos = [target_x, target_y]
            # Special case: Large player destroying an obstacle
            elif self.player_size_idx == 2 and self._is_obstacle(target_x, target_y):
                self.player_grid_pos = [target_x, target_y]
                self.maze[target_y, target_x] = 0 # Destroy obstacle
                self.obstacles.remove([target_x, target_y])
                self._create_particles(self._grid_to_pixel([target_x, target_y]), self.COLOR_OBSTACLE, 20)
                # Sound: sfx_obstacle_break.wav

        # --- Distance-based reward ---
        if self.crystals:
            dist_after = self._get_dist_to_nearest_crystal()
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.01

        # --- Shape-shifting ---
        if space_pressed and not self.prev_space_held:
            # Sound: sfx_shapeshift.wav
            next_size_idx = (self.player_size_idx + 1) % 3
            if self._can_fit(self.player_grid_pos, next_size_idx):
                self.player_size_idx = next_size_idx
        self.prev_space_held = space_pressed
        
        return reward

    def _update_game_state(self):
        # Smooth player visual movement
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_visual_pos[0] = self._lerp(self.player_visual_pos[0], target_pixel_pos[0], 0.3)
        self.player_visual_pos[1] = self._lerp(self.player_visual_pos[1], target_pixel_pos[1], 0.3)

        # Crystal collection
        collected_crystal = None
        for crystal_pos in self.crystals:
            if crystal_pos == self.player_grid_pos:
                collected_crystal = crystal_pos
                break
        
        if collected_crystal:
            self.crystals.remove(collected_crystal)
            self.score += 10 # Base reward for crystal
            self._create_particles(self._grid_to_pixel(collected_crystal), self.COLOR_CRYSTAL, 30)
            # Sound: sfx_crystal_collect.wav
            
            if not self.crystals: # Level complete
                self.level += 1
                if self.level <= self.MAX_LEVELS:
                    self._reset_level()

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Maze ---
        side_offset = max(1, self.cell_size // 8)
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                cell_type = self.maze[r, c]
                px, py = self.maze_offset[0] + c * self.cell_size, self.maze_offset[1] + r * self.cell_size
                
                if cell_type == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL_SIDE, (px, py, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.cell_size, self.cell_size - side_offset))
                elif cell_type == 2: # Obstacle
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_SIDE, (px, py, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (px, py, self.cell_size, self.cell_size - side_offset))

        # --- Draw Crystals ---
        for cx, cy in self.crystals:
            px, py = self._grid_to_pixel([cx, cy])
            size = self.cell_size * 0.2
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
            points = [
                (px, py - size - pulse/2), (px + size, py),
                (px, py + size), (px - size, py)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_CRYSTAL)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_CRYSTAL)

        # --- Draw Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'] * (p['life'] / p['max_life']))
            self.screen.blit(s, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))
            
        # --- Draw Player ---
        px, py = self.player_visual_pos
        size_factor = self.PLAYER_SIZES[self.player_size_idx]
        player_rad = (self.cell_size * size_factor) / 2
        
        # Glow
        glow_color = self.PLAYER_GLOWS[self.player_size_idx]
        for i in range(4, 0, -1):
            alpha = 40 - i * 8
            s = pygame.Surface(( (player_rad + i * 2)*2, (player_rad + i * 2)*2 ), pygame.SRCALPHA)
            pygame.draw.circle(s, (*glow_color, alpha), (player_rad + i * 2, player_rad + i * 2), player_rad + i * 2)
            self.screen.blit(s, (int(px - (player_rad + i * 2)), int(py - (player_rad + i * 2))))

        # Body
        side_color = self.PLAYER_SIDES[self.player_size_idx]
        top_color = self.PLAYER_COLORS[self.player_size_idx]
        rect = pygame.Rect(px - player_rad, py - player_rad, player_rad * 2, player_rad * 2)
        pygame.draw.rect(self.screen, side_color, rect.move(0, side_offset), border_radius=3)
        pygame.draw.rect(self.screen, top_color, rect, border_radius=3)

    def _render_ui(self):
        # Crystal Counter
        crystal_text = f"CRYSTALS: {self.CRYSTALS_PER_LEVEL - len(self.crystals)}/{self.CRYSTALS_PER_LEVEL}"
        text_surf = self.font_ui.render(crystal_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (20, 10))

        # Timer
        time_str = f"TIME: {self.time_left // self.STEPS_PER_SECOND:02d}"
        text_surf = self.font_ui.render(time_str, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.width - text_surf.get_width() - 20, 10))

        # Level
        level_str = f"LEVEL {self.level}"
        text_surf = self.font_level.render(level_str, True, (255, 255, 255))
        self.screen.blit(text_surf, (self.width/2 - text_surf.get_width()/2, 5))
        
        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.level > self.MAX_LEVELS else "TIME UP!"
            msg_surf = self.font_level.render(msg, True, (255, 255, 255))
            self.screen.blit(msg_surf, (self.width/2 - msg_surf.get_width()/2, self.height/2 - msg_surf.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "crystals_left": len(self.crystals),
            "time_left": self.time_left // self.STEPS_PER_SECOND
        }
        
    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.int8)
        stack = deque()
        
        start_x, start_y = self.np_random.integers(0, width//2)*2, self.np_random.integers(0, height//2)*2
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                wall_x, wall_y = (cx + nx) // 2, (cy + ny) // 2
                maze[ny, nx] = 0
                maze[wall_y, wall_x] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _grid_to_pixel(self, grid_pos):
        px = self.maze_offset[0] + grid_pos[0] * self.cell_size + self.cell_size / 2
        py = self.maze_offset[1] + grid_pos[1] * self.cell_size + self.cell_size / 2
        return [px, py]

    def _is_walkable(self, x, y):
        if not (0 <= x < self.maze.shape[1] and 0 <= y < self.maze.shape[0]):
            return False
        
        size_idx = self.player_size_idx
        return self._can_fit([x, y], size_idx) and self.maze[y, x] != 2 # Can't walk into obstacles directly

    def _is_obstacle(self, x, y):
        if not (0 <= x < self.maze.shape[1] and 0 <= y < self.maze.shape[0]):
            return False
        return self.maze[y, x] == 2
        
    def _can_fit(self, grid_pos, size_idx):
        player_size = self.PLAYER_SIZES[size_idx]
        if player_size <= 0.6: # Small and Medium fit in 1x1 cell
            if not (0 <= grid_pos[0] < self.maze.shape[1] and 0 <= grid_pos[1] < self.maze.shape[0]):
                return False
            return self.maze[grid_pos[1], grid_pos[0]] != 1
        
        # Large needs a 3x3 clearance of non-wall cells
        # Simplified to just check immediate neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_pos[0] + dx, grid_pos[1] + dy
                if not (0 <= nx < self.maze.shape[1] and 0 <= ny < self.maze.shape[0]):
                    return False # Out of bounds
                if self.maze[ny, nx] == 1:
                    return False # Collides with a wall
        return True

    def _get_dist_to_nearest_crystal(self):
        if not self.crystals: return 0
        player_pos = np.array(self.player_grid_pos)
        crystal_positions = np.array(self.crystals)
        distances = np.linalg.norm(crystal_positions - player_pos, axis=1)
        return np.min(distances)

    def _lerp(self, start, end, t):
        return start + t * (end - start)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example of how to run the environment ---
    # This block is for human play and debugging, not for the agent.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Shape-Shifter Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Action mapping for human play ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # [movement, space, shift]
    action = [0, 0, 0] 
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        action = [0, 0, 0] # Reset action each frame
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
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            # Optionally, reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.STEPS_PER_SECOND) # Match the game's logic speed

    env.close()