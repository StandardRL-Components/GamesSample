import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:05:20.026424
# Source Brief: brief_02567.md
# Brief Index: 2567
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a shape-shifting player navigates a procedurally
    generated maze to reach an exit within a time limit. The player can switch
    between three forms: a fast Sphere, a balanced Cube, and a slow Pyramid,
    each with different movement speeds. The core challenge lies in choosing the
    right form for different parts of the maze to optimize for time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated maze to reach the exit. "
        "Switch between different shapes to change your speed and find the fastest path."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Press space or shift to cycle through your different shapes."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (100, 110, 130)
    COLOR_PATH = (40, 45, 60)
    COLOR_EXIT = (0, 255, 128)
    COLOR_EXIT_GLOW = (100, 255, 200)

    PLAYER_COLORS = [(70, 130, 220), (255, 69, 0), (255, 215, 0)] # Blue, Red, Yellow
    PLAYER_SHAPES = ["CUBE", "SPHERE", "PYRAMID"]
    PLAYER_SPEEDS = [4.0, 6.0, 2.5] # Cube, Sphere, Pyramid speeds

    # Screen and Maze Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_CELL_SIZE = 20
    MAZE_BASE_WIDTH = (SCREEN_WIDTH // MAZE_CELL_SIZE) // 2 * 2 + 1
    MAZE_BASE_HEIGHT = (SCREEN_HEIGHT // MAZE_CELL_SIZE) // 2 * 2 + 1

    # Game parameters
    MAX_STEPS = 1800 # 60 seconds at 30fps
    PARTICLE_LIFESPAN = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen_dim = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface(self.screen_dim)
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_shape = pygame.font.SysFont("sans", 24, bold=True)
        
        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_shape_idx = 0
        self.maze = np.array([])
        self.exit_pos = np.array([0, 0])
        self.time_remaining = 0
        self.last_dist_to_exit = 0
        self.particles = []
        self.last_button_state = [False, False] # [space, shift]
        self.successful_escapes = 0
        self.difficulty_level = 0
        
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.successful_escapes > 0 and self.successful_escapes % 5 == 0:
            self.difficulty_level = min(5, self.difficulty_level + 1)

        self._generate_maze()
        
        start_cell = (1, 1)
        self.player_pos = np.array([
            start_cell[0] * self.MAZE_CELL_SIZE + self.MAZE_CELL_SIZE / 2,
            start_cell[1] * self.MAZE_CELL_SIZE + self.MAZE_CELL_SIZE / 2
        ], dtype=float)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.player_shape_idx = 0 # Default to Cube
        self.particles = []
        self.last_button_state = [False, False]
        
        dist = np.linalg.norm(self.player_pos - self.exit_pos)
        self.last_dist_to_exit = dist

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_shape_shifting(space_held, shift_held)
        self._handle_movement(movement)
        
        # --- Game Logic Update ---
        self.steps += 1
        self.time_remaining -= 1
        self._update_particles()
        
        # --- Reward Calculation ---
        reward = self._calculate_reward()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        if terminated:
            self.game_over = True
            if np.linalg.norm(self.player_pos - self.exit_pos) < self.MAZE_CELL_SIZE / 2:
                reward = 100.0  # Win
                self.score = 100
                self.successful_escapes += 1
            else:
                reward = -100.0 # Loss
                self.score = -100
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_maze(self):
        w = self.MAZE_BASE_WIDTH - self.difficulty_level * 2
        h = self.MAZE_BASE_HEIGHT - self.difficulty_level * 2
        
        # Ensure dimensions are odd and at least 3
        w = max(5, w // 2 * 2 + 1)
        h = max(5, h // 2 * 2 + 1)

        maze = np.ones((w, h), dtype=np.uint8)
        
        # Recursive backtracking
        def carve(x, y):
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < w and 0 <= ny < h and maze[nx, ny] == 1:
                    maze[x + dx, y + dy] = 0
                    maze[nx, ny] = 0
                    carve(nx, ny)

        start_x, start_y = (1, 1)
        maze[start_x, start_y] = 0
        carve(start_x, start_y)

        # Find the furthest point for the exit
        q = [(start_x, start_y, 0)]
        visited = {(start_x, start_y)}
        max_dist = -1
        end_pos = (start_x, start_y)

        while q:
            x, y, dist = q.pop(0)
            if dist > max_dist:
                max_dist = dist
                end_pos = (x, y)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and maze[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))
        
        self.maze = maze
        self.exit_pos = np.array([
            end_pos[0] * self.MAZE_CELL_SIZE + self.MAZE_CELL_SIZE / 2,
            end_pos[1] * self.MAZE_CELL_SIZE + self.MAZE_CELL_SIZE / 2
        ])

    def _handle_shape_shifting(self, space_held, shift_held):
        space_pressed = space_held and not self.last_button_state[0]
        shift_pressed = shift_held and not self.last_button_state[1]

        if space_pressed:
            self.player_shape_idx = (self.player_shape_idx + 1) % len(self.PLAYER_SHAPES)
        elif shift_pressed:
            self.player_shape_idx = (self.player_shape_idx - 1 + len(self.PLAYER_SHAPES)) % len(self.PLAYER_SHAPES)

        self.last_button_state = [space_held, shift_held]

    def _handle_movement(self, movement_action):
        direction_vectors = {
            1: np.array([0, -1]),  # Up
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0]),   # Right
        }
        if movement_action in direction_vectors:
            move_vec = direction_vectors[movement_action]
            speed = self.PLAYER_SPEEDS[self.player_shape_idx]
            new_pos = self.player_pos + move_vec * speed
            
            # Collision detection
            player_radius = self.MAZE_CELL_SIZE / 4
            grid_pos = (int(new_pos[0] / self.MAZE_CELL_SIZE), int(new_pos[1] / self.MAZE_CELL_SIZE))
            
            can_move = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = grid_pos[0] + dx, grid_pos[1] + dy
                    if 0 <= check_x < self.maze.shape[0] and 0 <= check_y < self.maze.shape[1]:
                        if self.maze[check_x, check_y] == 1:
                            wall_rect = pygame.Rect(check_x * self.MAZE_CELL_SIZE, check_y * self.MAZE_CELL_SIZE, self.MAZE_CELL_SIZE, self.MAZE_CELL_SIZE)
                            if wall_rect.colliderect(
                                new_pos[0] - player_radius, new_pos[1] - player_radius,
                                player_radius * 2, player_radius * 2
                            ):
                                can_move = False
                                break
                if not can_move:
                    break

            if can_move:
                self.player_pos = new_pos
                self._add_particle()

    def _add_particle(self):
        particle_color = list(self.PLAYER_COLORS[self.player_shape_idx])
        particle = {
            "pos": self.player_pos.copy(),
            "vel": (self.np_random.random(2) - 0.5) * 1.5,
            "life": self.PARTICLE_LIFESPAN,
            "color": tuple(particle_color)
        }
        self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _calculate_reward(self):
        current_dist = np.linalg.norm(self.player_pos - self.exit_pos)
        reward = (self.last_dist_to_exit - current_dist) * 0.1 # Reward for getting closer
        self.last_dist_to_exit = current_dist
        return reward

    def _check_termination(self):
        at_exit = np.linalg.norm(self.player_pos - self.exit_pos) < self.MAZE_CELL_SIZE / 2
        time_out = self.time_remaining <= 0
        return at_exit or time_out

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_remaining}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw maze
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                rect = (x * self.MAZE_CELL_SIZE, y * self.MAZE_CELL_SIZE, self.MAZE_CELL_SIZE, self.MAZE_CELL_SIZE)
                color = self.COLOR_WALL if self.maze[x, y] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit with glow
        exit_center = (int(self.exit_pos[0]), int(self.exit_pos[1]))
        for i in range(10, 0, -2):
            s = self.MAZE_CELL_SIZE * (0.5 + i / 20.0)
            alpha = 100 - i * 10
            glow_surf = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_EXIT_GLOW, alpha), glow_surf.get_rect())
            self.screen.blit(glow_surf, (exit_center[0] - s, exit_center[1] - s))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_center[0] - self.MAZE_CELL_SIZE/4, exit_center[1] - self.MAZE_CELL_SIZE/4, self.MAZE_CELL_SIZE/2, self.MAZE_CELL_SIZE/2))
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, p["life"] / self.PARTICLE_LIFESPAN)
            radius = int(alpha * 4)
            if radius > 0:
                color = (*p["color"], int(alpha * 200))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

        # Draw player
        self._draw_player()

    def _draw_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        color = self.PLAYER_COLORS[self.player_shape_idx]
        radius = self.MAZE_CELL_SIZE // 3

        # Draw a subtle glow under the player
        glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 60), (radius*2, radius*2), radius*2)
        self.screen.blit(glow_surf, (pos[0] - radius*2, pos[1] - radius*2))

        shape_name = self.PLAYER_SHAPES[self.player_shape_idx]
        if shape_name == "SPHERE":
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,255,255))
        elif shape_name == "CUBE":
            size = radius * 1.8
            rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1)
        elif shape_name == "PYRAMID":
            points = [
                (pos[0], pos[1] - radius),
                (pos[0] - radius, pos[1] + radius * 0.7),
                (pos[0] + radius, pos[1] + radius * 0.7)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255))

    def _render_ui(self):
        # Timer display
        time_text = f"TIME: {self.time_remaining / 30:.1f}"
        time_surf = self.font_ui.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        # Shape display
        shape_name = self.PLAYER_SHAPES[self.player_shape_idx]
        shape_color = self.PLAYER_COLORS[self.player_shape_idx]
        shape_surf = self.font_shape.render(shape_name, True, shape_color)
        self.screen.blit(shape_surf, (10, 10))
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Shape Shifter Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    while running:
        # --- Human Input Handling ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered surface, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Score: {info['score']}. Press 'R' to restart.")
            # Wait for restart input
            while terminated and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                pygame.time.wait(10) # Prevent busy-waiting

        env.clock.tick(60) # Limit to 60 FPS for human play

    env.close()