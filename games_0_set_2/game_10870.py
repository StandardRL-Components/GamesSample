import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:11:57.441519
# Source Brief: brief_00870.md
# Brief Index: 870
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a shape-shifting agent navigates a maze to collect orbs.

    The agent can take one of three forms, each with different speeds:
    - Cube (Default): Standard speed.
    - Sphere (Spacebar): Slower, agile.
    - Pyramid (Shift): Faster, powerful.

    The goal is to collect 100 orbs within 60 seconds. The maze is procedurally
    generated for each new episode.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated maze as a shape-shifting agent, collecting orbs before time runs out. "
        "Switch between forms to alter your speed."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Hold space to become a slow sphere or shift to become a fast pyramid."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    EPISODE_DURATION_SECONDS = 60
    MAX_STEPS = EPISODE_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 18, 22)
    COLOR_WALL = (40, 45, 55)
    COLOR_FLOOR = (25, 30, 38)
    COLOR_ORB = (255, 200, 0)
    COLOR_ORB_GLOW = (255, 200, 0, 50)
    COLOR_TEXT = (220, 220, 230)
    
    PLAYER_COLORS = {
        "cube": (60, 120, 255),    # Blue
        "sphere": (255, 80, 80),   # Red
        "pyramid": (80, 255, 120), # Green
    }
    
    # Maze Generation
    MAZE_COLS = 32  # 640 / 20
    MAZE_ROWS = 20  # 400 / 20
    CELL_SIZE = 20
    NUM_ORBS_TO_SPAWN = 120 # More than needed to ensure win is possible

    # Player Shapes & Speeds
    SHAPE_CUBE = 0
    SHAPE_SPHERE = 1
    SHAPE_PYRAMID = 2
    
    PLAYER_SPEEDS = {
        SHAPE_CUBE: 4.0,
        SHAPE_SPHERE: 2.5,
        SHAPE_PYRAMID: 6.0,
    }
    PLAYER_SIZE = 12
    ORB_SIZE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_shape = self.SHAPE_CUBE
        self.maze = []
        self.orbs = []
        self.particles = []
        self.last_dist_to_orb = float('inf')

        # --- Self-Validation ---
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Generate maze and place entities
        self._generate_maze()
        self.player_pos = self._find_valid_spawn_point() * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player_shape = self.SHAPE_CUBE
        self._spawn_orbs()
        
        self.last_dist_to_orb = self._get_dist_to_nearest_orb()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Update player shape
        if shift_held:
            self.player_shape = self.SHAPE_PYRAMID
        elif space_held:
            self.player_shape = self.SHAPE_SPHERE
        else:
            self.player_shape = self.SHAPE_CUBE

        # 2. Update player position
        self._move_player(movement)
        
        # 3. Check for orb collection
        reward = self._collect_orbs()

        # 4. Calculate shaping reward
        current_dist_to_orb = self._get_dist_to_nearest_orb()
        if self.orbs:
            # Reward for getting closer
            reward += 0.1 * (self.last_dist_to_orb - current_dist_to_orb) / self.PLAYER_SPEEDS[self.SHAPE_PYRAMID]
        self.last_dist_to_orb = current_dist_to_orb

        # 5. Update particles
        self._update_particles()

        # 6. Check for termination
        terminated = False
        truncated = False
        if self.score >= 100:
            reward += 100  # Victory bonus
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 50  # Timeout penalty
            terminated = True # Per Gymnasium API, timeout is termination
            truncated = True  # And also truncation
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _move_player(self, movement_action):
        speed = self.PLAYER_SPEEDS[self.player_shape]
        direction = np.array([0.0, 0.0])
        
        if movement_action == 1: # Up
            direction[1] = -1
        elif movement_action == 2: # Down
            direction[1] = 1
        elif movement_action == 3: # Left
            direction[0] = -1
        elif movement_action == 4: # Right
            direction[0] = 1

        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        new_pos = self.player_pos + direction * speed
        
        # Collision detection with walls
        # Check four corners of the player's bounding box
        player_radius = self.PLAYER_SIZE / 2
        
        # Check X-axis movement
        next_x_pos = np.array([new_pos[0], self.player_pos[1]])
        if not self._is_colliding(next_x_pos):
            self.player_pos[0] = new_pos[0]
            
        # Check Y-axis movement
        next_y_pos = np.array([self.player_pos[0], new_pos[1]])
        if not self._is_colliding(next_y_pos):
            self.player_pos[1] = new_pos[1]

        # Clamp to screen boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], player_radius, self.SCREEN_WIDTH - player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], player_radius, self.SCREEN_HEIGHT - player_radius)

    def _is_colliding(self, pos):
        player_radius = self.PLAYER_SIZE / 2
        points_to_check = [
            (pos[0], pos[1]),
            (pos[0] + player_radius, pos[1]),
            (pos[0] - player_radius, pos[1]),
            (pos[0], pos[1] + player_radius),
            (pos[0], pos[1] - player_radius),
        ]
        for p in points_to_check:
            grid_x, grid_y = int(p[0] // self.CELL_SIZE), int(p[1] // self.CELL_SIZE)
            if not (0 <= grid_x < self.MAZE_COLS and 0 <= grid_y < self.MAZE_ROWS):
                return True # Out of bounds
            if self.maze[grid_y][grid_x] == 1:
                return True # Wall
        return False

    def _collect_orbs(self):
        collected_reward = 0
        orbs_remaining = []
        for orb_pos in self.orbs:
            dist = np.linalg.norm(self.player_pos - orb_pos)
            if dist < self.PLAYER_SIZE / 2 + self.ORB_SIZE:
                self.score += 1
                collected_reward += 1.0
                # sfx: orb collection sound
                self._spawn_particles(orb_pos)
            else:
                orbs_remaining.append(orb_pos)
        self.orbs = orbs_remaining
        return collected_reward

    def _get_dist_to_nearest_orb(self):
        if not self.orbs:
            return 0
        distances = [np.linalg.norm(self.player_pos - orb_pos) for orb_pos in self.orbs]
        return min(distances)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_FLOOR)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "orbs_left": len(self.orbs),
        }

    def _render_game(self):
        self._render_maze()
        self._render_orbs()
        self._render_particles()
        self._render_player()

    def _render_maze(self):
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL,
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    def _render_orbs(self):
        for orb_pos in self.orbs:
            x, y = int(orb_pos[0]), int(orb_pos[1])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ORB_SIZE + 3, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ORB_SIZE + 3, self.COLOR_ORB_GLOW)
            # Core orb
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ORB_SIZE, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ORB_SIZE, self.COLOR_ORB)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        r = int(self.PLAYER_SIZE / 2)
        color = self.PLAYER_COLORS["cube"]
        shadow_color = tuple(c // 2 for c in color)
        
        # Shadow
        shadow_offset = 2
        if self.player_shape == self.SHAPE_SPHERE:
            color = self.PLAYER_COLORS["sphere"]
            pygame.draw.circle(self.screen, shadow_color, (x + shadow_offset, y + shadow_offset), r)
        elif self.player_shape == self.SHAPE_PYRAMID:
            color = self.PLAYER_COLORS["pyramid"]
            points = [(x, y - r), (x - r, y + r//2), (x + r, y + r//2)]
            shadow_points = [(p[0] + shadow_offset, p[1] + shadow_offset) for p in points]
            pygame.draw.polygon(self.screen, shadow_color, shadow_points)
        else: # Cube
            shadow_rect = pygame.Rect(x - r + shadow_offset, y - r + shadow_offset, self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=2)

        # Main shape
        if self.player_shape == self.SHAPE_SPHERE:
            pygame.draw.circle(self.screen, color, (x, y), r)
        elif self.player_shape == self.SHAPE_PYRAMID:
            points = [(x, y - r), (x - r, y + r//2), (x + r, y + r//2)]
            pygame.draw.polygon(self.screen, color, points)
        else: # Cube
            rect = pygame.Rect(x - r, y - r, self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=2)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score:03d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Shape indicator
        shape_names = {self.SHAPE_CUBE: "CUBE", self.SHAPE_SPHERE: "SPHERE", self.SHAPE_PYRAMID: "PYRAMID"}
        shape_name_str = shape_names[self.player_shape]
        shape_text = self.font_small.render(f"FORM: {shape_name_str}", True, self.PLAYER_COLORS[shape_name_str.lower()])
        self.screen.blit(shape_text, (10, self.SCREEN_HEIGHT - shape_text.get_height() - 10))


    def _generate_maze(self):
        self.maze = [[1] * self.MAZE_COLS for _ in range(self.MAZE_ROWS)]
        start_x, start_y = (self.np_random.integers(1, self.MAZE_COLS//2) * 2,
                            self.np_random.integers(1, self.MAZE_ROWS//2) * 2)
        
        stack = [(start_x, start_y)]
        self.maze[start_y][start_x] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.MAZE_COLS and 0 <= ny < self.MAZE_ROWS and self.maze[ny][nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors) # Use standard random for maze gen consistency if needed
                self.maze[ny][nx] = 0
                self.maze[y + (ny-y)//2][x + (nx-x)//2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _find_valid_spawn_point(self):
        while True:
            x = self.np_random.integers(0, self.MAZE_COLS)
            y = self.np_random.integers(0, self.MAZE_ROWS)
            if self.maze[y][x] == 0:
                return np.array([x, y], dtype=float)

    def _spawn_orbs(self):
        self.orbs = []
        path_cells = []
        for r in range(self.MAZE_ROWS):
            for c in range(self.MAZE_COLS):
                if self.maze[r][c] == 0:
                    path_cells.append((c, r))
        
        self.np_random.shuffle(path_cells)
        
        for i in range(min(self.NUM_ORBS_TO_SPAWN, len(path_cells))):
            c, r = path_cells[i]
            # Avoid spawning too close to player start
            if np.linalg.norm(np.array([c, r]) * self.CELL_SIZE - self.player_pos) > self.CELL_SIZE * 3:
                pos = np.array([c * self.CELL_SIZE + self.CELL_SIZE / 2, 
                                r * self.CELL_SIZE + self.CELL_SIZE / 2])
                self.orbs.append(pos)

    def _spawn_particles(self, pos):
        # sfx: particle burst
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(10, 25)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': self.COLOR_ORB})
    
    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_particles(self):
        for p in self.particles:
            # Create a temporary surface for alpha blending
            particle_surf = pygame.Surface((p['life'], p['life']), pygame.SRCALPHA)
            alpha = int(255 * (p['life'] / 25))
            color = (*p['color'], alpha)
            size = int(max(1, 3 * (p['life'] / 25)))
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Shape Shifter Maze")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # --- Rendering ---
        # The observation is the rendered frame, so we convert it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Tick Clock ---
        env.clock.tick(GameEnv.FPS)

    env.close()