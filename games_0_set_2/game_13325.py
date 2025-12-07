import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:43:10.762364
# Source Brief: brief_03325.md
# Brief Index: 3325
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Ball:
    """Represents a single bouncing ball."""
    def __init__(self, x, y, radius, color, glow_color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.radius = radius
        self.color = color
        self.glow_color = glow_color

    def update(self, accel_x, gravity, damping, bounds):
        """Update ball physics for one step."""
        self.vx += accel_x
        self.vy += gravity

        # Apply damping/friction
        self.vx *= damping
        self.vy *= damping

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Wall collisions
        if self.x - self.radius < bounds['left']:
            self.x = bounds['left'] + self.radius
            self.vx *= -1
            # sfx: bounce
        elif self.x + self.radius > bounds['right']:
            self.x = bounds['right'] - self.radius
            self.vx *= -1
            # sfx: bounce
        if self.y - self.radius < bounds['top']:
            self.y = bounds['top'] + self.radius
            self.vy *= -1
            # sfx: bounce
        elif self.y + self.radius > bounds['bottom']:
            self.y = bounds['bottom'] - self.radius
            self.vy *= -1
            # sfx: bounce

    def draw(self, surface):
        """Draw the ball with a glow effect."""
        # Draw glow
        glow_radius = int(self.radius * 1.8)
        # Use a surface for alpha blending the glow
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surface, glow_radius, glow_radius, glow_radius, (*self.glow_color, 50))
        surface.blit(glow_surface, (int(self.x) - glow_radius, int(self.y) - glow_radius))

        # Draw main ball
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)

class Maze:
    """Handles procedural generation and rendering of the maze."""
    def __init__(self, width, height, cell_size):
        self.maze_w = width // cell_size
        self.maze_h = height // cell_size
        self.cell_size = cell_size
        self.grid = [[{'up': True, 'down': True, 'left': True, 'right': True, 'visited': False} for _ in range(self.maze_w)] for _ in range(self.maze_h)]
        self.walls = []
        self._generate()

    def _generate(self):
        stack = [(0, 0)]
        self.grid[0][0]['visited'] = True
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not self.grid[cy - 1][cx]['visited']: neighbors.append(('up', cx, cy - 1))
            if cy < self.maze_h - 1 and not self.grid[cy + 1][cx]['visited']: neighbors.append(('down', cx, cy + 1))
            if cx > 0 and not self.grid[cy][cx - 1]['visited']: neighbors.append(('left', cx - 1, cy))
            if cx < self.maze_w - 1 and not self.grid[cy][cx + 1]['visited']: neighbors.append(('right', cx + 1, cy))

            if neighbors:
                direction, nx, ny = random.choice(neighbors)
                if direction == 'up':
                    self.grid[cy][cx]['up'] = False
                    self.grid[ny][nx]['down'] = False
                elif direction == 'down':
                    self.grid[cy][cx]['down'] = False
                    self.grid[ny][nx]['up'] = False
                elif direction == 'left':
                    self.grid[cy][cx]['left'] = False
                    self.grid[ny][nx]['right'] = False
                elif direction == 'right':
                    self.grid[cy][cx]['right'] = False
                    self.grid[ny][nx]['left'] = False
                
                self.grid[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        self._cache_walls()

    def _cache_walls(self):
        self.walls = []
        for y in range(self.maze_h):
            for x in range(self.maze_w):
                px, py = x * self.cell_size, y * self.cell_size
                if self.grid[y][x]['up']: self.walls.append(((px, py), (px + self.cell_size, py)))
                if self.grid[y][x]['down']: self.walls.append(((px, py + self.cell_size), (px + self.cell_size, py + self.cell_size)))
                if self.grid[y][x]['left']: self.walls.append(((px, py), (px, py + self.cell_size)))
                if self.grid[y][x]['right']: self.walls.append(((px + self.cell_size, py), (px + self.cell_size, py + self.cell_size)))

    def draw(self, surface, color):
        for wall in self.walls:
            pygame.draw.line(surface, color, wall[0], wall[1], 2)

    def is_wall_at(self, x, y):
        """Check if a point collides with any maze wall."""
        if not (0 <= x < self.maze_w * self.cell_size and 0 <= y < self.maze_h * self.cell_size):
            return True # Out of bounds
        
        grid_x, grid_y = int(x // self.cell_size), int(y // self.cell_size)
        cell_x, cell_y = x % self.cell_size, y % self.cell_size
        
        wall_thickness = 2
        cell = self.grid[grid_y][grid_x]

        if cell['up'] and cell_y < wall_thickness: return True
        if cell['down'] and cell_y > self.cell_size - wall_thickness: return True
        if cell['left'] and cell_x < wall_thickness: return True
        if cell['right'] and cell_x > self.cell_size - wall_thickness: return True
        
        return False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide three synchronized balls through a maze. Keep them vertically aligned to trigger chain "
        "reactions that clear obstacles and advance to the next level."
    )
    user_guide = (
        "Controls: Use ← and → to move the selected ball. Hold 'space' to control the green ball, "
        "or hold 'shift' to control the blue ball. Releasing both selects the default red ball."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 18, 23)
    COLOR_WALL = (200, 200, 220)
    COLOR_OBSTACLE = (255, 150, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_BALLS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255)    # Blue
    ]
    COLOR_BALLS_GLOW = [
        (80, 20, 20),
        (20, 80, 20),
        (20, 20, 80)
    ]
    MAX_LEVELS = 5
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.render_mode = render_mode
        self._initialize_state()

    def _initialize_state(self):
        """Initializes all game state variables. Called from init and reset."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        
        self.balls = [
            Ball(50, 50, 10, self.COLOR_BALLS[0], self.COLOR_BALLS_GLOW[0]),
            Ball(70, 50, 10, self.COLOR_BALLS[1], self.COLOR_BALLS_GLOW[1]),
            Ball(90, 50, 10, self.COLOR_BALLS[2], self.COLOR_BALLS_GLOW[2]),
        ]
        self.obstacles = []
        self.particles = []
        self.maze = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        self._setup_level()
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        """Generates a new maze and places balls and obstacles."""
        self.maze = Maze(self.WIDTH, self.HEIGHT, cell_size=80)
        
        # Place balls at start
        start_pos = (40, 40)
        for i, ball in enumerate(self.balls):
            ball.x = start_pos[0] + i * 5
            ball.y = start_pos[1]
            ball.vx, ball.vy = random.uniform(-1, 1), random.uniform(-1, 1)

        # Place obstacles
        self.obstacles = []
        num_obstacles = 5 + int(math.ceil((self.current_level - 1) * 0.1 * 5))
        obstacle_size = 20
        
        for _ in range(num_obstacles):
            while True:
                x = self.np_random.integers(obstacle_size, self.WIDTH - obstacle_size)
                y = self.np_random.integers(obstacle_size, self.HEIGHT - obstacle_size)
                
                # Ensure not in a wall or too close to start
                if not self.maze.is_wall_at(x, y) and math.dist((x, y), start_pos) > 100:
                    self.obstacles.append(pygame.Rect(x - obstacle_size // 2, y - obstacle_size // 2, obstacle_size, obstacle_size))
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        truncated = False
        self.steps += 1
        
        # --- 1. Handle Action ---
        # A creative control scheme to map MultiDiscrete to 3 balls:
        # actions[0] (movement) controls a ball.
        # actions[1] (space) selects ball 2.
        # actions[2] (shift) selects ball 3.
        # Default is ball 1.
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        accel = [0.0, 0.0, 0.0]
        accel_strength = 0.3
        
        active_ball_idx = 0 # Default to ball 1 (red)
        if shift_held:
            active_ball_idx = 2 # Shift overrides space for ball 3 (blue)
        elif space_held:
            active_ball_idx = 1 # Space for ball 2 (green)

        if movement == 3: # Left
            accel[active_ball_idx] = -accel_strength
        elif movement == 4: # Right
            accel[active_ball_idx] = accel_strength

        # --- 2. Update Physics & Collisions ---
        bounds = {'left': 0, 'right': self.WIDTH, 'top': 0, 'bottom': self.HEIGHT}
        for i, ball in enumerate(self.balls):
            ball.update(accel[i], gravity=0.1, damping=0.99, bounds=bounds)
            
            # Maze wall collisions
            if self.maze.is_wall_at(ball.x, ball.y):
                # A simple resolution: nudge out and reverse velocity
                # This is imperfect but works for this arcade feel.
                ball.x -= ball.vx * 2
                ball.y -= ball.vy * 2
                ball.vx *= -0.5
                ball.vy *= -0.5

            # Obstacle collisions
            for obs_rect in self.obstacles:
                if obs_rect.collidepoint(ball.x, ball.y):
                    terminated = True
                    self.game_over = True
                    reward -= 100
                    # sfx: game_over_negative
                    break
            if terminated: break
        
        # --- 3. Synchronization & Chain Reaction ---
        if not terminated:
            ball_ys = [b.y for b in self.balls]
            sync_delta = max(ball_ys) - min(ball_ys)
            sync_threshold = 10 # pixels
            
            if sync_delta < self.HEIGHT * 0.1: # Continuous reward for getting close
                reward += 0.1

            if sync_delta < sync_threshold:
                # Chain Reaction Triggered!
                # sfx: chain_reaction_trigger
                reward += 1
                destroyed_obstacles = []
                reaction_radius = 60
                
                for ball in self.balls:
                    self._create_explosion(ball.x, ball.y, reaction_radius)
                    for obs_rect in self.obstacles:
                        if math.dist((ball.x, ball.y), obs_rect.center) < reaction_radius:
                            if obs_rect not in destroyed_obstacles:
                                destroyed_obstacles.append(obs_rect)
                
                if destroyed_obstacles:
                    # sfx: obstacle_destroy
                    self.obstacles = [obs for obs in self.obstacles if obs not in destroyed_obstacles]
                    self.score += len(destroyed_obstacles)

        # --- 4. Level/Win Progression ---
        if not self.obstacles and not terminated:
            self.current_level += 1
            if self.current_level > self.MAX_LEVELS:
                # VICTORY
                reward += 100
                terminated = True
                self.game_over = True
                # sfx: victory_fanfare
            else:
                # Next Level
                reward += 10
                self._setup_level()
                # sfx: level_up

        # --- 5. Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['radius'] += p['expand_rate']
            p['alpha'] = max(0, p['alpha'] - p['fade_rate'])

        # --- 6. Termination ---
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_explosion(self, x, y, max_radius):
        """Adds particles for a chain reaction explosion."""
        num_particles = 1
        for _ in range(num_particles):
            self.particles.append({
                'x': x, 'y': y,
                'radius': 10,
                'life': 30, # frames
                'expand_rate': (max_radius - 10) / 30,
                'alpha': 200,
                'fade_rate': 200 / 30,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Maze
        if self.maze:
            self.maze.draw(self.screen, self.COLOR_WALL)

        # Render Obstacles
        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
        
        # Render Particles
        for p in self.particles:
            if p['alpha'] > 0:
                s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(s, int(p['radius']), int(p['radius']), int(p['radius']), (255, 255, 255, int(p['alpha'])))
                self.screen.blit(s, (int(p['x'] - p['radius']), int(p['y'] - p['radius'])))

        # Render Balls
        for ball in self.balls:
            ball.draw(self.screen)

    def _render_ui(self):
        # Level Text
        level_text = self.font_main.render(f"Level: {self.current_level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Score Text
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))
        
        # Sync Bar
        ball_ys = [b.y for b in self.balls]
        sync_delta = max(ball_ys) - min(ball_ys) if self.balls else 0
        max_possible_delta = self.HEIGHT - self.balls[0].radius * 2
        sync_progress = 1.0 - min(1.0, sync_delta / (max_possible_delta / 2))
        
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        # Draw bar background
        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Draw filled portion
        fill_color = (100, 220, 100)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * sync_progress, bar_height), border_radius=4)

        # Draw bar label
        sync_label = self.font_small.render("Sync", True, self.COLOR_TEXT)
        self.screen.blit(sync_label, (bar_x - sync_label.get_width() - 5, bar_y + 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level}
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    # This part of the script is for human interaction and will not be executed by the validation tests.
    # It requires a display to run, so we conditionally unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Synchronized Bouncer")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # This control scheme demonstrates the action space logic
        # Shift selects ball 3, Space selects ball 2, default is ball 1
        # The agent uses the MultiDiscrete action space defined.
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            if keys[pygame.K_LEFT]: action[0] = 3
            if keys[pygame.K_RIGHT]: action[0] = 4
        elif keys[pygame.K_SPACE]:
            action[1] = 1
            if keys[pygame.K_LEFT]: action[0] = 3
            if keys[pygame.K_RIGHT]: action[0] = 4
        else:
            if keys[pygame.K_LEFT]: action[0] = 3
            if keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_r]: # Reset
             obs, info = env.reset()
             total_reward = 0
             terminated = False
             truncated = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()