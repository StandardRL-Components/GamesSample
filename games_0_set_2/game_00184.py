
# Generated: 2025-08-27T12:52:22.384063
# Source Brief: brief_00184.md
# Brief Index: 184

        
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
        "Controls: Use ← and → to choose jump direction. ↑ or no-op to jump straight up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, grid-based arcade game. Hop upwards on a scrolling grid, "
        "dodging procedurally generated obstacles to reach the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 16
        self.CELL_SIZE = self.SCREEN_WIDTH // self.GRID_COLS
        self.GRID_ROWS_ON_SCREEN = self.SCREEN_HEIGHT // self.CELL_SIZE + 2
        
        self.WIN_HEIGHT = 200
        self.MAX_LIVES = 5
        self.MAX_STEPS = 1000
        self.GAP_WIDTH = 4 # in grid cells

        # Colors
        self.COLOR_BG = (20, 20, 60)
        self.COLOR_GRID = (40, 40, 100)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (150, 30, 30)
        self.COLOR_PARTICLE = (255, 150, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 0, 0)
        self.COLOR_HEART_EMPTY = (70, 70, 70)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
        
        # --- State Variables ---
        self.steps = None
        self.score = None
        self.lives = None
        self.player_pos = None # [grid_x, grid_y]
        self.camera_world_y = None # World grid_y at the bottom of the screen
        self.obstacles = None
        self.particles = None
        self.obstacle_spawn_timer = None
        self.np_random = None

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.player_pos = [self.GRID_COLS // 2, 1]
        self.camera_world_y = 0
        self.obstacles = []
        self.particles = []
        self.obstacle_spawn_timer = 0
        
        # Pre-populate some safe space at the beginning
        for i in range(self.GRID_ROWS_ON_SCREEN // 2):
             self._spawn_obstacle(force_y=i * 4)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = False
        
        # 1. Unpack action and determine player movement
        movement = action[0]
        dx, dy = 0, 0
        
        if movement == 3: # Left
            dx, dy = -1, 1
            reward -= 0.2
            # sfx: player_jump_side.wav
        elif movement == 4: # Right
            dx, dy = 1, 1
            reward -= 0.2
            # sfx: player_jump_side.wav
        elif movement in [0, 1]: # None or Up
            dx, dy = 0, 1
            # sfx: player_jump_up.wav
        # movement == 2 (down) results in no player movement
        
        # 2. Apply player movement
        if dy > 0:
            self.player_pos[0] += dx
            self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_COLS - 1)
            self.player_pos[1] += dy

        # 3. Update world state
        self._update_obstacles()
        self._update_particles()
        
        # 4. Collision detection
        collided = False
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 1, 1)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], obs['w'], 1)
            if player_rect.colliderect(obs_rect):
                collided = True
                break
        
        if collided:
            self.lives -= 1
            reward -= 5.0
            self._create_particles(self.player_pos[0], self.player_pos[1], 20)
            # sfx: player_hit.wav
            self.player_pos = [self.GRID_COLS // 2, int(self.camera_world_y) + 2]

        # 5. Update score and calculate height-based reward
        prev_score = self.score
        self.score = max(self.score, self.player_pos[1])
        reward += (self.score - prev_score) * 0.1

        # 6. Check for termination conditions
        self.steps += 1
        if self.score >= self.WIN_HEIGHT:
            reward += 100.0
            terminated = True
            # sfx: game_win.wav
        elif self.lives <= 0:
            terminated = True
            # sfx: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        # Obstacle speed increases by 0.1 every 50 steps.
        # This is interpreted as a probability of moving 1 grid cell down per step.
        speed_prob = min(1.0, 0.1 + (self.steps / 50) * 0.1)
        if self.np_random.random() < speed_prob:
            for obs in self.obstacles:
                obs['y'] -= 1
        
        self.obstacles = [obs for obs in self.obstacles if obs['y'] > self.camera_world_y - 5]

        # Obstacle spawn frequency increases by 1 step every 100 steps
        spawn_rate = max(1, 5 - self.steps // 100)
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = spawn_rate

    def _spawn_obstacle(self, force_y=None):
        spawn_y = force_y if force_y is not None else int(self.camera_world_y + self.GRID_ROWS_ON_SCREEN)
        gap_start_col = self.np_random.integers(1, self.GRID_COLS - self.GAP_WIDTH)
        
        if gap_start_col > 0:
            self.obstacles.append({'x': 0, 'y': spawn_y, 'w': gap_start_col})
        if gap_start_col + self.GAP_WIDTH < self.GRID_COLS:
            self.obstacles.append({'x': gap_start_col + self.GAP_WIDTH, 'y': spawn_y, 'w': self.GRID_COLS - (gap_start_col + self.GAP_WIDTH)})
        # sfx: obstacle_spawn.wav

    def _create_particles(self, grid_x, grid_y, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': (grid_x + 0.5) * self.CELL_SIZE,
                'y': (grid_y + 0.5) * self.CELL_SIZE,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40)
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _world_to_screen(self, world_x, world_y):
        screen_y = self.SCREEN_HEIGHT - (world_y - self.camera_world_y * self.CELL_SIZE)
        return int(world_x), int(screen_y)

    def _grid_to_screen(self, grid_x, grid_y):
        world_pixel_x = grid_x * self.CELL_SIZE
        world_pixel_y = grid_y * self.CELL_SIZE
        camera_pixel_y = self.camera_world_y * self.CELL_SIZE
        screen_x = world_pixel_x
        screen_y = self.SCREEN_HEIGHT - (world_pixel_y - camera_pixel_y)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        # Camera follows player, keeping them in the lower third of the screen
        target_camera_y = self.player_pos[1] - (self.GRID_ROWS_ON_SCREEN * 0.3)
        self.camera_world_y = max(0, target_camera_y)

        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        camera_pixel_y_offset = (self.camera_world_y * self.CELL_SIZE) % self.CELL_SIZE
        for i in range(self.GRID_COLS + 1):
            x = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for i in range(self.GRID_ROWS_ON_SCREEN + 1):
            y = self.SCREEN_HEIGHT - (i * self.CELL_SIZE - camera_pixel_y_offset)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_obstacles(self):
        for obs in self.obstacles:
            sx, sy = self._grid_to_screen(obs['x'], obs['y'])
            rect = pygame.Rect(sx, sy, obs['w'] * self.CELL_SIZE, self.CELL_SIZE)
            glow_rect = rect.inflate(8, 8)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _render_player(self):
        px, py = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        player_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        glow_center = player_rect.center
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], int(self.CELL_SIZE * 0.7), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], int(self.CELL_SIZE * 0.7), self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4), border_radius=4)
    
    def _render_particles(self):
        for p in self.particles:
            sx, sy = self._world_to_screen(p['x'], p['y'])
            alpha = max(0, min(255, int(p['life'] * 8)))
            color = (*self.COLOR_PARTICLE, alpha)
            radius = int(p['life'] * 0.2)
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (sx - radius, sy - radius))

    def _render_ui(self):
        score_text = self.font_large.render(f"Height: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))
        for i in range(self.MAX_LIVES):
            heart_color = self.COLOR_HEART if i < self.lives else self.COLOR_HEART_EMPTY
            self._draw_heart(self.screen, self.SCREEN_WIDTH - 25 - (i * 30), 25, heart_color)

    def _draw_heart(self, surface, x, y, color):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.draw.polygon(surface, color, points)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    pygame.display.set_caption("Grid Hopper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        movement = 0 
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0]
        
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                obs, reward, terminated, _, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Lives: {info['lives']}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("Game Over!")
            pygame.time.wait(2000)

    env.close()