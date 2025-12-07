
# Generated: 2025-08-28T03:46:08.064007
# Source Brief: brief_05034.md
# Brief Index: 5034

        
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
        "Controls: Press Space to jump. Time your landings to the beat to build your combo."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, neon-infused rhythm runner. Jump over obstacles to the beat, time your landings for bonus points, and survive to the end of the track."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 10 * FPS  # 10 seconds

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_GROUND = (60, 20, 100)
    COLOR_GRID = (120, 60, 200)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_OBSTACLE = (255, 0, 80)
    COLOR_OBSTACLE_OUTLINE = (255, 100, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_PARTICLE_PERFECT = (255, 255, 255)
    COLOR_PARTICLE_JUMP = (0, 255, 255)

    # Physics & Gameplay
    GROUND_Y = HEIGHT - 50
    PLAYER_X = WIDTH // 4
    PLAYER_SIZE = 20
    GRAVITY = 0.5
    JUMP_STRENGTH = -10
    
    # Rhythm
    BEAT_PERIOD = 30 # 120 BPM at 60 FPS
    PERFECT_LANDING_WINDOW = 2 # frames
    
    # Obstacles
    OBSTACLE_WIDTH = 30
    OBSTACLE_HEIGHT = 40
    INITIAL_OBSTACLE_SPEED = 3.0
    
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
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except pygame.error:
            # Fallback if default font isn't found (e.g., in some minimal environments)
            self.font_large = pygame.font.SysFont("monospace", 36)
            self.font_small = pygame.font.SysFont("monospace", 24)

        # Game state variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = False
        self.obstacles = []
        self.particles = []
        self.combo = 0
        self.beat_timer = 0
        self.obstacle_speed = 0
        self.last_space_held = False
        self.next_obstacle_spawn_step = 0
        self.grid_lines = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.GROUND_Y - self.PLAYER_SIZE
        self.player_vy = 0
        self.on_ground = True
        
        self.obstacles = []
        self.particles = []
        
        self.combo = 0
        self.beat_timer = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.last_space_held = False
        
        # Schedule first obstacle
        self.next_obstacle_spawn_step = self.FPS * 2

        # Initialize grid lines for parallax effect
        self.grid_lines = [self.GROUND_Y + 5 + i * 5 for i in range(10)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Tick the clock for auto-advance
        if self.auto_advance:
            self.clock.tick(self.FPS)
            
        reward = 0.0
        
        # --- Unpack factorized action ---
        movement = action[0]  # 0-4: none/up/down/left/right (unused)
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        # --- Handle Input ---
        jump_initiated = space_held and not self.last_space_held and self.on_ground
        self.last_space_held = space_held

        if jump_initiated:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            # SFX: Jump
            self._create_particles(self.PLAYER_X, self.player_y + self.PLAYER_SIZE, 20, self.COLOR_PARTICLE_JUMP, (0, 2), (-1, 1))

        # --- Update Game Logic ---
        self.steps += 1
        self.beat_timer = (self.beat_timer + 1) % self.BEAT_PERIOD
        
        # --- Update Player ---
        was_on_ground = self.on_ground
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y - self.PLAYER_SIZE:
            self.player_y = self.GROUND_Y - self.PLAYER_SIZE
            self.player_vy = 0
            self.on_ground = True
            
            # Check for perfect landing if we just landed
            if not was_on_ground:
                # SFX: Land
                is_on_beat = self.beat_timer <= self.PERFECT_LANDING_WINDOW or \
                             self.beat_timer >= self.BEAT_PERIOD - self.PERFECT_LANDING_WINDOW
                if is_on_beat:
                    # SFX: Perfect Land
                    reward += 2.0
                    self.score += 10 * self.combo
                    self._create_particles(self.PLAYER_X, self.player_y + self.PLAYER_SIZE, 30, self.COLOR_PARTICLE_PERFECT, (-2, 0), (-2, 2))
                else:
                    self.combo = 0 # Reset combo on imperfect landing

        # --- Update Obstacles ---
        player_rect = pygame.Rect(self.PLAYER_X - self.PLAYER_SIZE / 2, self.player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for obs in self.obstacles[:]:
            obs['x'] -= self.obstacle_speed
            obs_rect = pygame.Rect(obs['x'], obs['y'], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)
            
            # Collision check
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                # SFX: Crash/Fail
                reward = -10.0
                self.score -= 50
                break

            # Check for clearing an obstacle
            if not obs['cleared'] and obs_rect.right < player_rect.left:
                obs['cleared'] = True
                reward += 1.0
                self.score += 10
                self.combo += 1
                # SFX: Clear Obstacle
            
            # Remove off-screen obstacles
            if obs_rect.right < 0:
                self.obstacles.remove(obs)
        
        # --- Spawn New Obstacles ---
        if self.steps >= self.next_obstacle_spawn_step and self.steps < self.MAX_STEPS - self.FPS:
            self._spawn_obstacle()
            spawn_interval = self.np_random.choice([self.BEAT_PERIOD * 2, self.BEAT_PERIOD * 3, self.BEAT_PERIOD * 4])
            self.next_obstacle_spawn_step = self.steps + int(spawn_interval)

        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed = min(self.obstacle_speed + 0.05, 10.0)

        # --- Update Particles & Grid ---
        self._update_particles()
        self._update_grid()
        
        # --- Termination & Survival Reward ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not self.game_over:
            reward += 0.1 # Survival reward
            if self.steps >= self.MAX_STEPS:
                # SFX: Level Complete
                reward += 100.0
                self.score += 1000

        self.score = max(0, self.score) # Score cannot be negative

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_obstacle(self):
        self.obstacles.append({
            'x': self.WIDTH,
            'y': self.GROUND_Y - self.OBSTACLE_HEIGHT,
            'cleared': False
        })

    def _create_particles(self, x, y, count, color, vy_range, vx_range):
        for _ in range(count):
            self.particles.append({
                'x': x + self.np_random.uniform(-5, 5),
                'y': y + self.np_random.uniform(-5, 5),
                'vx': self.np_random.uniform(vx_range[0], vx_range[1]),
                'vy': self.np_random.uniform(vy_range[0], vy_range[1]),
                'life': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_grid(self):
        for i in range(len(self.grid_lines)):
            self.grid_lines[i] += 0.5
            if self.grid_lines[i] > self.HEIGHT:
                self.grid_lines[i] = self.GROUND_Y + 5

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [max(0, int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw pulsing beat indicator
        beat_progress = (self.beat_timer / self.BEAT_PERIOD)
        pulse = (1 - abs(2 * beat_progress - 1)) # Triangle wave from 0 to 1
        pulse_color = (20 + pulse * 20, 10, 40 + pulse * 20)
        pygame.draw.circle(self.screen, pulse_color, (self.PLAYER_X, self.GROUND_Y - 100), int(50 + pulse * 15))

        # Draw ground and grid
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 5)
        for y in self.grid_lines:
            alpha = max(0, 255 * (1 - (y - self.GROUND_Y) / (self.HEIGHT - self.GROUND_Y)))
            color = (*self.COLOR_GRID, alpha)
            s = pygame.Surface((self.WIDTH, 1), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (0, int(y)))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            size = int(p['size'])
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (int(p['x'] - size), int(p['y'] - size)))

        # Draw obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs['x']), int(obs['y']), self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, rect, 2)
            
        # Draw player
        player_rect = pygame.Rect(int(self.PLAYER_X - self.PLAYER_SIZE / 2), int(self.player_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Player glow
        for i in range(4):
            alpha = 60 - i * 15
            radius = self.PLAYER_SIZE // 2 + i * 4
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, radius, (*self.COLOR_PLAYER_GLOW, alpha))
            pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, radius, (*self.COLOR_PLAYER_GLOW, alpha))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
    
    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, x, y, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, (x, y))

        # Draw score and combo
        score_text = f"SCORE: {int(self.score)}"
        combo_text = f"COMBO: {self.combo}x"
        draw_text(score_text, self.font_small, 20, 10)
        draw_text(combo_text, self.font_large, 20, 40)

        # Draw progress bar
        progress = self.steps / self.MAX_STEPS
        bar_width = self.WIDTH - 40
        bar_height = 10
        bar_x = 20
        bar_y = self.HEIGHT - 25
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=5)
        
        # Draw game over message
        if self.game_over:
            if self.steps < self.MAX_STEPS:
                msg = "FAILED"
            else:
                msg = "LEVEL COMPLETE"
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            draw_text(msg, self.font_large, self.WIDTH // 2 - self.font_large.size(msg)[0] // 2, self.HEIGHT // 2 - 50)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Rhythm Runner")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    running = True
    
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()