
# Generated: 2025-08-27T20:30:19.409893
# Source Brief: brief_02482.md
# Brief Index: 2482

        
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
        "Controls: ↑ to move up, ↓ to move down. Avoid the red spikes."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced avoidance game. Survive as long as you can against an onslaught of deadly spikes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60 # As per brief
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200)
        self.COLOR_SPIKE = (255, 50, 80)
        self.COLOR_SPIKE_GLOW = (255, 150, 150)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_PARTICLE = self.COLOR_PLAYER

        # Game Parameters
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.SPIKE_SIZE = 20
        self.VICTORY_STEPS = 30 * self.FPS # 30 seconds to win

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        
        # Initialize state variables (will be properly set in reset)
        self.player_pos = None
        self.spikes = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.spike_speed = 0
        self.current_spike_spawn_rate = 0
        self.spike_spawn_timer = 0
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.spikes = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty parameters
        self.spike_speed = 2.0
        self.current_spike_spawn_rate = 2.0 * self.FPS # 1 spike every 2 seconds
        self.spike_spawn_timer = self.current_spike_spawn_rate

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is already over, do nothing and return terminal state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_spikes()
        self._update_particles()
        self._update_difficulty()
        
        self.steps += 1
        
        # --- Check for Termination ---
        terminated = False
        victory = False
        
        if self._check_collisions():
            # sfx: player_death
            terminated = True
            self._create_death_particles()
        
        if not terminated and self.steps >= self.VICTORY_STEPS:
            # sfx: victory_fanfare
            terminated = True
            victory = True
            
        self.game_over = terminated
        
        # --- Calculate Reward ---
        reward = 0
        if terminated:
            if victory:
                reward = 100.0
            else: # Collision
                reward = -10.0
        else:
            reward = 0.1  # Survival reward
            if movement == 0:
                reward -= 0.2 # No-op penalty

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

    def _update_spikes(self):
        # Move existing spikes
        for spike in self.spikes:
            spike['pos'].x -= self.spike_speed
        
        # Remove off-screen spikes
        self.spikes = [s for s in self.spikes if s['pos'].x > -self.SPIKE_SIZE]

        # Spawn new spikes
        self.spike_spawn_timer -= 1
        if self.spike_spawn_timer <= 0:
            y_pos = self.np_random.uniform(self.SPIKE_SIZE, self.HEIGHT - self.SPIKE_SIZE)
            self.spikes.append({'pos': pygame.Vector2(self.WIDTH + self.SPIKE_SIZE, y_pos)})
            self.spike_spawn_timer = self.current_spike_spawn_rate
            # sfx: spawn_spike

    def _update_difficulty(self):
        # Increase spike speed every 10 seconds (600 steps)
        if self.steps > 0 and self.steps % 600 == 0:
            self.spike_speed += 0.01

        # Increase spawn frequency every 5 seconds (300 steps)
        if self.steps > 0 and self.steps % 300 == 0:
            self.current_spike_spawn_rate /= 1.01
            self.current_spike_spawn_rate = max(20, self.current_spike_spawn_rate) # Cap to prevent impossible scenarios

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )
        for spike in self.spikes:
            # Using a slightly smaller rect for spikes for forgiving hitboxes
            spike_rect = pygame.Rect(
                spike['pos'].x - self.SPIKE_SIZE / 2.5,
                spike['pos'].y - self.SPIKE_SIZE / 2.5,
                self.SPIKE_SIZE * 0.8,
                self.SPIKE_SIZE * 0.8
            )
            if player_rect.colliderect(spike_rect):
                return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_spikes()
        if not self.game_over:
            self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_background(self):
        grid_spacing = 50
        # Parallax scrolling effect: grid moves slower than spikes
        offset = -(self.steps * (self.spike_speed / 2.5)) % grid_spacing
        
        for i in range(self.HEIGHT // grid_spacing + 1):
            y = i * grid_spacing
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
        
        for i in range(self.WIDTH // grid_spacing + 2):
            x = i * grid_spacing + offset
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(x), 0), (int(x), self.HEIGHT), 1)

    def _render_player(self):
        player_rect = pygame.Rect(
            int(self.player_pos.x - self.PLAYER_SIZE / 2),
            int(self.player_pos.y - self.PLAYER_SIZE / 2),
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )
        # Glow effect using multiple transparent circles
        glow_size = int(self.PLAYER_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, 30))
        pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, int(glow_size * 0.7), (*self.COLOR_PLAYER_GLOW, 50))
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size, player_rect.centery - glow_size))
        
        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_spikes(self):
        for spike in self.spikes:
            # Draw spike as a triangle pointing left
            center_x, center_y = int(spike['pos'].x), int(spike['pos'].y)
            half_size = self.SPIKE_SIZE / 2
            p1 = (center_x + half_size, center_y)
            p2 = (center_x - half_size, center_y - half_size)
            p3 = (center_x - half_size, center_y + half_size)
            
            # Glow effect
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_SPIKE_GLOW)
            # Main shape
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_SPIKE)

    def _render_ui(self):
        # Time survived
        time_survived = self.steps / self.FPS
        time_text = f"TIME: {time_survived:.2f}s"
        time_surf = self.font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(score_surf, score_rect)

    def _create_death_particles(self):
        for _ in range(40):
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)),
                'lifespan': self.np_random.integers(30, 60),
                'max_lifespan': 60,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            # Fade out effect
            alpha = max(0, int(255 * (p['lifespan'] / p['max_lifespan'])))
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(p['pos'].x),
                int(p['pos'].y),
                int(p['radius']),
                (*self.COLOR_PARTICLE, alpha)
            )

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