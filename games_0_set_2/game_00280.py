import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Arrow Keys (↑, ↓, ←, →) to pilot your ship and dodge the asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a ship through a dense asteroid field. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    PLAYER_SPEED = 5.0
    PLAYER_RADIUS = 12
    PLAYER_VISUAL_RADIUS = 15 # For glow effect

    ASTEROID_COUNT = 10
    ASTEROID_SPEED_MIN = 1.0
    ASTEROID_SPEED_MAX = 3.0
    ASTEROID_RADIUS_MIN = 10
    ASTEROID_RADIUS_MAX = 30
    
    NEAR_MISS_DISTANCE_FACTOR = 2.5 # multiplier for radii sum to trigger particles

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (100, 150, 255, 50)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_OUTLINE = (90, 100, 110)
    COLOR_UI = (220, 220, 220)
    COLOR_PARTICLE = (200, 220, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables. They will be properly re-initialized in reset(),
        # but we need a valid state for the validation call during __init__.
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.asteroids = []
        self.stars = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # The validation function needs a minimally valid state to run.
        # We also need an RNG for the validation, which is created by super().reset()
        super().reset(seed=0)
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self._init_stars()
        self._init_asteroids()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _init_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.integers(1, 3)
            self.stars.append((x, y, size))

    def _init_asteroids(self):
        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            while True:
                pos = np.array([
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT)
                ], dtype=np.float32)
                
                # Ensure asteroids don't spawn on the player
                if np.linalg.norm(pos - self.player_pos) > 150:
                    break

            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(self.ASTEROID_SPEED_MIN, self.ASTEROID_SPEED_MAX)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            radius = self.np_random.integers(self.ASTEROID_RADIUS_MIN, self.ASTEROID_RADIUS_MAX + 1)
            
            self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius})

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game State ---
        self._update_player(action)
        self._update_asteroids()
        self._update_particles()
        
        # --- Check for Events and Calculate Reward ---
        terminated = False
        reward = 0.1  # Survival reward per frame

        # Check for collision
        collision = self._check_collision()
        if collision:
            # sfx: player_explosion.wav
            reward = -100.0
            terminated = True
            self.game_over = True
        else:
            # Check for near misses to spawn particles for visual feedback
            self._check_near_miss()

        # Check for win condition (time limit reached)
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            # sfx: victory_fanfare.wav
            reward += 100.0
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

    def _update_player(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Screen wrapping
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            # Screen wrapping
            asteroid['pos'][0] %= self.SCREEN_WIDTH
            asteroid['pos'][1] %= self.SCREEN_HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _check_collision(self):
        for asteroid in self.asteroids:
            # Check for collision, considering screen wrapping
            for dx in [-self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH]:
                for dy in [-self.SCREEN_HEIGHT, 0, self.SCREEN_HEIGHT]:
                    wrapped_pos = asteroid['pos'] + np.array([dx, dy])
                    distance = np.linalg.norm(self.player_pos - wrapped_pos)
                    if distance < self.PLAYER_RADIUS + asteroid['radius']:
                        return True
        return False

    def _check_near_miss(self):
        for asteroid in self.asteroids:
            distance = np.linalg.norm(self.player_pos - asteroid['pos'])
            min_dist = self.PLAYER_RADIUS + asteroid['radius']
            if min_dist < distance < min_dist * self.NEAR_MISS_DISTANCE_FACTOR:
                # sfx: near_miss_whoosh.wav
                self._spawn_particles(self.player_pos, asteroid['pos'])

    def _spawn_particles(self, pos1, pos2):
        midpoint = (pos1 + pos2) / 2
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': midpoint.copy(), 'vel': vel, 'lifespan': lifespan})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
        
    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_UI, (x, y), size / 2)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = asteroid['pos'].astype(int)
            radius = int(asteroid['radius'])
            # Draw for screen wrapping
            for dx in [-self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH]:
                for dy in [-self.SCREEN_HEIGHT, 0, self.SCREEN_HEIGHT]:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0] + dx, pos[1] + dy, radius, self.COLOR_ASTEROID)
                    pygame.gfxdraw.aacircle(self.screen, pos[0] + dx, pos[1] + dy, radius, self.COLOR_ASTEROID_OUTLINE)

    def _render_player(self):
        pos = self.player_pos.astype(int)
        
        # Define triangle points (pointing up)
        p1 = (pos[0], pos[1] - self.PLAYER_VISUAL_RADIUS)
        p2 = (pos[0] - self.PLAYER_VISUAL_RADIUS * 0.7, pos[1] + self.PLAYER_VISUAL_RADIUS * 0.7)
        p3 = (pos[0] + self.PLAYER_VISUAL_RADIUS * 0.7, pos[1] + self.PLAYER_VISUAL_RADIUS * 0.7)
        points = [p1, p2, p3]

        # Draw for screen wrapping
        for dx in [-self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH]:
            for dy in [-self.SCREEN_HEIGHT, 0, self.SCREEN_HEIGHT]:
                wrapped_points = [(p[0] + dx, p[1] + dy) for p in points]
                
                # Glow effect
                pygame.gfxdraw.filled_polygon(self.screen, wrapped_points, self.COLOR_PLAYER_GLOW)
                
                # Main ship
                pygame.gfxdraw.aapolygon(self.screen, wrapped_points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, wrapped_points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color = (*self.COLOR_PARTICLE, alpha)
            # Create a temporary surface for transparency
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (2, 2), 2)
            self.screen.blit(particle_surf, (pos[0] - 2, pos[1] - 2))

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        text_surface = self.font.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                end_text = "SURVIVED!"
            else:
                end_text = "GAME OVER"
            
            end_surface = self.font.render(end_text, True, self.COLOR_UI)
            text_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")