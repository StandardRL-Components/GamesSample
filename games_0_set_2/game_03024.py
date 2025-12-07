
# Generated: 2025-08-27T22:08:15.472927
# Source Brief: brief_03024.md
# Brief Index: 3024

        
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
        "Controls: Use arrow keys to move your ship. Avoid the asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a ship through an asteroid field, dodging incoming rocks for 60 seconds to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_NEAR_MISS = (255, 255, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_STAR = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        
        # Game state variables (initialized in reset)
        self.player_pos = None
        self.player_radius = 12
        self.player_speed = 5.0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_starfield()
        # Initialize state variables
        self.reset()
    
    def _generate_starfield(self):
        """Creates a static starfield for the background."""
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.randint(1, 2)
            self.stars.append(((x, y), size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8], dtype=np.float32)
        self.asteroids = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, return a terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Update game logic
        self._update_player(movement)
        self._update_asteroids()
        self._spawn_asteroids()
        self._update_particles()
        
        # Calculate reward and check for termination
        reward = 0.01  # Survival reward per frame
        
        dodged_reward = self._handle_offscreen_asteroids()
        reward += dodged_reward
        self.score += int(dodged_reward)
        
        terminated = False
        
        # Check termination conditions
        if self._check_collision():
            # sfx: player_explosion.wav
            reward = -100.0
            terminated = True
            self.game_over = True
        
        if not terminated and self.steps >= self.MAX_STEPS:
            # sfx: victory_fanfare.wav
            reward = 100.0
            terminated = True
            self.game_over = True

        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed
            
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.SCREEN_HEIGHT - self.player_radius)

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            distance = np.linalg.norm(self.player_pos - asteroid['pos'])
            # Trigger near-miss effect
            if distance < (self.player_radius + asteroid['radius']) * 1.8:
                 self._create_near_miss_effect(asteroid['pos'])

    def _handle_offscreen_asteroids(self):
        dodged_reward = 0
        asteroids_on_screen = []
        for asteroid in self.asteroids:
            # If asteroid is still on screen (with buffer)
            if asteroid['pos'][1] < self.SCREEN_HEIGHT + asteroid['radius']:
                asteroids_on_screen.append(asteroid)
            else:
                # sfx: asteroid_pass.wav
                dodged_reward += 1.0  # Reward for dodging
        self.asteroids = asteroids_on_screen
        return dodged_reward

    def _spawn_asteroids(self):
        base_rate = 1.0 / self.FPS
        final_rate = 5.0 / self.FPS
        progress = min(1.0, self.steps / self.MAX_STEPS)
        current_rate = base_rate + (final_rate - base_rate) * progress

        if self.np_random.random() < current_rate:
            radius = self.np_random.integers(10, 31)
            x = self.np_random.uniform(radius, self.SCREEN_WIDTH - radius)
            pos = np.array([x, -radius], dtype=np.float32)
            
            can_spawn = True
            for ast in self.asteroids:
                if ast['pos'][1] < radius * 2:
                    dist = np.linalg.norm(pos - ast['pos'])
                    if dist < radius + ast['radius'] + 10: # Add buffer
                        can_spawn = False
                        break
            
            if can_spawn:
                angle = self.np_random.uniform(-0.3, 0.3)
                speed = self.np_random.uniform(1.5, 4.0)
                vel = np.array([math.sin(angle) * speed, math.cos(angle) * speed], dtype=np.float32)
                self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['radius'] += p['growth']
            p['life'] -= 1

    def _create_near_miss_effect(self, pos):
        if len(self.particles) < 50:
            self.particles.append({
                'pos': pos.copy(), 'radius': 5, 'life': 15, 'growth': 0.5
            })

    def _check_collision(self):
        for asteroid in self.asteroids:
            distance = np.linalg.norm(self.player_pos - asteroid['pos'])
            if distance < self.player_radius + asteroid['radius']:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, round((self.MAX_STEPS - self.steps) / self.FPS, 1))
        }

    def _render_game(self):
        self._draw_starfield()
        self._draw_particles()
        self._draw_asteroids()
        self._draw_player()

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _draw_starfield(self):
        for pos, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, pos, size)

    def _draw_player(self):
        p = self.player_pos
        r = self.player_radius
        points = [
            (p[0], p[1] - r),
            (p[0] - r * 0.7, p[1] + r * 0.7),
            (p[0] + r * 0.7, p[1] + r * 0.7)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_asteroids(self):
        for asteroid in self.asteroids:
            pos = asteroid['pos'].astype(int)
            radius = int(asteroid['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _draw_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            radius = int(p['radius'])
            alpha = max(0, min(255, int(255 * (p['life'] / 15.0))))
            color = (*self.COLOR_NEAR_MISS, alpha)
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

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