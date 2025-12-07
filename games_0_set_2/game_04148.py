
# Generated: 2025-08-28T01:33:31.512650
# Source Brief: brief_04148.md
# Brief Index: 4148

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, rng):
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 5)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = rng.integers(20, 40)  # Lifespan in frames
        self.radius = rng.integers(4, 8)
        self.color = (255, rng.integers(180, 220), 0)  # Orange-yellow

    def update(self):
        """Move the particle and reduce its lifespan."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.radius -= 0.15
        return self.life > 0 and self.radius > 0

    def draw(self, surface):
        """Draw the particle on the screen."""
        if self.radius > 0:
            pygame.gfxdraw.filled_circle(
                surface, int(self.x), int(self.y), int(self.radius), self.color
            )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the net and catch the fish."
    )

    game_description = (
        "Catch 20 fish with your net. If you miss 3 fish in a row, the game is over. Fish get faster as you catch more!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 20
        self.MAX_MISSES = 3
        self.MAX_STEPS = 1000
        self.NET_SPEED = 9.0
        self.NET_RADIUS = 30
        self.BASE_FISH_SPEED = 2.0
        self.FISH_SPEED_INCREASE = 0.05
        self.FISH_SIZE = (30, 15)

        # Colors
        self.COLOR_BG = (135, 206, 235)  # SkyBlue
        self.COLOR_NET_FILL = (50, 205, 50, 150) # LimeGreen with alpha
        self.COLOR_NET_OUTLINE = (0, 100, 0) # DarkGreen
        self.COLOR_FISH = (255, 165, 0) # Orange
        self.COLOR_FISH_EYE = (0, 0, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0, 100)
        self.COLOR_MISS_TEXT = (255, 69, 0) # Tomato Red
        self.COLOR_WIN_TEXT = (50, 205, 50) # LimeGreen

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 36)

        # State Variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.consecutive_misses = 0
        self.game_over = False
        self.net_pos = np.zeros(2, dtype=np.float32)
        self.fish_pos = np.zeros(2, dtype=np.float32)
        self.fish_vel = np.zeros(2, dtype=np.float32)
        self.fish_speed = 0.0
        self.last_dist_to_fish = 0.0
        self.particles = []

        # This call is not strictly necessary for the env to be used, but
        # it ensures the env is in a valid state for the validation check.
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Reset Game State
        self.steps = 0
        self.score = 0
        self.consecutive_misses = 0
        self.game_over = False
        self.particles = []

        # Reset Net Position
        self.net_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        # Spawn First Fish
        self._spawn_fish()

        # Reset distance for reward calculation
        self.last_dist_to_fish = self._get_dist_to_fish()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        dist_before = self._get_dist_to_fish()
        
        self._update_net(movement)
        self._update_fish()
        self._update_particles()
        
        dist_after = self._get_dist_to_fish()

        # Continuous reward for moving closer/further
        if dist_after < dist_before:
            reward += 0.01
        elif dist_after > dist_before:
            reward -= 0.01

        # Event-based rewards and state changes
        caught_fish = self._check_catch()
        if caught_fish:
            reward += 1
            # sfx: catch_fish.wav
        else:
            missed_fish = self._check_miss()
            if missed_fish:
                reward -= 1
                # sfx: miss.wav
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 10
            elif self.consecutive_misses >= self.MAX_MISSES:
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_net(self, movement):
        if movement == 1: self.net_pos[1] -= self.NET_SPEED
        elif movement == 2: self.net_pos[1] += self.NET_SPEED
        elif movement == 3: self.net_pos[0] -= self.NET_SPEED
        elif movement == 4: self.net_pos[0] += self.NET_SPEED
        
        self.net_pos[0] = np.clip(self.net_pos[0], 0, self.WIDTH)
        self.net_pos[1] = np.clip(self.net_pos[1], 0, self.HEIGHT)

    def _update_fish(self):
        self.fish_pos += self.fish_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _check_catch(self):
        if self._get_dist_to_fish() < self.NET_RADIUS:
            self.score += 1
            self.consecutive_misses = 0
            self._create_particles(self.fish_pos[0], self.fish_pos[1])
            self._spawn_fish()
            return True
        return False

    def _check_miss(self):
        is_off_screen = not (
            -self.FISH_SIZE[0] < self.fish_pos[0] < self.WIDTH + self.FISH_SIZE[0] and
            -self.FISH_SIZE[0] < self.fish_pos[1] < self.HEIGHT + self.FISH_SIZE[0]
        )
        if is_off_screen:
            self.consecutive_misses += 1
            self._spawn_fish()
            return True
        return False

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.consecutive_misses >= self.MAX_MISSES or
            self.steps >= self.MAX_STEPS
        )

    def _spawn_fish(self):
        self.fish_speed = self.BASE_FISH_SPEED + (self.score // 5) * self.FISH_SPEED_INCREASE

        edge = self.np_random.integers(4)
        if edge == 0:  # Top
            start_pos = [self.np_random.uniform(0, self.WIDTH), -self.FISH_SIZE[0]]
            target_pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.FISH_SIZE[0]]
        elif edge == 1:  # Bottom
            start_pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.FISH_SIZE[0]]
            target_pos = [self.np_random.uniform(0, self.WIDTH), -self.FISH_SIZE[0]]
        elif edge == 2:  # Left
            start_pos = [-self.FISH_SIZE[0], self.np_random.uniform(0, self.HEIGHT)]
            target_pos = [self.WIDTH + self.FISH_SIZE[0], self.np_random.uniform(0, self.HEIGHT)]
        else:  # Right
            start_pos = [self.WIDTH + self.FISH_SIZE[0], self.np_random.uniform(0, self.HEIGHT)]
            target_pos = [-self.FISH_SIZE[0], self.np_random.uniform(0, self.HEIGHT)]

        self.fish_pos = np.array(start_pos, dtype=np.float32)
        
        direction = np.array(target_pos) - self.fish_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.fish_vel = (direction / norm) * self.fish_speed
        else:
            self.fish_vel = np.array([self.fish_speed, 0], dtype=np.float32)

    def _get_dist_to_fish(self):
        return np.linalg.norm(self.net_pos - self.fish_pos)

    def _create_particles(self, x, y):
        for _ in range(25):
            self.particles.append(Particle(x, y, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)

        fish_rect = pygame.Rect(0, 0, self.FISH_SIZE[0], self.FISH_SIZE[1])
        fish_rect.center = (int(self.fish_pos[0]), int(self.fish_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_FISH, fish_rect, border_radius=5)
        
        eye_dir = self.fish_vel / (np.linalg.norm(self.fish_vel) + 1e-6)
        eye_pos = self.fish_pos + eye_dir * 8
        pygame.draw.circle(self.screen, self.COLOR_FISH_EYE, (int(eye_pos[0]), int(eye_pos[1])), 2)

        net_x, net_y = int(self.net_pos[0]), int(self.net_pos[1])
        temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surface, net_x, net_y, self.NET_RADIUS, self.COLOR_NET_FILL)
        self.screen.blit(temp_surface, (0,0))
        pygame.gfxdraw.aacircle(self.screen, net_x, net_y, self.NET_RADIUS, self.COLOR_NET_OUTLINE)
        pygame.gfxdraw.aacircle(self.screen, net_x, net_y, self.NET_RADIUS-1, self.COLOR_NET_OUTLINE)

    def _render_ui(self):
        def draw_text_with_shadow(text, font, color, center_pos):
            shadow_surf = font.render(text, True, self.COLOR_UI_SHADOW)
            text_surf = font.render(text, True, color)
            shadow_rect = shadow_surf.get_rect(center=(center_pos[0] + 2, center_pos[1] + 2))
            text_rect = text_surf.get_rect(center=center_pos)
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

        score_str = f"Score: {self.score}"
        draw_text_with_shadow(score_str, self.font_small, self.COLOR_UI_TEXT, (len(score_str)*8 + 20, 25))

        misses_str = f"Misses: {self.consecutive_misses}/{self.MAX_MISSES}"
        miss_color = self.COLOR_MISS_TEXT if self.consecutive_misses > 0 else self.COLOR_UI_TEXT
        misses_width = self.font_small.render(misses_str, True, miss_color).get_width()
        draw_text_with_shadow(misses_str, self.font_small, miss_color, (self.WIDTH - misses_width/2 - 20, 25))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text, color = "YOU WIN!", self.COLOR_WIN_TEXT
            else:
                end_text, color = "GAME OVER", self.COLOR_MISS_TEXT
            draw_text_with_shadow(end_text, self.font_large, color, (self.WIDTH / 2, self.HEIGHT / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "consecutive_misses": self.consecutive_misses,
            "fish_pos": tuple(self.fish_pos.astype(int)),
            "net_pos": tuple(self.net_pos.astype(int)),
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        # Test fish speed progression
        self.reset()
        self.score = 4
        self._spawn_fish()
        assert math.isclose(self.fish_speed, self.BASE_FISH_SPEED), f"Speed was {self.fish_speed}"
        self.score = 5
        self._spawn_fish()
        assert math.isclose(self.fish_speed, self.BASE_FISH_SPEED + self.FISH_SPEED_INCREASE), f"Speed was {self.fish_speed}"

        # Test termination conditions
        self.reset()
        self.consecutive_misses = self.MAX_MISSES - 1
        self.fish_pos = np.array([-100, -100], dtype=np.float32) # Force a miss
        _, _, terminated, _, _ = self.step(self.action_space.sample())
        assert terminated, "Game should terminate after MAX_MISSES"
        assert self.consecutive_misses == self.MAX_MISSES

        self.reset()
        self.score = self.WIN_SCORE - 1
        self.net_pos = np.copy(self.fish_pos) # Force a catch
        _, _, terminated, _, _ = self.step(self.action_space.sample())
        assert terminated, "Game should terminate after WIN_SCORE"
        assert self.score == self.WIN_SCORE
        
        print("✓ Implementation validated successfully")