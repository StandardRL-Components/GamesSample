import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:15:04.891823
# Source Brief: brief_00949.md
# Brief Index: 949
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for falling shapes to keep their data organized
class Shape:
    def __init__(self, screen_width):
        self.size = random.randint(15, 35)
        self.rect = pygame.Rect(
            random.randint(0, screen_width - self.size),
            -self.size,
            self.size,
            self.size
        )
        self.color = random.choice([
            (3, 252, 240),   # Bright Cyan
            (252, 3, 186),   # Bright Magenta
            (252, 240, 3),   # Bright Yellow
            (57, 255, 20)    # Neon Green
        ])
        self.velocity = 0.0

# Helper class for visual feedback particles
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = random.randint(20, 40)
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1
        return self.lifespan > 0 and self.radius > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Catch the falling neon shapes with your platform to score points. "
        "Miss three shapes and the game is over."
    )
    user_guide = "Controls: Use the ← and → arrow keys to move the platform left and right."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    MAX_MISSES = 3

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLATFORM = (255, 255, 255)
    COLOR_PLATFORM_GLOW = (200, 200, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_X_ACTIVE = (255, 50, 50)
    COLOR_UI_X_INACTIVE = (80, 80, 80)

    # Platform Physics
    PLATFORM_WIDTH = 100
    PLATFORM_HEIGHT = 15
    PLATFORM_ACCEL = 1.2
    PLATFORM_FRICTION = 0.92
    PLATFORM_MAX_SPEED = 12.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.missed_count = 0
        self.terminated = False
        self.platform = None
        self.platform_vel = 0.0
        self.shapes = []
        self.particles = []
        self.spawn_timer = 0
        self.initial_spawn_interval = 45
        self.min_spawn_interval = 15
        self.spawn_interval = self.initial_spawn_interval
        self.initial_base_speed = 1.5
        self.max_base_speed = 8.0
        self.base_speed = self.initial_base_speed
        
        # self.reset() # reset is called by the agent, no need to call it here
        # self.validate_implementation() # This is for dev, not needed in final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.missed_count = 0
        self.terminated = False

        self.platform = pygame.Rect(
            (self.SCREEN_WIDTH - self.PLATFORM_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT - 10,
            self.PLATFORM_WIDTH,
            self.PLATFORM_HEIGHT
        )
        self.platform_vel = 0.0

        self.shapes = []
        self.particles = []

        self.spawn_interval = self.initial_spawn_interval
        self.spawn_timer = self.spawn_interval
        self.base_speed = self.initial_base_speed

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # Unpack factorized action
        movement = action[0]
        # space_held and shift_held are unused per brief
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        if movement == 3:  # Left
            self.platform_vel -= self.PLATFORM_ACCEL
        elif movement == 4:  # Right
            self.platform_vel += self.PLATFORM_ACCEL
        
        # Apply friction if no horizontal input
        if movement not in [3, 4]:
            self.platform_vel *= self.PLATFORM_FRICTION

        # Clamp velocity and update position, ensuring it's an int for rect
        self.platform_vel = np.clip(self.platform_vel, -self.PLATFORM_MAX_SPEED, self.PLATFORM_MAX_SPEED)
        self.platform.x += int(self.platform_vel)
        self.platform.x = np.clip(self.platform.x, 0, self.SCREEN_WIDTH - self.PLATFORM_WIDTH)

        self._update_difficulty()
        self._update_spawner()
        reward += self._update_shapes()
        self._update_particles()

        # Termination Check
        truncated = False
        if self.missed_count >= self.MAX_MISSES:
            self.terminated = True
            reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            self.terminated = True
            truncated = True # truncated because of time limit
            reward = 10.0

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info()
        )

    def _update_difficulty(self):
        # Increase shape speed over time
        speed_increase = (self.steps // 500) * 0.1
        self.base_speed = min(self.max_base_speed, self.initial_base_speed + speed_increase)

        # Decrease spawn interval over time
        interval_decrease = self.steps // 200
        self.spawn_interval = max(self.min_spawn_interval, self.initial_spawn_interval - interval_decrease)

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            new_shape = Shape(self.SCREEN_WIDTH)
            new_shape.velocity = self.base_speed + random.uniform(0, 2.0)
            self.shapes.append(new_shape)
            self.spawn_timer = int(self.spawn_interval)

    def _update_shapes(self):
        step_reward = 0.0
        shapes_to_remove = []
        
        is_colliding = False
        for shape in self.shapes:
            shape.rect.y += int(shape.velocity)
            
            if self.platform.colliderect(shape.rect):
                # Brief: +10 for catching a shape
                step_reward += 10.0
                # Brief: score is shape's area
                self.score += shape.rect.width * shape.rect.height
                shapes_to_remove.append(shape)
                self._create_particles(shape.rect.centerx, shape.rect.top, shape.color)
                # Sound placeholder: # sfx_catch.play()
                is_colliding = True

            elif shape.rect.top > self.SCREEN_HEIGHT:
                self.missed_count += 1
                shapes_to_remove.append(shape)
                # Sound placeholder: # sfx_miss.play()
        
        # Brief: +1 reward per frame of contact.
        # This is interpreted as +1 if any shape is caught in this frame.
        if is_colliding:
            step_reward += 1.0

        if shapes_to_remove:
            self.shapes = [s for s in self.shapes if s not in shapes_to_remove]
            
        return step_reward

    def _create_particles(self, x, y, color, count=20):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _update_particles(self):
        # Update and filter out dead particles in one list comprehension
        self.particles = [p for p in self.particles if p.update()]

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
            "missed": self.missed_count,
        }
        
    def _render_game(self):
        # Render particles
        for p in self.particles:
            if p.radius > 0:
                alpha = max(0, min(255, int(255 * (p.lifespan / 40.0))))
                p_color = (p.color[0], p.color[1], p.color[2], alpha)
                pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), int(p.radius), p_color)
                pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(p.radius), p_color)

        # Render shapes
        for shape in self.shapes:
            pygame.draw.rect(self.screen, shape.color, shape.rect, border_radius=3)
            inner_rect = shape.rect.inflate(-4, -4)
            s = pygame.Surface(inner_rect.size, pygame.SRCALPHA)
            s.fill((255, 255, 255, 30))
            self.screen.blit(s, inner_rect.topleft)

        # Render platform with glow for better visibility and feel
        glow_rect = self.platform.inflate(8, 8)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLATFORM_GLOW, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, self.platform, border_radius=5)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Render missed markers as 'X's for clear visual feedback
        for i in range(self.MAX_MISSES):
            color = self.COLOR_UI_X_ACTIVE if i < self.missed_count else self.COLOR_UI_X_INACTIVE
            x_pos = self.SCREEN_WIDTH - 30 - (i * 35)
            miss_text = self.font_large.render("X", True, color)
            self.screen.blit(miss_text, (x_pos, 15))
            
        if self.terminated and self.missed_count >= self.MAX_MISSES:
            end_text = self.font_large.render("GAME OVER", True, self.COLOR_UI_X_ACTIVE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # The reset method is called before this, so platform is not None
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run when the environment is used by an agent
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Override pygame screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Catcher")
    
    total_reward = 0.0
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Pygame uses (width, height) but obs is (height, width, 3), so we transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(60)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    env.close()