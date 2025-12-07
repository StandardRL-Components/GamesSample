
# Generated: 2025-08-28T02:00:46.077258
# Source Brief: brief_04310.md
# Brief Index: 4310

        
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


class Particle(pygame.sprite.Sprite):
    """A small particle for visual effects."""
    def __init__(self, pos, color, lifespan, velocity, size, gravity=0.1):
        super().__init__()
        self.pos = list(pos)
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.velocity = list(velocity)
        self.size = size
        self.gravity = gravity

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.velocity[1] += self.gravity
        self.lifespan -= 1
        if self.lifespan <= 0:
            self.kill()

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / self.initial_lifespan))))
            current_size = int(self.size * (self.lifespan / self.initial_lifespan))
            if current_size > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (current_size, current_size), current_size)
                surface.blit(temp_surf, (int(self.pos[0] - current_size), int(self.pos[1] - current_size)))

class Fruit(pygame.sprite.Sprite):
    """A falling fruit."""
    def __init__(self, pos_x, color, highlight_color, size, np_random):
        super().__init__()
        self.pos = [pos_x, -size]
        self.color = color
        self.highlight_color = highlight_color
        self.size = size
        self.rect = pygame.Rect(pos_x - size, -size - size, size * 2, size * 2)
        self.angle = np_random.uniform(0, 2 * math.pi)
        self.rotation_speed = np_random.uniform(-0.1, 0.1)

    def update(self, fall_speed):
        self.pos[1] += fall_speed
        self.angle += self.rotation_speed
        self.rect.center = (int(self.pos[0]), int(self.pos[1]))

    def draw(self, surface):
        # Main fruit body
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.size, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.size, self.color)

        # Highlight for 3D effect
        highlight_offset = self.size * 0.3
        highlight_x = int(self.pos[0] - highlight_offset * math.sin(self.angle + math.pi/4))
        highlight_y = int(self.pos[1] - highlight_offset * math.cos(self.angle + math.pi/4))
        pygame.gfxdraw.filled_circle(surface, highlight_x, highlight_y, int(self.size * 0.4), self.highlight_color)
        pygame.gfxdraw.aacircle(surface, highlight_x, highlight_y, int(self.size * 0.4), self.highlight_color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ← to move left, → to move right."
    game_description = "Catch the falling fruit in your basket! Reach 50 points to win, but be careful: 5 misses and you lose."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 50
    MAX_MISSES = 5
    MAX_STEPS = 1000

    CATCHER_WIDTH = 90
    CATCHER_HEIGHT = 20
    CATCHER_Y = HEIGHT - 40
    CATCHER_SPEED = 8.0

    INITIAL_FALL_SPEED = 2.0
    FALL_SPEED_INCREASE = 0.05

    # --- Colors ---
    COLOR_BG_TOP = (25, 20, 50)
    COLOR_BG_BOTTOM = (60, 50, 100)
    COLOR_CATCHER = (255, 215, 0)
    COLOR_CATCHER_RIM = (200, 160, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_MISS_X = (255, 80, 80)
    COLOR_CATCH_PARTICLE = (255, 255, 180)
    COLOR_MISS_PARTICLE = (150, 50, 50)
    FRUIT_COLORS = [
        ((220, 40, 40), (255, 120, 120)),  # Red
        ((40, 200, 40), (120, 255, 120)),  # Green
        ((80, 120, 255), (150, 180, 255)), # Blue
        ((255, 150, 40), (255, 200, 100))  # Orange
    ]

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
        self.font_miss = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self.bg_surface = self._create_gradient_background()

        # Initialize attributes to prevent errors before reset
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.catcher_pos_x = 0
        self.fall_speed = 0
        self.fruit_spawn_timer = 0
        self.fruits = pygame.sprite.Group()
        self.particles = pygame.sprite.Group()
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.catcher_pos_x = self.WIDTH / 2
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.fruit_spawn_timer = 30

        self.fruits.empty()
        self.particles.empty()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.1  # Per-step penalty

        # 1. Handle player input
        movement = action[0]
        if movement == 3:  # Left
            self.catcher_pos_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_pos_x += self.CATCHER_SPEED
        
        self.catcher_pos_x = np.clip(
            self.catcher_pos_x, self.CATCHER_WIDTH / 2, self.WIDTH - self.CATCHER_WIDTH / 2
        )

        # 2. Update game world
        catch_events, miss_events = self._update_world()
        
        for event in catch_events:
            self.score += 1
            reward += 1.0  # Base catch reward
            if event['risky']:
                reward += 5.0  # Risky catch bonus
            self._spawn_particles(event['pos'], self.COLOR_CATCH_PARTICLE, 20, gravity=0.05)
            # Placeholder: pygame.mixer.Sound("catch.wav").play()
        
        for event in miss_events:
            self.misses += 1
            reward -= 1.0  # Miss penalty
            self._spawn_particles(event['pos'], self.COLOR_MISS_PARTICLE, 10, gravity=0.2)
            # Placeholder: pygame.mixer.Sound("miss.wav").play()

        # 3. Update difficulty and step counter
        self.steps += 1
        if self.steps > 0 and self.steps % 50 == 0:
            self.fall_speed += self.FALL_SPEED_INCREASE
        
        # 4. Check for termination
        terminated = self.score >= self.WIN_SCORE or self.misses >= self.MAX_MISSES or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Win bonus
            elif self.misses >= self.MAX_MISSES:
                reward -= 100.0  # Lose penalty

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_world(self):
        catch_events = []
        miss_events = []

        # Spawn new fruits
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.np_random.integers(30, 60)

        self.fruits.update(self.fall_speed)
        self.particles.update()

        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.CATCHER_WIDTH / 2,
            self.CATCHER_Y,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        
        fruits_to_remove = []
        for fruit in self.fruits:
            if catcher_rect.colliderect(fruit.rect):
                is_risky = fruit.rect.bottom > (self.HEIGHT - 50)
                catch_events.append({'pos': fruit.rect.center, 'risky': is_risky})
                fruits_to_remove.append(fruit)
            elif fruit.rect.top > self.HEIGHT:
                miss_events.append({'pos': (fruit.rect.centerx, self.HEIGHT - 10)})
                fruits_to_remove.append(fruit)
        
        for fruit in fruits_to_remove:
            fruit.kill()

        return catch_events, miss_events

    def _spawn_fruit(self):
        pos_x = self.np_random.uniform(20, self.WIDTH - 20)
        color, highlight = random.choice(self.FRUIT_COLORS)
        size = self.np_random.integers(12, 18)
        fruit = Fruit(pos_x, color, highlight, size, self.np_random)
        self.fruits.add(fruit)

    def _spawn_particles(self, pos, color, count, gravity=0.1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed - 1]
            lifespan = self.np_random.integers(20, 40)
            size = self.np_random.integers(2, 5)
            particle = Particle(pos, color, lifespan, velocity, size, gravity=gravity)
            self.particles.add(particle)

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for particle in self.particles:
            particle.draw(self.screen)
            
        for fruit in self.fruits:
            fruit.draw(self.screen)

        # Draw catcher
        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.CATCHER_WIDTH / 2,
            self.CATCHER_Y,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_RIM, catcher_rect, width=3, border_radius=8)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        misses_text = " ".join(["X"] * self.misses)
        misses_surface = self.font_miss.render(misses_text, True, self.COLOR_MISS_X)
        misses_rect = misses_surface.get_rect(topright=(self.WIDTH - 10, 5))
        self.screen.blit(misses_surface, misses_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.misses}

    def _create_gradient_background(self):
        surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(surf, color, (0, y), (self.WIDTH, y))
        return surf

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()