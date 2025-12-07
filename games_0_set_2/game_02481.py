import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ← to move the basket left, → to move it right. Catch the falling fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a basket for points before they hit the ground. "
        "Different colored fruits are worth different points. "
        "Win by catching 15 fruits, but lose if you miss 5 or run out of time."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.MAX_LIVES = 5
        self.TOTAL_FRUIT_TO_CATCH = 15

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (144, 238, 144)  # Light Green
        self.COLOR_BASKET = (139, 69, 19)  # Saddle Brown
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.FRUIT_TYPES = {
            "green": {"color": (50, 205, 50), "value": 1, "radius": 12},
            "yellow": {"color": (255, 255, 0), "value": 2, "radius": 14},
            "red": {"color": (255, 69, 0), "value": 3, "radius": 16},
        }

        # Gameplay
        self.BASKET_SPEED = 8.0
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_ACCEL = 0.05
        self.FALL_SPEED_ACCEL_INTERVAL = 50  # frames
        self.INITIAL_SPAWN_INTERVAL = 2 * self.FPS  # frames
        self.SPAWN_RATE_ACCEL = 1 * self.FPS  # frames
        self.SPAWN_RATE_ACCEL_INTERVAL = 10 * self.FPS  # frames

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.time_remaining = 0
        self.basket_pos = [0, 0]
        self.fruits = []
        self.particles = []
        self.fruit_fall_speed = 0.0
        self.fruit_spawn_timer = 0
        self.fruit_spawn_interval = 0
        self.fall_speed_increase_timer = 0
        self.spawn_rate_increase_timer = 0
        self.fruit_caught_count = 0
        self.total_fruit_spawned = 0
        self.random = None
        self.game_over_message = ""

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.time_remaining = self.MAX_STEPS

        self.basket_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.fruits = []
        self.particles = []

        self.fruit_fall_speed = self.INITIAL_FALL_SPEED
        self.fruit_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.fruit_spawn_timer = self.fruit_spawn_interval // 2
        self.fall_speed_increase_timer = self.FALL_SPEED_ACCEL_INTERVAL
        self.spawn_rate_increase_timer = self.SPAWN_RATE_ACCEL_INTERVAL

        self.fruit_caught_count = 0
        self.total_fruit_spawned = 0
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False

        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos[0] -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos[0] += self.BASKET_SPEED

        # Clamp basket position
        basket_width = 80
        self.basket_pos[0] = max(basket_width // 2, min(self.WIDTH - basket_width // 2, self.basket_pos[0]))

        # --- Continuous Reward ---
        closest_fruit = self._get_closest_fruit()
        if closest_fruit:
            fruit_x = closest_fruit['pos'][0]
            basket_x = self.basket_pos[0]
            is_moving_left = movement == 3
            is_moving_right = movement == 4

            if (is_moving_left and fruit_x < basket_x) or (is_moving_right and fruit_x > basket_x):
                reward += 0.1  # Moving towards fruit
            elif (is_moving_left and fruit_x > basket_x) or (is_moving_right and fruit_x < basket_x):
                reward -= 0.1  # Moving away from fruit

        # --- Game Logic ---
        self._update_timers()
        self._update_difficulty()
        self._spawn_fruit()

        catch_reward, miss_penalty = self._update_fruits()
        reward += catch_reward + miss_penalty

        self._update_particles()

        # --- Termination Check ---
        if self.lives <= 0:
            terminated = True
            reward -= 100  # Loss penalty
            self.game_over_message = "GAME OVER"
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 100  # Loss penalty
            self.game_over_message = "TIME'S UP!"
        elif self.fruit_caught_count >= self.TOTAL_FRUIT_TO_CATCH:
            terminated = True
            reward += 100  # Win bonus
            self.game_over_message = "YOU WIN!"

        if terminated:
            self.game_over = True

        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        self.time_remaining -= 1
        self.fruit_spawn_timer -= 1
        self.fall_speed_increase_timer -= 1
        self.spawn_rate_increase_timer -= 1

    def _update_difficulty(self):
        if self.fall_speed_increase_timer <= 0:
            self.fruit_fall_speed += self.FALL_SPEED_ACCEL
            self.fall_speed_increase_timer = self.FALL_SPEED_ACCEL_INTERVAL

        if self.spawn_rate_increase_timer <= 0:
            self.fruit_spawn_interval = max(self.FPS // 2, self.fruit_spawn_interval - self.SPAWN_RATE_ACCEL // 3)
            self.spawn_rate_increase_timer = self.SPAWN_RATE_ACCEL_INTERVAL

    def _spawn_fruit(self):
        if self.fruit_spawn_timer <= 0 and self.total_fruit_spawned < self.TOTAL_FRUIT_TO_CATCH:
            self.fruit_spawn_timer = self.fruit_spawn_interval
            self.total_fruit_spawned += 1

            fruit_name = self.random.choices(list(self.FRUIT_TYPES.keys()), weights=[0.5, 0.3, 0.2], k=1)[0]
            fruit_info = self.FRUIT_TYPES[fruit_name]

            new_fruit = {
                'pos': [self.random.uniform(fruit_info['radius'], self.WIDTH - fruit_info['radius']), -fruit_info['radius']],
                'type': fruit_name,
                'value': fruit_info['value'],
                'color': fruit_info['color'],
                'radius': fruit_info['radius'],
            }
            self.fruits.append(new_fruit)

    def _update_fruits(self):
        catch_reward = 0
        miss_penalty = 0
        basket_rect = pygame.Rect(self.basket_pos[0] - 40, self.basket_pos[1] - 10, 80, 20)

        for fruit in self.fruits[:]:
            fruit['pos'][1] += self.fruit_fall_speed

            fruit_rect = pygame.Rect(fruit['pos'][0] - fruit['radius'], fruit['pos'][1] - fruit['radius'], fruit['radius'] * 2, fruit['radius'] * 2)

            if basket_rect.colliderect(fruit_rect):
                # --- Catch ---
                self.score += fruit['value']
                catch_reward += fruit['value']
                self.fruit_caught_count += 1
                self._create_particles(fruit['pos'], fruit['color'], 20)
                self.fruits.remove(fruit)
            elif fruit['pos'][1] > self.HEIGHT + fruit['radius']:
                # --- Miss ---
                self.lives -= 1
                miss_penalty -= 5
                self._create_particles([fruit['pos'][0], self.HEIGHT - 5], (100, 100, 255), 15, is_splash=True)
                self.fruits.remove(fruit)

        return catch_reward, miss_penalty

    def _create_particles(self, pos, color, count, is_splash=False):
        for _ in range(count):
            if is_splash:
                vel = [self.random.uniform(-1.5, 1.5), self.random.uniform(-4, -1)]
            else:
                angle = self.random.uniform(0, 2 * math.pi)
                speed = self.random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifetime': self.random.randint(15, 30),
                'color': color,
                'radius': self.random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _get_closest_fruit(self):
        if not self.fruits:
            return None

        basket_x = self.basket_pos[0]
        closest_fruit = min(self.fruits, key=lambda f: abs(f['pos'][0] - basket_x))
        return closest_fruit

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = fruit['radius']
            color = fruit['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

            # Add a small highlight for 3D effect
            highlight_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 3, (255, 255, 255, 100))

        # Render basket
        basket_rect = (int(self.basket_pos[0] - 40), int(self.basket_pos[1] - 10), 80, 20)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, tuple(max(0, c - 40) for c in self.COLOR_BASKET), basket_rect, 3, border_radius=5)

    def _render_ui(self):
        # Score
        self._draw_text(f"Score: {self.score}", (10, 10), self.font_medium)

        # Lives
        lives_text = "Lives: " + "♥" * self.lives
        self._draw_text(lives_text, (self.WIDTH - 150, 10), self.font_medium, color=(255, 105, 180))

        # Timer
        time_str = f"{self.time_remaining // self.FPS:02d}"
        self._draw_text(time_str, (self.WIDTH // 2, 10), self.font_large, center=True)

        # Game Over Message
        if self.game_over:
            self._draw_text(self.game_over_message, (self.WIDTH // 2, self.HEIGHT // 2 - 50), self.font_large, center=True)
            final_score_text = f"Final Score: {self.score}"
            self._draw_text(final_score_text, (self.WIDTH // 2, self.HEIGHT // 2), self.font_medium, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        if shadow_color is None:
            shadow_color = self.COLOR_TEXT_SHADOW

        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        shadow_rect = shadow_surf.get_rect()

        if center:
            text_rect.center = pos
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft = pos
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_remaining": self.time_remaining,
            "fruit_caught": self.fruit_caught_count,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To use, you might need to unset the dummy video driver
    # and install pygame. For example:
    # pip install pygame
    # unset SDL_VIDEODRIVER
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    total_reward = 0

    while not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        action = [movement, 0, 0]  # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()