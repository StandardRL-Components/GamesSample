import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the basket. Catch fruit and avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Dodge bombs and catch falling fruit in this fast-paced arcade game to reach a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 100
        self.MAX_LIVES = 3

        # Set SDL_VIDEODRIVER to dummy if not already set, for headless operation
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BASKET = (139, 69, 19)
        self.COLOR_BASKET_RIM = (160, 82, 45)
        self.COLOR_BOMB = (40, 40, 40)
        self.COLOR_BOMB_SKULL = (220, 220, 220)
        self.COLOR_TEXT = (240, 240, 240)
        self.FRUIT_COLORS = {
            "apple": (220, 20, 60),
            "banana": (255, 225, 53),
            "grapes": (128, 0, 128),
        }

        # Initialize state variables
        self.np_random = None
        self.basket = None
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.fruit_combo_counter = 0
        self.total_fruits_collected = 0
        self.current_fall_speed = 0
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = 0
        self.initial_bomb_spawn_interval = self.FPS * 10  # 1 bomb every 10 seconds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.np_random = np.random.default_rng(seed)

        player_width = 80
        player_height = 20
        self.basket = pygame.Rect(
            (self.SCREEN_WIDTH - player_width) // 2,
            self.SCREEN_HEIGHT - player_height - 10,
            player_width,
            player_height
        )
        self.player_speed = 12

        self.fruits = []
        self.bombs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.fruit_combo_counter = 0
        self.total_fruits_collected = 0

        self.current_fall_speed = 2.0
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]

            # Update game logic
            self._handle_input(movement)
            self._update_spawners()
            self._update_entities()

            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._update_particles()

        self.steps += 1

        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.total_fruits_collected >= self.WIN_SCORE:
                reward += 100  # Win bonus
            elif self.lives <= 0:
                reward -= 100  # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.basket.x -= self.player_speed
        elif movement == 4:  # Right
            self.basket.x += self.player_speed

        # Clamp basket to screen
        self.basket.x = max(0, min(self.basket.x, self.SCREEN_WIDTH - self.basket.width))

    def _update_spawners(self):
        # Fruit Spawner (1 per second)
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self.fruit_spawn_timer = self.FPS
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            self.fruits.append({
                "rect": pygame.Rect(x_pos, -20, 20, 20),
                "type": fruit_type,
                "rotation": self.np_random.uniform(0, 360)
            })

        # Bomb Spawner (starts at 0.1 per second, increases)
        self.bomb_spawn_timer -= 1
        if self.bomb_spawn_timer <= 0:
            difficulty_factor = 1 + (0.01 * (self.total_fruits_collected // 5))
            current_bomb_interval = self.initial_bomb_spawn_interval / difficulty_factor
            self.bomb_spawn_timer = int(current_bomb_interval)

            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            self.bombs.append({
                "rect": pygame.Rect(x_pos, -30, 30, 30)
            })

    def _update_entities(self):
        # Update fruits
        for fruit in self.fruits[:]:
            fruit["rect"].y += self.current_fall_speed
            fruit["rotation"] += 2
            if fruit["rect"].top > self.SCREEN_HEIGHT:
                self.fruits.remove(fruit)
                self.fruit_combo_counter = 0  # Missed fruit resets combo

        # Update bombs
        for bomb in self.bombs[:]:
            bomb["rect"].y += self.current_fall_speed
            if bomb["rect"].top > self.SCREEN_HEIGHT:
                self.bombs.remove(bomb)

    def _handle_collisions(self):
        reward = 0

        # Fruit collisions
        for fruit in self.fruits[:]:
            if self.basket.colliderect(fruit["rect"]):
                self.fruits.remove(fruit)
                self.score += 1
                self.total_fruits_collected += 1
                self.fruit_combo_counter += 1
                reward += 1

                # Combo bonus
                if self.fruit_combo_counter == 3:
                    self.score += 5
                    reward += 5
                    self.fruit_combo_counter = 0
                    # Combo effect
                    self._create_particles(self.basket.center, 30, (255, 215, 0), 2, 4)

                # Difficulty scaling
                if self.total_fruits_collected > 0 and self.total_fruits_collected % 10 == 0:
                    self.current_fall_speed += 0.2

                # Catch effect
                color = self.FRUIT_COLORS[fruit["type"]]
                self._create_particles(fruit["rect"].center, 15, color, 1, 2)
                # sfx: fruit_catch.wav

        # Bomb collisions
        for bomb in self.bombs[:]:
            if self.basket.colliderect(bomb["rect"]):
                self.bombs.remove(bomb)
                self.lives -= 1
                reward -= 5
                self.fruit_combo_counter = 0  # Bomb hit resets combo
                # Explosion effect
                self._create_particles(bomb["rect"].center, 50, (255, 69, 0), 3, 6)
                # sfx: explosion.wav

        return reward

    def _create_particles(self, pos, count, color, speed_min, speed_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_min, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.lives <= 0 or self.total_fruits_collected >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

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
            "lives": self.lives,
            "fruits_collected": self.total_fruits_collected,
        }

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40))
            color = (*p["color"], max(0, min(255, alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color)

        # Draw basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, self.basket, width=3, border_radius=5)

        # Draw bombs
        for bomb in self.bombs:
            center = bomb["rect"].center
            radius = bomb["rect"].width // 2
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, (60, 60, 60))
            # Skull
            skull_center_x, skull_center_y = center[0], center[1] + 2
            pygame.gfxdraw.filled_circle(self.screen, skull_center_x - 5, skull_center_y - 2, 3, self.COLOR_BOMB_SKULL)
            pygame.gfxdraw.filled_circle(self.screen, skull_center_x + 5, skull_center_y - 2, 3, self.COLOR_BOMB_SKULL)
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (skull_center_x - 5, skull_center_y, 10, 4))

        # Draw fruits
        for fruit in self.fruits:
            self._draw_fruit(self.screen, fruit)

    def _draw_fruit(self, surface, fruit):
        rect = fruit["rect"]
        color = self.FRUIT_COLORS[fruit["type"]]

        if fruit["type"] == "apple":
            pygame.gfxdraw.filled_circle(surface, rect.centerx, rect.centery, rect.width // 2, color)
            # FIX: Convert the generator to a tuple, as pygame color arguments cannot be generators.
            darker_color = tuple(c // 2 for c in color)
            pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, rect.width // 2, darker_color)
            pygame.draw.line(surface, self.COLOR_BASKET, (rect.centerx, rect.top), (rect.centerx + 2, rect.top - 5), 3)
        elif fruit["type"] == "banana":
            points = []
            for i in range(10):
                angle = math.pi * 0.7 + (i / 9) * math.pi * 0.6
                x = rect.centerx + math.cos(angle) * rect.width * 0.8
                y = rect.centery - math.sin(angle) * rect.height * 0.8
                points.append((x, y))
            pygame.draw.lines(surface, color, False, points, 10)
        elif fruit["type"] == "grapes":
            offsets = [(-5, 5), (5, 5), (0, 0), (-5, -5), (5, -5)]
            for dx, dy in offsets:
                pygame.gfxdraw.filled_circle(surface, rect.centerx + dx, rect.centery + dy, rect.width // 3, color)

    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render lives
        for i in range(self.lives):
            bomb_rect = pygame.Rect(self.SCREEN_WIDTH - 40 - (i * 35), 10, 30, 30)
            center = bomb_rect.center
            radius = bomb_rect.width // 2
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, (60, 60, 60))

        # Render Game Over/Win message
        if self.game_over:
            if self.total_fruits_collected >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == "__main__":
    # For local testing, you might want to use a different video driver.
    # 'x11', 'dga', 'fbcon', 'directfb', 'ggi', 'vgl', 'svgalib', 'aalib'
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv(render_mode="rgb_array")

    # --- To play manually ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False

    while True:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Get keyboard input
        keys = pygame.key.get_pressed()
        movement = 0  # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # Construct the action
        action = [movement, 0, 0]  # space and shift are not used

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()