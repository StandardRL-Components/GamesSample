
# Generated: 2025-08-27T19:14:56.923027
# Source Brief: brief_02094.md
# Brief Index: 2094

        
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
        "Controls: ←→ to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch the falling fruit to score points. Miss too many and it's game over!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.BASKET_WIDTH, self.BASKET_HEIGHT = 80, 20
        self.BASKET_SPEED = 12
        self.FRUIT_RADIUS = 12
        self.MAX_STEPS = 1500 # Increased to allow for longer games
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 10

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_BASKET = (255, 180, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.FRUIT_TYPES = {
            "apple": {"color": (220, 30, 30), "score": 1, "reward": 1},
            "orange": {"color": (255, 140, 0), "score": 1, "reward": 1},
            "pear": {"color": (210, 230, 50), "score": 1, "reward": 1},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 36)

        # State variables (initialized in reset)
        self.basket_rect = None
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.game_over = False
        self.win_state = False
        self.base_fruit_speed = 0.0
        self.fruit_spawn_timer = 0
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.game_over = False
        self.win_state = False

        self.basket_rect = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) / 2,
            self.HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT,
        )

        self.fruits = []
        self.particles = []
        self.base_fruit_speed = 2.0
        self.fruit_spawn_timer = 30  # Spawn first fruit after 1s

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            terminated = self._check_termination()

            if terminated:
                self.game_over = True
                if self.win_state:
                    reward += 100  # Win reward
                else:
                    reward -= 100  # Loss reward

        # Per-frame penalty for existing fruits
        reward -= 0.1 * len(self.fruits)

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.basket_rect.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_rect.x += self.BASKET_SPEED

        # Clamp basket to screen
        self.basket_rect.left = max(0, self.basket_rect.left)
        self.basket_rect.right = min(self.WIDTH, self.basket_rect.right)

    def _update_game_state(self):
        frame_reward = 0

        # Update and check fruits
        for fruit in self.fruits[:]:
            fruit["pos"][1] += fruit["speed"]
            fruit["rect"].centery = int(fruit["pos"][1])

            if self.basket_rect.colliderect(fruit["rect"]):
                # sfx: catch_fruit.wav
                self.fruits.remove(fruit)
                self.fruits_caught += 1
                self.score += fruit["type_info"]["score"]
                frame_reward += fruit["type_info"]["reward"]
                
                if fruit["speed"] > 5:
                    frame_reward += 5 # Fast fruit bonus reward
                    self.score += 5 # Fast fruit bonus score

                self._create_particles(fruit["rect"].center, fruit["type_info"]["color"])
                
                # Increase difficulty every 10 fruits
                if self.fruits_caught > 0 and self.fruits_caught % 10 == 0:
                    self.base_fruit_speed += 0.2

            elif fruit["rect"].top > self.HEIGHT:
                # sfx: miss_fruit.wav
                self.fruits.remove(fruit)
                self.fruits_missed += 1

        # Spawn new fruits
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.rng.integers(30, 90) - int(self.base_fruit_speed * 5)

        # Update particles
        self._update_particles()
        
        return frame_reward

    def _spawn_fruit(self):
        fruit_name = self.rng.choice(list(self.FRUIT_TYPES.keys()))
        type_info = self.FRUIT_TYPES[fruit_name]
        
        x_pos = self.rng.integers(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS)
        speed = self.base_fruit_speed + self.rng.random() * 0.5

        fruit_rect = pygame.Rect(0, 0, self.FRUIT_RADIUS * 2, self.FRUIT_RADIUS * 2)
        fruit_rect.center = (x_pos, -self.FRUIT_RADIUS)

        self.fruits.append({
            "pos": [float(x_pos), float(-self.FRUIT_RADIUS)],
            "rect": fruit_rect,
            "speed": speed,
            "type_info": type_info,
        })

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2.5 + 1.0
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.fruits_caught >= self.WIN_CONDITION:
            self.win_state = True
            return True
        if self.fruits_missed >= self.LOSS_CONDITION:
            self.win_state = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles (behind other elements)
        for p in self.particles:
            size = max(0, int(p["life"] * 0.2))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit["rect"].centerx), int(fruit["rect"].centery))
            color = fruit["type_info"]["color"]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, color)

        # Draw basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_rect, border_radius=5)

    def _render_ui(self):
        # Score display
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Missed display
        miss_color = (255, 100, 100) if self.fruits_missed > self.LOSS_CONDITION / 2 else self.COLOR_TEXT
        miss_text = self.font_small.render(f"Missed: {self.fruits_missed}/{self.LOSS_CONDITION}", True, miss_color)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(miss_text, miss_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                end_text_str = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 50, 50)
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed,
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

        print("✓ Implementation validated successfully")