
# Generated: 2025-08-28T03:34:43.755741
# Source Brief: brief_04972.md
# Brief Index: 4972

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Fruit:
    """Represents a single falling fruit."""
    def __init__(self, x, y, speed, fruit_type, radius=12):
        self.pos = pygame.Vector2(x, y)
        self.speed = speed
        self.type = fruit_type
        self.radius = radius

    def update(self):
        self.pos.y += self.speed

    def get_rect(self):
        return pygame.Rect(
            self.pos.x - self.radius,
            self.pos.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

class Particle:
    """Represents a single particle for effects."""
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 5)
        self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.lifespan = random.randint(20, 40)
        self.color = color
        self.radius = self.lifespan / 8

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = self.lifespan / 8

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the basket. Catch the fruit!"
    )

    game_description = (
        "Catch falling fruit in a moving basket to achieve the high score before missing too many."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (0, 0, 48)
    COLOR_BG_BOTTOM = (32, 64, 128)
    COLOR_BASKET = (51, 255, 51)
    COLOR_BASKET_OUTLINE = (30, 150, 30)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 255, 150)
    FRUIT_COLORS = [(255, 51, 51), (255, 255, 51), (255, 153, 51)] # Red, Yellow, Orange
    FRUIT_STEM_COLOR = (139, 69, 19)

    # Game parameters
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 12
    BASE_FRUIT_SPEED = 2.0
    BASE_SPAWN_RATE = 0.05
    MAX_STEPS = 1000
    WIN_CONDITION = 15
    LOSE_CONDITION = 5
    
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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.basket = None
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.current_fruit_speed = self.BASE_FRUIT_SPEED
        self.current_spawn_rate = self.BASE_SPAWN_RATE

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.BASKET_WIDTH // 2,
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fruits_caught = 0
        self.fruits_missed = 0
        
        self.current_fruit_speed = self.BASE_FRUIT_SPEED
        self.current_spawn_rate = self.BASE_SPAWN_RATE
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        # --- Continuous reward for positioning ---
        old_basket_center_x = self.basket.centerx
        lowest_fruit = self._get_lowest_fruit()
        
        # --- Update basket position ---
        if movement == 3:  # Left
            self.basket.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket.x += self.BASKET_SPEED
        
        self.basket.x = max(0, min(self.SCREEN_WIDTH - self.BASKET_WIDTH, self.basket.x))

        if lowest_fruit:
            old_dist = abs(lowest_fruit.pos.x - old_basket_center_x)
            new_dist = abs(lowest_fruit.pos.x - self.basket.centerx)
            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.1

        # --- Update fruits ---
        self._spawn_fruit()
        
        for fruit in self.fruits[:]:
            fruit.update()
            
            # Check for catch
            if self.basket.colliderect(fruit.get_rect()):
                reward += 1.0
                self.fruits_caught += 1
                self._create_particles(fruit.pos.x, fruit.pos.y)
                self.fruits.remove(fruit)
                # SFX: Catch sound
                
                # Increase difficulty
                self.current_spawn_rate = min(0.2, self.BASE_SPAWN_RATE + self.fruits_caught * 0.01)
                if self.fruits_caught > 0 and self.fruits_caught % 5 == 0:
                    self.current_fruit_speed = min(6.0, self.BASE_FRUIT_SPEED + (self.fruits_caught // 5) * 0.2)

            # Check for miss
            elif fruit.pos.y > self.SCREEN_HEIGHT + fruit.radius:
                reward -= 1.0
                self.fruits_missed += 1
                self.fruits.remove(fruit)
                # SFX: Miss sound

        # --- Update particles ---
        for particle in self.particles[:]:
            particle.update()
            if particle.lifespan <= 0:
                self.particles.remove(particle)

        # --- Update score and check for termination ---
        self.score += reward
        self.steps += 1
        
        if self.fruits_caught >= self.WIN_CONDITION:
            reward += 10.0
            self.score += 10.0
            terminated = True
            self.game_over = True
        elif self.fruits_missed >= self.LOSE_CONDITION:
            reward -= 10.0
            self.score -= 10.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_fruit(self):
        if self.np_random.random() < self.current_spawn_rate:
            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            fruit_type = self.np_random.integers(0, len(self.FRUIT_COLORS))
            self.fruits.append(Fruit(x_pos, -20, self.current_fruit_speed, fruit_type))
            
    def _create_particles(self, x, y):
        for _ in range(20):
            self.particles.append(Particle(x, y, self.COLOR_PARTICLE))

    def _get_lowest_fruit(self):
        if not self.fruits:
            return None
        return max(self.fruits, key=lambda f: f.pos.y)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw fruits
        for fruit in self.fruits:
            color = self.FRUIT_COLORS[fruit.type]
            pos = (int(fruit.pos.x), int(fruit.pos.y))
            # Fruit body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit.radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit.radius, color)
            # Fruit stem
            pygame.draw.line(self.screen, self.FRUIT_STEM_COLOR, (pos[0], pos[1] - fruit.radius), (pos[0] + 2, pos[1] - fruit.radius - 5), 3)

        # Draw basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_OUTLINE, self.basket, width=3, border_radius=5)

        # Draw particles
        for p in self.particles:
            if p.radius > 0:
                pos = (int(p.pos.x), int(p.pos.y))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p.radius), p.color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p.radius), p.color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Caught fruits
        self._draw_apple_icon(30, 50, (0, 200, 0), True)
        caught_text = self.font_main.render(f"{self.fruits_caught} / {self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(caught_text, (55, 40))

        # Missed fruits
        self._draw_apple_icon(self.SCREEN_WIDTH - 110, 50, (200, 0, 0), False)
        missed_text = self.font_main.render(f"{self.fruits_missed} / {self.LOSE_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(missed_text, (self.SCREEN_WIDTH - 85, 40))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.fruits_caught >= self.WIN_CONDITION:
                end_text_str = "YOU WIN!"
                color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_main.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _draw_apple_icon(self, x, y, color, is_caught):
        radius = 15
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.draw.line(self.screen, self.FRUIT_STEM_COLOR, (x, y - radius), (x + 2, y - radius - 6), 3)

        if is_caught: # Draw checkmark
            pygame.draw.lines(self.screen, (255, 255, 255), False, [(x-8, y), (x, y+8), (x+8, y-8)], 3)
        else: # Draw cross
            pygame.draw.line(self.screen, (255, 255, 255), (x-8, y-8), (x+8, y+8), 3)
            pygame.draw.line(self.screen, (255, 255, 255), (x-8, y+8), (x+8, y-8), 3)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set the environment variable to run Pygame in a headless mode
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = GameEnv()
    
    # --- To run with keyboard for human testing ---
    # Note: This requires a display. Comment out the os.environ line above.
    # To install play wrapper: pip install gymnasium[classic-control]
    # from gymnasium.utils.play import play
    # play(GameEnv(), keys_to_action={
    #     "a": np.array([3, 0, 0]),
    #     "d": np.array([4, 0, 0]),
    #     pygame.K_LEFT: np.array([3, 0, 0]),
    #     pygame.K_RIGHT: np.array([4, 0, 0]),
    # }, noop=np.array([0, 0, 0]), fps=30)

    # --- To run a simple random-agent loop (headless) ---
    print("Running a sample episode with a random agent...")
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    step_count = 0
    
    while not terminated:
        action = env.action_space.sample() # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

    print(f"Episode finished after {step_count} steps. Final Score: {info['score']:.2f}")
    env.close()