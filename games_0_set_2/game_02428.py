import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Fruit Catcher: A top-down arcade game where the player controls a basket
    to catch falling fruit. The goal is to achieve a high score by catching
    fruit, with bonuses for risky catches near the ground. The game ends
    if the player catches 100 fruits (win) or misses 10 (loss).
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to move the basket left and right to catch the falling fruit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in your basket! Get bonus points for risky catches near the ground. Miss 10 fruits and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 360

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GROUND = (60, 140, 70)
    COLOR_BASKET = (139, 69, 19)
    COLOR_APPLE = (220, 40, 40)
    COLOR_BANANA = (255, 225, 53)
    COLOR_ORANGE = (255, 165, 0)
    COLOR_STEM = (0, 128, 0)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_SPLAT = (100, 80, 70)

    # Game parameters
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 10.0
    FRUIT_SIZE = 12
    INITIAL_SPAWN_INTERVAL = 90  # frames (3 seconds at 30fps)
    MAX_STEPS = 1500 # Increased from 1000 to allow more time for 100 catches
    WIN_CONDITION_CATCHES = 100
    LOSE_CONDITION_MISSES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables are initialized in reset()
        self.basket_pos = None
        self.fruits = None
        self.particles = None
        self.steps = None
        self.score = None
        self.fruits_caught = None
        self.fruits_missed = None
        self.fruit_spawn_timer = None
        self.current_spawn_interval = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.basket_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GROUND_Y - self.BASKET_HEIGHT / 2)
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.current_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.fruit_spawn_timer = self.current_spawn_interval

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        reward = 0
        terminated = False
        truncated = False

        # --- 1. Handle Player Input ---
        closest_fruit = self._get_closest_fruit()
        old_dist = abs(self.basket_pos.x - closest_fruit['pos'].x) if closest_fruit else float('inf')

        if movement == 3:  # Left
            self.basket_pos.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos.x += self.BASKET_SPEED

        # Clamp basket position to screen bounds
        self.basket_pos.x = max(self.BASKET_WIDTH / 2, min(self.basket_pos.x, self.SCREEN_WIDTH - self.BASKET_WIDTH / 2))

        # Continuous reward for moving towards/away from the closest fruit
        if closest_fruit:
            new_dist = abs(self.basket_pos.x - closest_fruit['pos'].x)
            if new_dist < old_dist:
                reward += 1.0  # Moving towards fruit
            else:
                reward -= 0.1 # Moving away or staying still
        
        # --- 2. Update Game Logic ---
        self._update_fruits()
        self._update_particles()
        reward += self._handle_collisions()
        self._spawn_fruit()

        # --- 3. Calculate Reward & Check Termination ---
        self.steps += 1
        
        if self.fruits_caught >= self.WIN_CONDITION_CATCHES:
            reward += 100
            terminated = True
        elif self.fruits_missed >= self.LOSE_CONDITION_MISSES:
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_closest_fruit(self):
        if not self.fruits:
            return None
        return min(self.fruits, key=lambda f: abs(self.basket_pos.x - f['pos'].x))

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            fruit_type = self.np_random.choice(['apple', 'banana', 'orange'])
            start_x = self.np_random.uniform(self.FRUIT_SIZE, self.SCREEN_WIDTH - self.FRUIT_SIZE)
            self.fruits.append({
                'pos': pygame.Vector2(start_x, -self.FRUIT_SIZE),
                'type': fruit_type,
                'speed': self.np_random.uniform(2.0, 3.0),
                'rotation': self.np_random.uniform(0, 360)
            })
            self.fruit_spawn_timer = self.current_spawn_interval

    def _update_fruits(self):
        for fruit in self.fruits:
            fruit['speed'] += 0.02  # Acceleration
            fruit['pos'].y += fruit['speed']
            fruit['rotation'] = (fruit['rotation'] + fruit['speed'] / 2) % 360

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        basket_rect = pygame.Rect(
            self.basket_pos.x - self.BASKET_WIDTH / 2,
            self.basket_pos.y - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH, self.BASKET_HEIGHT
        )
        
        fruits_to_remove = []
        for fruit in self.fruits:
            # Collision with basket
            if basket_rect.collidepoint(fruit['pos']):
                self.fruits_caught += 1
                
                # Calculate reward based on catch height
                catch_height_ratio = fruit['pos'].y / self.GROUND_Y
                if catch_height_ratio > 0.8: # Risky catch
                    reward += 15 # +10 base, +5 bonus
                    self._create_particles(fruit['pos'], self.COLOR_PARTICLE, 20, is_bonus=True)
                elif catch_height_ratio < 0.2: # Safe catch
                    reward += 8 # +10 base, -2 penalty
                    self._create_particles(fruit['pos'], self.COLOR_PARTICLE, 10)
                else: # Normal catch
                    reward += 10
                    self._create_particles(fruit['pos'], self.COLOR_PARTICLE, 10)

                fruits_to_remove.append(fruit)
                
                # Increase difficulty every 5 catches
                if self.fruits_caught % 5 == 0 and self.current_spawn_interval > 20:
                    self.current_spawn_interval = max(20, self.current_spawn_interval * 0.95)

            # Collision with ground (miss)
            elif fruit['pos'].y > self.GROUND_Y:
                self.fruits_missed += 1
                reward -= 10
                self._create_particles(fruit['pos'], self.COLOR_SPLAT, 15, is_splat=True)
                fruits_to_remove.append(fruit)

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return reward

    def _create_particles(self, pos, color, count, is_bonus=False, is_splat=False):
        for _ in range(count):
            if is_splat:
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4) if is_bonus else self.np_random.uniform(0.5, 2.5)

            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4) if is_bonus else self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            # Create a temporary surface for alpha blending if gfxdraw doesn't support it directly
            # This is a common workaround for older pygame/gfxdraw versions
            target_rect = pygame.Rect(p['pos'].x - p['radius'], p['pos'].y - p['radius'], p['radius']*2, p['radius']*2)
            temp_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(p['radius']), int(p['radius']), int(p['radius']), p['color'] + (alpha,))
            self.screen.blit(temp_surf, target_rect.topleft)


        # Draw fruits
        for fruit in self.fruits:
            x, y = int(fruit['pos'].x), int(fruit['pos'].y)
            if fruit['type'] == 'apple':
                pygame.gfxdraw.aacircle(self.screen, x, y, self.FRUIT_SIZE, self.COLOR_APPLE)
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.FRUIT_SIZE, self.COLOR_APPLE)
                pygame.draw.line(self.screen, self.COLOR_STEM, (x, y - self.FRUIT_SIZE), (x, y - self.FRUIT_SIZE - 4), 2)
            elif fruit['type'] == 'orange':
                pygame.gfxdraw.aacircle(self.screen, x, y, self.FRUIT_SIZE, self.COLOR_ORANGE)
                pygame.gfxdraw.filled_circle(self.screen, x, y, self.FRUIT_SIZE, self.COLOR_ORANGE)
                pygame.gfxdraw.filled_circle(self.screen, x+1, y - self.FRUIT_SIZE, 2, self.COLOR_STEM)
            elif fruit['type'] == 'banana':
                points = []
                for i in range(11):
                    angle = math.pi * (i / 10.0) + math.radians(fruit['rotation'])
                    px = x + math.cos(angle) * self.FRUIT_SIZE * 1.5
                    py = y + math.sin(angle) * self.FRUIT_SIZE * 0.75
                    points.append((px, py))
                if len(points) > 1:
                    pygame.draw.aalines(self.screen, self.COLOR_BANANA, False, points, 5)


        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_pos.x - self.BASKET_WIDTH / 2,
            self.basket_pos.y - self.BASKET_HEIGHT / 2,
            self.BASKET_WIDTH, self.BASKET_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_large.render(f"Misses: {self.fruits_missed}/{self.LOSE_CONDITION_MISSES}", True, self.COLOR_UI_TEXT)
        text_rect = miss_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(miss_text, text_rect)
    
    def _get_info(self):
        # Update score before returning info
        self.score = self.fruits_caught * 10 # Simple score metric for info dict
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so we will unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for manual testing
    manual_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the manual display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        # Control the frame rate for manual play
        env.clock.tick(30)
        
    env.close()