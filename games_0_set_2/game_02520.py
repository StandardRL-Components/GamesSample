# Generated: 2025-08-27T20:36:11.231314
# Source Brief: brief_02520.md
# Brief Index: 2520

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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the basket. Catch the fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you catch falling fruit. "
        "Catch 50 to win, but miss 3 and it's game over!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_BASKET = (139, 69, 19) # Saddle Brown
        self.COLOR_BASKET_RIM = (160, 82, 45) # Sienna
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_OUTLINE = (50, 50, 50)
        self.FRUIT_COLORS = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
        ]
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            
        # Game constants
        self.BASKET_WIDTH = 100
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 10
        self.FRUIT_RADIUS = 12
        self.MAX_FRUITS_ON_SCREEN = 7
        self.FRUIT_SPAWN_CHANCE = 0.03
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 3
        self.MAX_STEPS = 1000

        # Game state variables
        self.basket_pos = None
        self.fruits = []
        self.particles = []
        self.score = 0
        self.misses = 0
        self.fruits_caught = 0
        self.base_fruit_speed = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        # Initialize state variables
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.basket_pos = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) // 2, 
            self.HEIGHT - self.BASKET_HEIGHT - 10, 
            self.BASKET_WIDTH, 
            self.BASKET_HEIGHT
        )
        self.fruits = []
        self.particles = []
        self.score = 0
        self.misses = 0
        self.fruits_caught = 0
        self.base_fruit_speed = 2.0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        # Spawn initial fruits
        for _ in range(3):
            self._spawn_fruit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.2  # Continuous penalty for time passing

        # --- Handle Input ---
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos.x += self.BASKET_SPEED
        
        # Clamp basket to screen
        self.basket_pos.x = max(0, min(self.WIDTH - self.BASKET_WIDTH, self.basket_pos.x))

        # --- Update Game State ---
        self._update_particles()
        catch_events, miss_events = self._update_fruits()
        
        # Calculate rewards from events
        for event in catch_events:
            reward += 5.0  # Base reward for catch
            self.score += 5
            self.fruits_caught += 1

            # Risk/reward for edge catches
            catch_pos_rel = event['pos_x'] - self.basket_pos.centerx
            edge_threshold = self.BASKET_WIDTH * 0.35
            if abs(catch_pos_rel) > edge_threshold:
                reward += 2.0
                self.score += 2
            else:
                reward += 1.0
                self.score += 1
            
            # Create particles
            self._create_catch_particles(event['pos_x'], event['pos_y'], event['color'])
            
            # Increase difficulty
            if self.fruits_caught > 0 and self.fruits_caught % 10 == 0:
                self.base_fruit_speed += 0.05
        
        for _ in miss_events:
            reward -= 1.0
            self.misses += 1

        # --- Spawn new fruit ---
        if len(self.fruits) < self.MAX_FRUITS_ON_SCREEN and self.np_random.random() < self.FRUIT_SPAWN_CHANCE:
            self._spawn_fruit()

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.misses >= self.LOSS_CONDITION:
            reward -= 50.0
            self.game_over = True
            terminated = True
        elif self.fruits_caught >= self.WIN_CONDITION:
            reward += 50.0
            self.game_over = True
            self.game_won = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard: terminated is True when truncated is True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _spawn_fruit(self):
        x_pos = self.np_random.integers(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS)
        fruit = {
            'pos': [x_pos, -self.FRUIT_RADIUS],
            'color': self.FRUIT_COLORS[self.np_random.integers(0, len(self.FRUIT_COLORS))],
            'speed': self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5),
            'miss_flash': 0
        }
        self.fruits.append(fruit)

    def _update_fruits(self):
        catch_events = []
        miss_events = []
        
        for fruit in reversed(self.fruits):
            fruit['pos'][1] += fruit['speed']
            
            fruit_rect = pygame.Rect(fruit['pos'][0] - self.FRUIT_RADIUS, fruit['pos'][1] - self.FRUIT_RADIUS, self.FRUIT_RADIUS * 2, self.FRUIT_RADIUS * 2)

            # Check for catch
            if not self.game_over and self.basket_pos.colliderect(fruit_rect):
                # sfx: catch_sound.play()
                catch_events.append({
                    'pos_x': fruit['pos'][0],
                    'pos_y': fruit['pos'][1],
                    'color': fruit['color']
                })
                self.fruits.remove(fruit)
                continue

            # Check for miss
            if fruit['pos'][1] > self.HEIGHT + self.FRUIT_RADIUS:
                if not self.game_over:
                    # sfx: miss_sound.play()
                    miss_events.append({})
                self.fruits.remove(fruit)
        
        return catch_events, miss_events

    def _create_catch_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': [x, y],
                'vel': velocity,
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40))))
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Render fruits
        for fruit in self.fruits:
            pygame.gfxdraw.aacircle(self.screen, int(fruit['pos'][0]), int(fruit['pos'][1]), self.FRUIT_RADIUS, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(fruit['pos'][0]), int(fruit['pos'][1]), self.FRUIT_RADIUS, fruit['color'])
            
        # Render basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_pos, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, self.basket_pos, width=3, border_radius=5)

    def _draw_text(self, text, font, color, pos, outline_color=None, outline_width=2):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        
        if outline_color:
            outline_surface = font.render(text, True, outline_color)
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        outline_rect = outline_surface.get_rect(center=(pos[0] + dx, pos[1] + dy))
                        self.screen.blit(outline_surface, outline_rect)

        self.screen.blit(text_surface, text_rect)

    def _render_ui(self):
        # Render Score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, self.font_medium, self.COLOR_TEXT, (80, 25), self.COLOR_TEXT_OUTLINE)
        
        # Render Misses
        miss_text = f"Misses: {self.misses}/{self.LOSS_CONDITION}"
        text_surf = self.font_medium.render(miss_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 20, 10))
        
        outline_surf = self.font_medium.render(miss_text, True, self.COLOR_TEXT_OUTLINE)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:
                    self.screen.blit(outline_surf, (text_rect.x + dx, text_rect.y + dy))
        self.screen.blit(text_surf, text_rect)
        
        # Render Game Over/Win message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 200, 0) if self.game_won else (200, 0, 0)
            self._draw_text(message, self.font_large, color, (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TEXT_OUTLINE, 3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "fruits_caught": self.fruits_caught
        }
    
    def close(self):
        pygame.quit()

# Example usage (for testing)
if __name__ == '__main__':
    # Set this to 'human' to play the game, or 'rgb_array' for headless testing
    render_mode = "rgb_array" # Default to headless for automated testing

    if render_mode == "human":
        # In human mode, we need a real display.
        # Unset the dummy video driver if it was set
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Override the screen to be a display surface
        pygame.display.set_caption(env.game_description)
        env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = [0, 0, 0] # Default action: no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display screen
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            env.screen.blit(draw_surface, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Control human play speed

        env.close()
    else:
        # Standard Gym loop for training/testing
        print("Running headless test...")
        env = GameEnv(render_mode="rgb_array")
        
        # Validate implementation
        try:
            test_obs = env.observation_space.sample()
            assert test_obs.shape == (env.HEIGHT, env.WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            obs, info = env.reset()
            assert obs.shape == (env.HEIGHT, env.WIDTH, 3)
            assert isinstance(info, dict)
            
            test_action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(test_action)
            assert obs.shape == (env.HEIGHT, env.WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
            print("✓ Implementation validated successfully")
        except Exception as e:
            print(f"✗ Implementation validation failed: {e}")
            env.close()
            exit(1)

        obs, info = env.reset()
        total_reward = 0
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward}")
                obs, info = env.reset()
                total_reward = 0
        env.close()
        print("Headless test finished.")