
# Generated: 2025-08-28T01:14:44.298902
# Source Brief: brief_04052.md
# Brief Index: 4052

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the basket. Catch the falling fruit!"
    )

    # Short, user-facing description of the game:
    game_description = (
        "Catch falling fruit to score points. Reach 1000 points to win, but don't miss more than 5 fruit!"
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TARGET_SCORE = 1000
        self.MAX_MISSES = 5
        self.MAX_STEPS = 10000

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
        self.font = pygame.font.Font(None, 36)
        
        # Color definitions
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (200, 235, 255) # Lighter Sky Blue
        self.COLOR_BASKET = (139, 69, 19) # Saddle Brown
        self.COLOR_BASKET_OUTLINE = (85, 43, 11) # Darker Brown
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_HEART = (255, 20, 20) # Bright Red

        self.FRUIT_TYPES = {
            "red": {"color": (255, 50, 50), "value": 10, "radius": 10},
            "green": {"color": (50, 205, 50), "value": 20, "radius": 12},
            "blue": {"color": (60, 130, 255), "value": 30, "radius": 14},
        }
        
        # Initialize state variables
        self.basket_pos_x = 0
        self.basket_width = 80
        self.basket_height = 20
        self.basket_speed = 10
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.fruit_fall_speed = 0
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 60 # frames between spawns

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_pos_x = self.SCREEN_WIDTH / 2
        self.fruits = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_MISSES
        self.game_over = False
        
        self.fruit_fall_speed = 2.0
        self.fruit_spawn_timer = self.fruit_spawn_rate

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right
        
        self._move_basket(movement)
        
        step_reward = self._update_game_logic()
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        # Terminal rewards
        reward = step_reward
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_basket(self, movement):
        if movement == 3:  # Left
            self.basket_pos_x -= self.basket_speed
        elif movement == 4:  # Right
            self.basket_pos_x += self.basket_speed
        
        # Clamp basket position to be within screen bounds
        self.basket_pos_x = max(
            self.basket_width / 2, 
            min(self.basket_pos_x, self.SCREEN_WIDTH - self.basket_width / 2)
        )

    def _update_game_logic(self):
        reward = 0
        
        # Update difficulty based on score
        self.fruit_fall_speed = 2.0 + (self.score // 100) * 0.2

        # Spawn new fruits
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_rate = max(15, 60 - (self.score // 50)) # Faster spawns at higher scores
            self.fruit_spawn_timer = self.fruit_spawn_rate

        # Update fruits
        for fruit in self.fruits[:]:
            fruit['pos'].y += self.fruit_fall_speed
            
            # Check for catch
            basket_rect = pygame.Rect(
                self.basket_pos_x - self.basket_width / 2, 
                self.SCREEN_HEIGHT - self.basket_height - 5,
                self.basket_width,
                self.basket_height
            )
            if basket_rect.collidepoint(fruit['pos'].x, fruit['pos'].y):
                self.score += fruit['value']
                reward += 1
                if fruit['type'] == 'blue':
                    reward += 10 # Special bonus for high-value fruit
                self._spawn_particles(fruit['pos'], fruit['color'], 20)
                self.fruits.remove(fruit)
                # Placeholder: pygame.mixer.Sound("catch.wav").play()
                continue

            # Check for miss
            if fruit['pos'].y > self.SCREEN_HEIGHT:
                self.lives -= 1
                reward -= 1
                self._spawn_particles(pygame.Vector2(fruit['pos'].x, self.SCREEN_HEIGHT - 5), (128, 128, 128), 10)
                self.fruits.remove(fruit)
                # Placeholder: pygame.mixer.Sound("miss.wav").play()

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                
        return reward

    def _spawn_fruit(self):
        fruit_type_name = random.choice(list(self.FRUIT_TYPES.keys()))
        fruit_type_props = self.FRUIT_TYPES[fruit_type_name]
        
        self.fruits.append({
            "pos": pygame.Vector2(random.uniform(20, self.SCREEN_WIDTH - 20), -fruit_type_props['radius']),
            "type": fruit_type_name,
            **fruit_type_props
        })

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "color": color,
                "lifespan": random.randint(15, 30)
            })

    def _check_termination(self):
        return self.score >= self.TARGET_SCORE or self.lives <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

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
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), 2, color_with_alpha
            )
            
        # Draw fruits
        for fruit in self.fruits:
            x, y = int(fruit['pos'].x), int(fruit['pos'].y)
            radius = fruit['radius']
            color = fruit['color']
            
            # Draw a simple shine effect
            shine_color = (255, 255, 255, 100)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, x - radius // 3, y - radius // 3, radius // 3, shine_color)

        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.basket_width / 2, 
            self.SCREEN_HEIGHT - self.basket_height - 5,
            self.basket_width,
            self.basket_height
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_OUTLINE, basket_rect, width=2, border_radius=5)

    def _render_ui(self):
        # Draw Score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, (15, 10))

        # Draw Lives (Hearts)
        for i in range(self.lives):
            self._draw_heart(
                self.screen, 
                self.SCREEN_WIDTH - 30 - (i * 35), 
                25, 
                15, 
                self.COLOR_HEART
            )
            
        if self.game_over:
            win_lose_text = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            text_surf = self.font.render(win_lose_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            shadow_surf = self.font.render(win_lose_text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 2, self.SCREEN_HEIGHT/2 - 18))
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos):
        text_surf = self.font.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _draw_heart(self, surface, x, y, size, color):
        # Simple heart shape using two circles and a polygon
        points = [
            (x, y + size * 0.25),
            (x - size / 2, y - size * 0.25),
            (x + size / 2, y - size * 0.25)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.gfxdraw.filled_circle(surface, int(x - size / 4), int(y - size * 0.25), int(size * 0.35), color)
        pygame.gfxdraw.filled_circle(surface, int(x + size / 4), int(y - size * 0.25), int(size * 0.35), color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Test a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Game ended. Final info: {info}")
            obs, info = env.reset()
    
    env.close()
    print("Environment test run completed.")