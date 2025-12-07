
# Generated: 2025-08-28T04:34:59.954583
# Source Brief: brief_05294.md
# Brief Index: 5294

        
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
    """
    Fruit Catcher: A fast-paced arcade game where the player controls a basket
    to catch falling fruit. The goal is to collect 100 fruits before missing 5.
    Different fruits provide different scores, and the game speed increases
    as more fruits are collected.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the basket left and right. "
        "Catch the falling fruit to score points."
    )

    game_description = (
        "Catch falling fruit to score points! Green fruit is worth +1, red is +3, "
        "but brown fruit will cost you -1. Reach 100 collected fruits to win, "
        "but miss 5 and it's game over."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.COLOR_BG_TOP = (10, 10, 40)
        self.COLOR_BG_BOTTOM = (40, 10, 60)
        self.COLOR_BASKET_BODY = (255, 220, 0)
        self.COLOR_BASKET_RIM = (255, 240, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_MISS_TEXT = (255, 100, 100)
        self.FRUIT_COLORS = {
            'green': (50, 200, 50),
            'red': (220, 50, 50),
            'brown': (140, 70, 20)
        }
        
        # --- Game Mechanics ---
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_Y_POS = self.HEIGHT - self.BASKET_HEIGHT - 10
        self.BASKET_SPEED = 12
        self.FRUIT_RADIUS = 12
        self.BASE_FRUIT_SPEED = 2.5
        self.SPEED_INCREMENT = 1.0
        self.FRUIT_SPAWN_PROB = 0.04
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 100
        self.LOSE_CONDITION = 5
        
        # --- State Variables ---
        self.basket_x = 0
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.collected_fruits = 0
        self.missed_fruits = 0
        self.fruit_speed = 0.0
        
        # Initialize state and validate
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.collected_fruits = 0
        self.missed_fruits = 0
        self.fruit_speed = self.BASE_FRUIT_SPEED
        self.basket_x = self.WIDTH // 2 - self.BASKET_WIDTH // 2
        self.fruits = []
        self.particles = []
        
        # Spawn initial fruits to fill the screen a bit
        for _ in range(5):
            self._spawn_fruit(initial_spawn=True)
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # 1. Handle player input
        movement = action[0]
        self._handle_movement(movement)
        
        # 2. Update fruits (move, check for catch/miss)
        catch_reward, miss_penalty = self._update_fruits()
        reward += catch_reward + miss_penalty

        # 3. Update particles
        self._update_particles()
        
        # 4. Spawn new fruits
        if self.np_random.random() < self.FRUIT_SPAWN_PROB:
            self._spawn_fruit()

        # 5. Update game progression
        self.steps += 1
        self.fruit_speed = self.BASE_FRUIT_SPEED + (self.collected_fruits // 50) * self.SPEED_INCREMENT
        
        # 6. Check for termination
        terminated = False
        if self.collected_fruits >= self.WIN_CONDITION:
            terminated = True
            reward += 100.0
        elif self.missed_fruits >= self.LOSE_CONDITION:
            terminated = True
            reward -= 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        
        self.basket_x = np.clip(self.basket_x, 0, self.WIDTH - self.BASKET_WIDTH)

    def _update_fruits(self):
        catch_reward = 0.0
        miss_penalty = 0.0
        basket_rect = pygame.Rect(self.basket_x, self.BASKET_Y_POS, self.BASKET_WIDTH, self.BASKET_HEIGHT)

        for fruit in self.fruits[:]:
            fruit['pos'][1] += self.fruit_speed
            fruit_rect = pygame.Rect(fruit['pos'][0] - fruit['radius'], fruit['pos'][1] - fruit['radius'], fruit['radius']*2, fruit['radius']*2)

            if basket_rect.colliderect(fruit_rect):
                # SFX: Catch sound
                self.collected_fruits += 1
                catch_reward += 1.0  # Event-based reward
                
                if fruit['type'] == 'green':
                    self.score += 1
                    catch_reward += 0.1
                elif fruit['type'] == 'red':
                    self.score += 3
                    catch_reward += 0.3
                elif fruit['type'] == 'brown':
                    self.score -= 1
                    catch_reward -= 0.1
                
                self._create_particles(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)
                continue

            if fruit['pos'][1] > self.HEIGHT + self.FRUIT_RADIUS:
                # SFX: Miss/splash sound
                self.missed_fruits += 1
                miss_penalty -= 1.0  # Event-based reward
                self._create_particles([fruit['pos'][0], self.HEIGHT - 5], (100, 150, 255), 'splash')
                self.fruits.remove(fruit)
        
        return catch_reward, miss_penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_fruit(self, initial_spawn=False):
        fruit_type = self.np_random.choice(['green', 'red', 'brown'], p=[0.7, 0.15, 0.15])
        
        y_pos = -self.FRUIT_RADIUS
        if initial_spawn:
            y_pos = self.np_random.integers(-self.HEIGHT, -self.FRUIT_RADIUS)

        self.fruits.append({
            'pos': [self.np_random.integers(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS), y_pos],
            'type': fruit_type,
            'color': self.FRUIT_COLORS[fruit_type],
            'radius': self.FRUIT_RADIUS
        })

    def _create_particles(self, pos, color, p_type='catch'):
        num_particles = 15 if p_type == 'catch' else 20
        for _ in range(num_particles):
            if p_type == 'catch':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
                lifespan = self.np_random.integers(10, 20)
                radius = self.np_random.integers(2, 5)
            else:  # splash
                angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
                speed = self.np_random.uniform(2, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(15, 25)
                radius = self.np_random.integers(1, 4)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color, 'radius': radius})

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = fruit['radius']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, fruit['color'])
            shine_pos = (pos[0] + radius // 3, pos[1] - radius // 3)
            pygame.gfxdraw.aacircle(self.screen, shine_pos[0], shine_pos[1], radius // 4, (255, 255, 255, 150))
            pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], radius // 4, (255, 255, 255, 150))

        # Draw basket
        basket_rect = pygame.Rect(int(self.basket_x), self.BASKET_Y_POS, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_BODY, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, (basket_rect.x, basket_rect.y, basket_rect.width, 5), border_top_left_radius=5, border_top_right_radius=5)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, 255 * (p['lifespan'] / 20.0))
            color_with_alpha = (*p['color'], int(alpha))
            # GFXDraw doesn't handle alpha well in filled shapes, so we create a temporary surface
            particle_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(particle_surf, (pos[0] - p['radius'], pos[1] - p['radius']))
            
    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        missed_text = self.font_small.render(f"Missed: {self.missed_fruits} / {self.LOSE_CONDITION}", True, self.COLOR_MISS_TEXT)
        text_rect = missed_text.get_rect(topright=(self.WIDTH - 10, 15))
        self.screen.blit(missed_text, text_rect)

        collected_text = self.font_large.render(f"{self.collected_fruits} / {self.WIN_CONDITION}", True, self.COLOR_TEXT)
        text_rect = collected_text.get_rect(midbottom=(self.WIDTH // 2, self.HEIGHT - 5))
        self.screen.blit(collected_text, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_fruits": self.collected_fruits,
            "missed_fruits": self.missed_fruits,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space/Shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Maintain 30 FPS
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()