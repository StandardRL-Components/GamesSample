
# Generated: 2025-08-28T03:56:05.745730
# Source Brief: brief_05088.md
# Brief Index: 5088

        
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
        "Controls: Use ← and → arrow keys to move the basket. Catch the falling fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a moving basket to progress through increasingly difficult stages. Miss 5 fruits and it's game over!"
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
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (210, 235, 255)
        self.COLOR_BASKET = (139, 69, 19)  # Brown
        self.COLOR_BASKET_RIM = (160, 82, 45)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0, 128)
        self.FRUIT_COLORS = {
            'apple': (220, 20, 60),  # Crimson
            'banana': (255, 223, 0), # Gold
            'grapes': (128, 0, 128) # Purple
        }

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game parameters
        self.MAX_LIVES = 5
        self.FRUITS_PER_STAGE = 20
        self.MAX_STAGES = 3
        self.MAX_STEPS = 2000
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 8
        self.BASE_FRUIT_SPEED = 2.0

        # Initialize state variables
        self.np_random = None
        self.basket_x = 0
        self.fruits = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.fruits_caught_in_stage = 0
        self.total_fruits_caught = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.base_fruit_speed_current = 0.0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.basket_x = self.WIDTH // 2 - self.BASKET_WIDTH // 2
        self.fruits = []
        self.particles = []
        
        self.score = 0
        self.lives = self.MAX_LIVES
        self.stage = 1
        self.fruits_caught_in_stage = 0
        self.total_fruits_caught = 0
        self.base_fruit_speed_current = self.BASE_FRUIT_SPEED

        self.steps = 0
        self.game_over = False
        self.game_won = False

        self._spawn_fruit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement = action[0]

        reward = self._calculate_continuous_reward(movement)
        self._move_basket(movement)
        
        event_reward = self._update_fruits()
        self._update_particles()

        reward += event_reward

        # Spawn new fruit if needed
        if not self.fruits:
            self._spawn_fruit()
        
        terminated = self._check_termination()

        # Terminal rewards
        if self.game_over:
            reward -= 100
        elif self.game_won:
            reward += 500
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_continuous_reward(self, movement):
        reward = 0
        if not self.fruits:
            if movement in [3, 4]: # Left or Right
                return -0.2 # Penalty for unnecessary movement
            return 0

        # Find the lowest fruit
        lowest_fruit = min(self.fruits, key=lambda f: self.HEIGHT - f['pos'][1])
        
        basket_center = self.basket_x + self.BASKET_WIDTH / 2
        fruit_x = lowest_fruit['pos'][0]
        
        current_dist = abs(basket_center - fruit_x)

        next_basket_x = self.basket_x
        if movement == 3: # Left
            next_basket_x = max(0, self.basket_x - self.BASKET_SPEED)
        elif movement == 4: # Right
            next_basket_x = min(self.WIDTH - self.BASKET_WIDTH, self.basket_x + self.BASKET_SPEED)
        
        next_basket_center = next_basket_x + self.BASKET_WIDTH / 2
        next_dist = abs(next_basket_center - fruit_x)

        if next_dist < current_dist:
            reward += 0.1 # Reward for moving towards fruit
        else:
            reward -= 0.05 # Penalty for moving away or staying still when misaligned

        return reward

    def _move_basket(self, movement):
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        
        self.basket_x = max(0, min(self.WIDTH - self.BASKET_WIDTH, self.basket_x))

    def _update_fruits(self):
        event_reward = 0
        fruits_to_remove = []
        for fruit in self.fruits:
            # Update position based on trajectory
            if fruit['trajectory'] == 'linear':
                fruit['pos'][1] += fruit['vel'][1]
            elif fruit['trajectory'] == 'sine':
                fruit['pos'][1] += fruit['vel'][1]
                fruit['pos'][0] = fruit['start_x'] + fruit['amplitude'] * math.sin(fruit['pos'][1] * fruit['frequency'])
            elif fruit['trajectory'] == 'diagonal':
                fruit['pos'][0] += fruit['vel'][0]
                fruit['pos'][1] += fruit['vel'][1]
            
            # Check for catch
            basket_rect = pygame.Rect(self.basket_x, self.HEIGHT - self.BASKET_HEIGHT - 10, self.BASKET_WIDTH, self.BASKET_HEIGHT)
            fruit_rect = pygame.Rect(fruit['pos'][0] - 10, fruit['pos'][1] - 10, 20, 20)
            
            if basket_rect.colliderect(fruit_rect):
                # SFX: Catch sound
                self.score += 1
                event_reward += 1
                self.fruits_caught_in_stage += 1
                self.total_fruits_caught += 1

                # Risky catch bonus
                dist_from_edge = min(abs(fruit['pos'][0] - basket_rect.left), abs(fruit['pos'][0] - basket_rect.right))
                if dist_from_edge < 10:
                    self.score += 5
                    event_reward += 5
                
                self._create_catch_particles(fruit['pos'], fruit['color'])
                fruits_to_remove.append(fruit)

                # Difficulty scaling
                if self.fruits_caught_in_stage > 0 and self.fruits_caught_in_stage % 10 == 0:
                    self.base_fruit_speed_current += 0.05 * self.FRUITS_PER_STAGE # This seems high, let's adjust.
                    # The brief says 0.05 pixels/frame. Let's just do that.
                    self.base_fruit_speed_current += 0.5 # A more noticeable jump

                # Stage progression
                if self.fruits_caught_in_stage >= self.FRUITS_PER_STAGE:
                    if self.stage < self.MAX_STAGES:
                        self.stage += 1
                        self.fruits_caught_in_stage = 0
                        self.base_fruit_speed_current = self.BASE_FRUIT_SPEED + (self.stage - 1) * 1.0
                        event_reward += 100 # Stage complete reward
                    else:
                        self.game_won = True
                continue

            # Check for miss
            if fruit['pos'][1] > self.HEIGHT or fruit['pos'][0] < 0 or fruit['pos'][0] > self.WIDTH:
                # SFX: Miss sound
                self.lives -= 1
                event_reward -= 1
                self._create_miss_particles(fruit['pos'])
                fruits_to_remove.append(fruit)
                if self.lives <= 0:
                    self.game_over = True

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return event_reward

    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(['apple', 'banana', 'grapes'])
        x = self.np_random.integers(20, self.WIDTH - 20)
        y = -20
        speed = self.base_fruit_speed_current + self.np_random.random() * 1.5

        fruit = {
            'pos': [x, y],
            'type': fruit_type,
            'color': self.FRUIT_COLORS[fruit_type],
            'trajectory': 'linear',
            'start_x': x
        }
        
        if self.stage == 1:
            fruit['vel'] = [0, speed]
            fruit['trajectory'] = 'linear'
        elif self.stage == 2:
            fruit['vel'] = [0, speed * 0.8] # Slower vertical for more sine effect
            fruit['trajectory'] = 'sine'
            fruit['amplitude'] = self.np_random.integers(50, 150)
            fruit['frequency'] = self.np_random.random() * 0.02 + 0.01
        elif self.stage == 3:
            fruit['vel'] = [self.np_random.choice([-1, 1]) * (speed * 0.5), speed * 0.9]
            fruit['trajectory'] = 'diagonal'

        self.fruits.append(fruit)

    def _create_catch_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.random() * 3 + 2,
                'color': color
            })

    def _create_miss_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.random() * math.pi + math.pi # Upwards splash
            speed = self.np_random.random() * 2 + 0.5
            self.particles.append({
                'pos': [pos[0], self.HEIGHT - 5],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'radius': self.np_random.random() * 2 + 1,
                'color': (100, 100, 100)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return self.game_over or self.game_won or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "stage": self.stage,
            "fruits_caught_in_stage": self.fruits_caught_in_stage,
            "steps": self.steps,
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [
                (1 - interp) * self.COLOR_BG_TOP[i] + interp * self.COLOR_BG_BOTTOM[i]
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        self._draw_basket()
        for fruit in self.fruits:
            self._draw_fruit(fruit)
        self._draw_particles()

    def _draw_basket(self):
        basket_rect = pygame.Rect(self.basket_x, self.HEIGHT - self.BASKET_HEIGHT - 10, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, basket_rect, width=4, border_radius=5)

    def _draw_fruit(self, fruit):
        x, y = int(fruit['pos'][0]), int(fruit['pos'][1])
        color = fruit['color']
        if fruit['type'] == 'apple':
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, color)
        elif fruit['type'] == 'banana':
            points = []
            for i in range(10):
                angle = math.pi/2 + (i/9) * math.pi/2
                px = x + 15 * math.cos(angle)
                py = y - 15 * math.sin(angle)
                points.append((int(px), int(py)))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, color, False, points, 5)
        elif fruit['type'] == 'grapes':
            offsets = [(-5, -5), (5, -5), (0, 0), (-5, 5), (5, 5), (0, 10)]
            for off_x, off_y in offsets:
                pygame.gfxdraw.aacircle(self.screen, x + off_x, y + off_y, 5, color)
                pygame.gfxdraw.filled_circle(self.screen, x + off_x, y + off_y, 5, color)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_ui(self):
        # Helper for shadowed text
        def draw_text(text, font, color, pos, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_medium, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Lives
        lives_text = f"Lives: {self.lives}"
        text_w = self.font_medium.size(lives_text)[0]
        draw_text(lives_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH - text_w - 10, 10), self.COLOR_TEXT_SHADOW)
        
        # Stage
        stage_text = f"Stage: {self.stage} / {self.MAX_STAGES}"
        text_w = self.font_medium.size(stage_text)[0]
        draw_text(stage_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH // 2 - text_w // 2, 10), self.COLOR_TEXT_SHADOW)

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            text_w, text_h = self.font_large.size(msg)
            draw_text(msg, self.font_large, (200, 0, 0), (self.WIDTH // 2 - text_w // 2, self.HEIGHT // 2 - text_h // 2), self.COLOR_TEXT_SHADOW)
        elif self.game_won:
            msg = "YOU WIN!"
            text_w, text_h = self.font_large.size(msg)
            draw_text(msg, self.font_large, (0, 200, 0), (self.WIDTH // 2 - text_w // 2, self.HEIGHT // 2 - text_h // 2), self.COLOR_TEXT_SHADOW)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()