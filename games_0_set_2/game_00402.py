
# Generated: 2025-08-27T13:31:49.449971
# Source Brief: brief_00402.md
# Brief Index: 402

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the baskets left and right. Catch the fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in your baskets. Green fruit is good, red is bonus, but avoid the blue penalty fruit! Catch 50 to win, but miss 10 and you lose."
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
        
        # Game constants
        self._define_constants()
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def _define_constants(self):
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_BASKET = (139, 69, 19)
        self.COLOR_BASKET_RIM = (160, 82, 45)
        self.COLOR_FRUIT_GREEN = (50, 205, 50)
        self.COLOR_FRUIT_RED = (255, 69, 0)
        self.COLOR_FRUIT_BLUE = (30, 144, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SHADOW = (20, 20, 20)

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game parameters
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 10
        self.MAX_STEPS = 3000
        self.BASKET_SPEED = 8
        self.BASKET_WIDTH = 60
        self.BASKET_HEIGHT = 20
        self.BASKET_Y = self.HEIGHT - 40
        self.BASKET_SPACING = 120
        self.NUM_BASKETS = 3
        self.total_baskets_width = (self.NUM_BASKETS * self.BASKET_WIDTH) + ((self.NUM_BASKETS - 1) * self.BASKET_SPACING)
        self.FRUIT_RADIUS = 10
        self.FRUIT_SPAWN_RATE = 25 # steps
        self.INITIAL_FALL_SPEED = 1.5
        self.DIFFICULTY_INTERVAL = 50 # steps
        self.DIFFICULTY_INCREASE = 0.02
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.missed_fruits = 0
        self.caught_fruits = 0
        
        self.baskets_x = (self.WIDTH - self.total_baskets_width) / 2
        self.baskets = self._create_baskets()
        
        self.fruits = []
        self.particles = []
        
        self.fruit_fall_speed = self.INITIAL_FALL_SPEED
        self.fruit_spawn_timer = 0
        
        return self._get_observation(), self._get_info()

    def _create_baskets(self):
        baskets = []
        for i in range(self.NUM_BASKETS):
            x = self.baskets_x + i * (self.BASKET_WIDTH + self.BASKET_SPACING)
            baskets.append(pygame.Rect(x, self.BASKET_Y, self.BASKET_WIDTH, self.BASKET_HEIGHT))
        return baskets
        
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # --- 1. Handle Input ---
            self._handle_input(movement)

            # --- 2. Update Game Logic ---
            reward += self._update_fruits()
            self._update_particles()
            self._update_difficulty()
            
        # --- 3. Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.caught_fruits >= self.WIN_CONDITION:
                reward += 100 # Win bonus
            elif self.missed_fruits >= self.LOSS_CONDITION:
                reward -= 100 # Loss penalty
        
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.baskets_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.baskets_x += self.BASKET_SPEED

        # Clamp basket position to screen bounds
        self.baskets_x = max(0, min(self.baskets_x, self.WIDTH - self.total_baskets_width))
        
        # Update individual basket rects
        for i, basket in enumerate(self.baskets):
            basket.x = self.baskets_x + i * (self.BASKET_WIDTH + self.BASKET_SPACING)

    def _update_fruits(self):
        step_reward = 0
        
        # Spawn new fruits
        self.fruit_spawn_timer += 1
        if self.fruit_spawn_timer >= self.FRUIT_SPAWN_RATE:
            self.fruit_spawn_timer = 0
            self._spawn_fruit()

        # Move and check existing fruits
        for fruit in reversed(self.fruits):
            fruit['pos'][1] += self.fruit_fall_speed
            fruit['rect'].center = fruit['pos']

            # Check for catch
            catch_index = fruit['rect'].collidelist(self.baskets)
            if catch_index != -1:
                # --- Sound effect: Catch --- #
                self.caught_fruits += 1
                if fruit['type'] == 'green':
                    self.score += 1
                    step_reward += 0.1
                elif fruit['type'] == 'red':
                    self.score += 3
                    step_reward += 0.3
                elif fruit['type'] == 'blue':
                    self.score = max(0, self.score - 1)
                    step_reward -= 0.1
                
                self._spawn_particles(fruit['rect'].center, fruit['color'], 20)
                self.fruits.remove(fruit)
                continue

            # Check for miss
            if fruit['rect'].top > self.HEIGHT:
                # --- Sound effect: Miss --- #
                if fruit['type'] != 'blue': # Missing a blue fruit is good
                    self.missed_fruits += 1
                    step_reward -= 0.1
                else:
                    step_reward += 0.05 # Small reward for avoiding penalty
                
                self._spawn_particles( (fruit['rect'].centerx, self.HEIGHT - 5), (100,100,100), 10)
                self.fruits.remove(fruit)
        
        return step_reward

    def _spawn_fruit(self):
        x = random.randint(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS)
        y = -self.FRUIT_RADIUS
        
        fruit_type_roll = self.np_random.random()
        if fruit_type_roll < 0.70:
            fruit_type = 'green'
            color = self.COLOR_FRUIT_GREEN
        elif fruit_type_roll < 0.85:
            fruit_type = 'red'
            color = self.COLOR_FRUIT_RED
        else:
            fruit_type = 'blue'
            color = self.COLOR_FRUIT_BLUE
            
        self.fruits.append({
            'pos': [x, y],
            'rect': pygame.Rect(x - self.FRUIT_RADIUS, y - self.FRUIT_RADIUS, self.FRUIT_RADIUS * 2, self.FRUIT_RADIUS * 2),
            'type': fruit_type,
            'color': color,
        })

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['radius'] -= 0.05
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fruit_fall_speed += self.DIFFICULTY_INCREASE

    def _check_termination(self):
        return (
            self.caught_fruits >= self.WIN_CONDITION or
            self.missed_fruits >= self.LOSS_CONDITION or
            self.steps >= self.MAX_STEPS
        )
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_baskets()
        self._render_fruits()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_baskets(self):
        for basket in self.baskets:
            # Shadow
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, basket.move(3, 3), border_radius=5)
            # Main basket
            pygame.draw.rect(self.screen, self.COLOR_BASKET, basket, border_radius=5)
            # Rim
            pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, (basket.x, basket.y, basket.width, 5), border_top_left_radius=5, border_top_right_radius=5)
            
    def _render_fruits(self):
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = int(self.FRUIT_RADIUS)
            
            # Glow effect for red fruits
            if fruit['type'] == 'red':
                glow_radius = int(radius * 1.5)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*fruit['color'], 80), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Anti-aliased circle for the fruit
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, fruit['color'])

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(max(0, p['radius']))
            alpha = int(max(0, min(255, (p['life'] / 20) * 255)))
            color_with_alpha = (*p['color'], alpha)
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
            
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = f"MISSES: {self.missed_fruits}/{self.LOSS_CONDITION}"
        miss_color = self.COLOR_TEXT if self.missed_fruits < self.LOSS_CONDITION - 3 else self.COLOR_FRUIT_RED
        miss_surf = self.font_main.render(miss_text, True, miss_color)
        self.screen.blit(miss_surf, (self.WIDTH - miss_surf.get_width() - 10, 10))

        # Catches
        catch_text = f"CAUGHT: {self.caught_fruits}/{self.WIN_CONDITION}"
        catch_surf = self.font_small.render(catch_text, True, self.COLOR_TEXT)
        self.screen.blit(catch_surf, (10, 40))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.caught_fruits >= self.WIN_CONDITION:
                end_text = "YOU WIN!"
                end_color = self.COLOR_FRUIT_GREEN
            else:
                end_text = "GAME OVER"
                end_color = self.COLOR_FRUIT_RED
            
            end_surf = self.font_main.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
            "caught_fruits": self.caught_fruits,
        }

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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    env = GameEnv()
    
    # To run interactively (requires a display)
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    # env = GameEnv(render_mode="human") # A proper human render mode would be needed
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    # Simulate a few steps with random actions
    while not done and step_count < 500:
        action = env.action_space.sample() # Replace with agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Info: {info}, Step Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    print(f"Episode finished after {step_count} steps. Final Info: {info}, Final Total Reward: {total_reward:.2f}")
    env.close()