
# Generated: 2025-08-28T02:59:45.119898
# Source Brief: brief_04636.md
# Brief Index: 4636

        
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
        "Controls: Use ← and → to move the basket. Catch the fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in your basket to score points. "
        "Reach 25 points to win, but miss 5 and you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_BASKET = (139, 69, 19)  # Brown
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    
    FRUIT_COLORS = [
        (255, 65, 54),   # Red
        (46, 204, 64),   # Green
        (255, 220, 0),   # Yellow
        (255, 133, 27),  # Orange
    ]

    BASKET_WIDTH = 90
    BASKET_HEIGHT = 20
    BASKET_SPEED = 12
    
    FRUIT_RADIUS = 12
    INITIAL_FRUIT_SPEED = 2.0
    FRUIT_SPEED_INCREASE = 0.1
    
    MAX_FRUITS_ON_SCREEN = 4
    
    WIN_SCORE = 25
    LOSE_MISSES = 5
    MAX_STEPS = 3000 # Increased to allow for longer games

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Internal state RNG
        self._np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.game_over = False
        
        self.basket_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 10)
        
        self.fruits = []
        self.particles = []
        
        self.current_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.last_difficulty_increase_score = -1

        # Initial fruit spawn
        for _ in range(self.MAX_FRUITS_ON_SCREEN // 2):
            self._spawn_fruit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0.0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # --- Update Game Logic ---
            
            # 1. Move basket
            if movement == 3:  # Left
                self.basket_pos.x -= self.BASKET_SPEED
                reward -= 0.01 # Small penalty for movement to encourage efficiency
            elif movement == 4:  # Right
                self.basket_pos.x += self.BASKET_SPEED
                reward -= 0.01

            self.basket_pos.x = np.clip(
                self.basket_pos.x, 
                self.BASKET_WIDTH / 2, 
                self.SCREEN_WIDTH - self.BASKET_WIDTH / 2
            )

            # 2. Update fruits and check for interactions
            basket_rect = pygame.Rect(
                self.basket_pos.x - self.BASKET_WIDTH / 2, 
                self.basket_pos.y - self.BASKET_HEIGHT / 2, 
                self.BASKET_WIDTH, 
                self.BASKET_HEIGHT
            )

            for fruit in reversed(self.fruits):
                fruit['pos'].y += self.current_fruit_speed
                
                fruit_rect = pygame.Rect(
                    fruit['pos'].x - self.FRUIT_RADIUS,
                    fruit['pos'].y - self.FRUIT_RADIUS,
                    self.FRUIT_RADIUS * 2,
                    self.FRUIT_RADIUS * 2
                )

                # Check for catch
                if basket_rect.colliderect(fruit_rect):
                    # Sound effect placeholder: # sfx_catch.play()
                    self.score += 1
                    catch_reward = 1.0
                    
                    # Risky catch bonus
                    dist_from_center = abs(fruit['pos'].x - self.basket_pos.x)
                    if dist_from_center > self.BASKET_WIDTH * 0.35: # In outer 15% on each side
                        catch_reward += 1.0

                    reward += catch_reward
                    self._spawn_particles(fruit['pos'], fruit['color'], 20)
                    self.fruits.remove(fruit)
                    
                    # Difficulty scaling
                    if self.score > 0 and self.score % 5 == 0 and self.score != self.last_difficulty_increase_score:
                        self.current_fruit_speed += self.FRUIT_SPEED_INCREASE
                        self.last_difficulty_increase_score = self.score

                # Check for miss
                elif fruit['pos'].y > self.SCREEN_HEIGHT + self.FRUIT_RADIUS:
                    # Sound effect placeholder: # sfx_miss.play()
                    self.missed_fruits += 1
                    reward -= 1.0
                    self.fruits.remove(fruit)

            # 3. Update particles
            self._update_particles()
            
            # 4. Spawn new fruits
            if len(self.fruits) < self.MAX_FRUITS_ON_SCREEN:
                # Use a probability to make spawning less predictable
                if self._np_random.random() < 0.05:
                    self._spawn_fruit()

        self.steps += 1
        
        # 5. Check for termination
        terminated = (
            self.score >= self.WIN_SCORE or 
            self.missed_fruits >= self.LOSE_MISSES or
            self.steps >= self.MAX_STEPS
        )
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
            else:
                reward -= 100.0 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_fruit(self):
        x_pos = self._np_random.integers(self.FRUIT_RADIUS, self.SCREEN_WIDTH - self.FRUIT_RADIUS)
        y_pos = -self.FRUIT_RADIUS
        color = self.FRUIT_COLORS[self._np_random.integers(0, len(self.FRUIT_COLORS))]
        
        self.fruits.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'color': color,
        })
        
    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self._np_random.uniform(0, 2 * math.pi)
            speed = self._np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self._np_random.integers(20, 40),
                'color': color
            })
            
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _render_text_with_shadow(self, text, font, color, pos):
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        text_surface_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(text_surface_shadow, shadow_pos)

        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

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
        # Draw particles first (background layer)
        for p in self.particles:
            size = max(0, int(p['lifespan'] / 8))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'].x), int(fruit['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])

        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_pos.x - self.BASKET_WIDTH / 2, 
            self.basket_pos.y - self.BASKET_HEIGHT / 2, 
            self.BASKET_WIDTH, 
            self.BASKET_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        
    def _render_ui(self):
        # Score display
        score_text = f"Score: {self.score}"
        self._render_text_with_shadow(score_text, self.font_main, self.COLOR_TEXT, (10, 10))
        
        # Misses display
        misses_text = f"Misses: {self.missed_fruits}/{self.LOSE_MISSES}"
        self._render_text_with_shadow(misses_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10))
        
        # Game over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "You Win!"
                color = (0, 255, 0)
            else:
                msg = "Game Over"
                color = (255, 0, 0)
            
            text_surface = self.font_gameover.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surface = self.font_gameover.render(msg, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surface.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))

            self.screen.blit(shadow_surface, shadow_rect)
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
            "fruit_speed": self.current_fruit_speed,
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Set SDL_VIDEODRIVER to a dummy value for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    running = True
    total_reward = 0

    while running:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}")

        if terminated or truncated:
            print(f"--- GAME OVER ---")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Render the Observation to the Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()