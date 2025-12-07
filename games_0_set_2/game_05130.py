
# Generated: 2025-08-28T04:04:01.188562
# Source Brief: brief_05130.md
# Brief Index: 5130

        
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
    user_guide = "Controls: ←→ to move the basket."

    # Must be a short, user-facing description of the game:
    game_description = "Catch falling fruit in a top-down arcade game to achieve a high score before missing too many."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For interpolation and game speed
        
        # Colors
        self.COLOR_BG_TOP = pygame.Color("#74BDE0")
        self.COLOR_BG_BOTTOM = pygame.Color("#ADD8E6")
        self.COLOR_TEXT = pygame.Color("white")
        self.COLOR_SHADOW = pygame.Color(0, 0, 0, 100)
        self.FRUIT_COLORS = {
            1: pygame.Color("#FF4136"), # Red
            2: pygame.Color("#FFDC00"), # Yellow
            3: pygame.Color("#2ECC40"), # Green
        }
        self.COMBO_COLORS = [
            pygame.Color("#FFFFFF"), # 0-4
            pygame.Color("#FFFF00"), # 5-9
            pygame.Color("#FFA500"), # 10-14
            pygame.Color("#FF4500"), # 15+
        ]

        # Player/Basket
        self.BASKET_WIDTH = 100
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 12

        # Fruit
        self.MAX_FRUITS_ON_SCREEN = 10
        self.FRUIT_SPAWN_RATE_INITIAL = 45 # Lower is faster
        self.FRUIT_SPAWN_RATE_MIN = 15
        
        # Game Rules
        self.MAX_MISSES = 5
        self.WIN_CONDITION_CATCHES = 50
        self.MAX_STEPS = 1000 * self.FPS // 30 # Scale max steps if FPS changes

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
        
        # Fonts
        self.font_s = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_m = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_l = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.basket_pos_x = self.WIDTH / 2
        
        self.fruits = []
        self.particles = []
        
        self.missed_fruits = 0
        self.fruits_caught = 0
        self.combo = 0
        self.combo_pop_timer = 0
        
        self.fruit_spawn_timer = self.FRUIT_SPAWN_RATE_INITIAL
        self.base_fruit_speed = 3.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for existing

        if not self.game_over:
            self._handle_input(action)
            
            # Update game state
            self._spawn_fruit()
            reward_delta = self._update_fruits()
            reward += reward_delta
            self._update_particles()
            
            self._update_difficulty()
            self.steps += 1
        
        # Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over: # Terminal state reached on this step
            self.game_over = True
            if self.win_state:
                reward += 50 # Win bonus
            else:
                reward -= 50 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos_x += self.BASKET_SPEED
        
        # Clamp basket position to screen bounds
        self.basket_pos_x = max(self.BASKET_WIDTH / 2, self.basket_pos_x)
        self.basket_pos_x = min(self.WIDTH - self.BASKET_WIDTH / 2, self.basket_pos_x)

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0 and len(self.fruits) < self.MAX_FRUITS_ON_SCREEN:
            spawn_x = self.np_random.integers(20, self.WIDTH - 20)
            speed = self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5)
            
            # Determine fruit type/value
            rand_val = self.np_random.random()
            if rand_val < 0.6: # 60% chance
                value = 1
                radius = 10
            elif rand_val < 0.9: # 30% chance
                value = 2
                radius = 12
            else: # 10% chance
                value = 3
                radius = 14

            self.fruits.append({
                'pos': np.array([spawn_x, -radius], dtype=np.float32),
                'speed': speed,
                'radius': radius,
                'value': value,
                'color': self.FRUIT_COLORS[value]
            })
            
            current_spawn_rate = self.FRUIT_SPAWN_RATE_INITIAL - (self.fruits_caught // 10) * 5
            self.fruit_spawn_timer = max(self.FRUIT_SPAWN_RATE_MIN, current_spawn_rate)

    def _update_fruits(self):
        reward_delta = 0
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.BASKET_WIDTH / 2, 
            self.HEIGHT - self.BASKET_HEIGHT - 5, 
            self.BASKET_WIDTH, 
            self.BASKET_HEIGHT
        )

        fruits_to_remove = []
        for fruit in self.fruits:
            fruit['pos'][1] += fruit['speed']
            
            # Check for catch
            if basket_rect.collidepoint(fruit['pos']):
                # sfx: catch_sound.play()
                reward_delta += fruit['value'] + self.combo # Reward for catch + combo bonus
                self.score += fruit['value'] * (1 + self.combo // 5) # Score bonus for high combo
                self.fruits_caught += 1
                self.combo += 1
                self.combo_pop_timer = 10 # Start combo text animation
                self._create_particles(fruit['pos'], fruit['color'])
                fruits_to_remove.append(fruit)

            # Check for miss
            elif fruit['pos'][1] > self.HEIGHT + fruit['radius']:
                # sfx: miss_sound.play()
                reward_delta -= 2 # Penalty for missing
                self.missed_fruits += 1
                self.combo = 0
                fruits_to_remove.append(fruit)

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return reward_delta

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _update_difficulty(self):
        self.base_fruit_speed = 3.0 + (self.fruits_caught // 10) * 0.25
        
    def _check_termination(self):
        if self.missed_fruits >= self.MAX_MISSES:
            self.win_state = False
            return True
        if self.fruits_caught >= self.WIN_CONDITION_CATCHES:
            self.win_state = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.win_state = False
            return True
        return False
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = self.COLOR_BG_TOP.lerp(self.COLOR_BG_BOTTOM, interp)
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color']
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), 2,
                (color.r, color.g, color.b, alpha)
            )

        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            radius = fruit['radius']
            color = fruit['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            
        # Draw basket
        combo_level = min(len(self.COMBO_COLORS) - 1, self.combo // 5)
        basket_color = self.COMBO_COLORS[combo_level]
        basket_rect = pygame.Rect(0, 0, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        basket_rect.center = (int(self.basket_pos_x), int(self.HEIGHT - self.BASKET_HEIGHT / 2 - 5))
        pygame.draw.rect(self.screen, basket_color, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, basket_color.lerp("black", 0.3), basket_rect, width=2, border_radius=5)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_m, self.COLOR_TEXT, (10, 10))
        
        # Misses
        miss_text = self._render_text("MISSES:", self.font_s, self.COLOR_TEXT, (self.WIDTH - 150, 15))
        for i in range(self.MAX_MISSES):
            color = pygame.Color("red") if i < self.missed_fruits else pygame.Color(100, 100, 100)
            pygame.draw.circle(self.screen, color, (self.WIDTH - 70 + i * 20, 24), 7)
            pygame.draw.circle(self.screen, "black", (self.WIDTH - 70 + i * 20, 24), 7, 1)

        # Combo
        if self.combo > 1:
            size_pop = 0
            if self.combo_pop_timer > 0:
                size_pop = math.sin((10 - self.combo_pop_timer) / 10 * math.pi) * 10
                self.combo_pop_timer -= 1
            
            combo_font = pygame.font.SysFont("Arial", int(28 + size_pop), bold=True)
            combo_color = self.COMBO_COLORS[min(len(self.COMBO_COLORS) - 1, self.combo // 5)]
            self._render_text(f"{self.combo}x COMBO!", combo_font, combo_color, (self.WIDTH/2, 20), center=True)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "YOU WIN!"
                color = pygame.Color("gold")
            else:
                msg = "GAME OVER"
                color = pygame.Color("tomato")
            
            self._render_text(msg, self.font_l, color, (self.WIDTH / 2, self.HEIGHT / 2 - 30), center=True)
            self._render_text(f"Final Score: {self.score}", self.font_m, self.COLOR_TEXT, (self.WIDTH / 2, self.HEIGHT / 2 + 30), center=True)

    def _render_text(self, text, font, color, pos, center=False):
        shadow_surf = font.render(text, True, self.COLOR_SHADOW)
        text_surf = font.render(text, True, color)
        
        shadow_rect = shadow_surf.get_rect()
        text_rect = text_surf.get_rect()

        if center:
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            text_rect.center = pos
        else:
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        return text_rect

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "missed_fruits": self.missed_fruits,
            "combo": self.combo,
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

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up display for human playing
    pygame.display.set_caption(GameEnv.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()