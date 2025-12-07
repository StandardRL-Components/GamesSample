
# Generated: 2025-08-27T23:10:24.440990
# Source Brief: brief_03374.md
# Brief Index: 3374

        
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


# Set SDL to dummy to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the basket. Catch the fruit, dodge the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit while dodging bombs in this fast-paced arcade game. Green fruit is worth 1 point, gold is 3, and rare red fruit is 5. Catch 50 fruits to win, but if you catch 3 bombs, you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_CONDITION_FRUITS = 50
        self.LOSE_CONDITION_BOMBS = 3

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (176, 224, 230) # Powder Blue
        self.COLOR_BASKET = (139, 69, 19) # Saddle Brown
        self.COLOR_BASKET_RIM = (160, 82, 45) # Sienna
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.FRUIT_COLORS = {
            "green": (50, 205, 50),
            "gold": (255, 215, 0),
            "red": (220, 20, 60),
        }
        self.FRUIT_REWARDS = {"green": 0.1, "gold": 0.3, "red": 0.5}
        self.BOMB_COLOR = (40, 40, 40)
        self.FUSE_COLOR = (255, 165, 0)
        self.SPARK_COLOR = (255, 255, 0)

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = 0
        self.player_width = 80
        self.player_height = 20
        self.player_speed = 12
        
        self.falling_objects = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.fall_speed = 0.0
        self.level_up_flags = {}

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.lives = self.LOSE_CONDITION_BOMBS
        self.game_over = False
        self.win = False
        
        self.player_pos = self.WIDTH // 2
        self.fall_speed = 2.0
        
        self.falling_objects.clear()
        self.particles.clear()
        
        # Reset difficulty scaling flags
        self.level_up_flags = {i * 10: False for i in range(1, self.WIN_CONDITION_FRUITS // 10 + 1)}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
            
        reward = 0
        self.steps += 1
        
        # --- Handle player input ---
        movement = action[0]
        if movement == 3:  # Left
            self.player_pos -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos += self.player_speed
        
        self.player_pos = np.clip(self.player_pos, self.player_width // 2, self.WIDTH - self.player_width // 2)

        # --- Game Logic ---
        if not self.game_over:
            # Spawn new objects
            if self.np_random.random() < 0.04:
                self._spawn_object()

            # Update falling objects
            for obj in self.falling_objects[:]:
                obj['y'] += self.fall_speed
                if obj['y'] > self.HEIGHT + 20:
                    self.falling_objects.remove(obj)
            
            # Update particles
            for p in self.particles[:]:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['life'] -= 1
                if p['life'] <= 0:
                    self.particles.remove(p)

            # Check for collisions
            player_rect = pygame.Rect(self.player_pos - self.player_width // 2, self.HEIGHT - self.player_height - 10, self.player_width, self.player_height)
            for obj in self.falling_objects[:]:
                obj_rect = pygame.Rect(obj['x'] - obj['radius'], obj['y'] - obj['radius'], obj['radius'] * 2, obj['radius'] * 2)
                if player_rect.colliderect(obj_rect):
                    if obj['type'] == 'bomb':
                        # SFX: Explosion
                        self.lives -= 1
                        reward -= 1.0
                        self._create_explosion(obj['x'], obj['y'])
                    else: # Fruit
                        # SFX: Fruit catch
                        fruit_reward = self.FRUIT_REWARDS[obj['type']]
                        reward += fruit_reward
                        self.score += int(fruit_reward * 10) # 1, 3, or 5 points
                        self.fruits_caught += 1
                        self._create_catch_particles(obj['x'], obj['y'], self.FRUIT_COLORS[obj['type']])
                    
                    self.falling_objects.remove(obj)

            # --- Difficulty Scaling ---
            if self.fruits_caught > 0 and self.fruits_caught % 10 == 0 and self.fruits_caught < self.WIN_CONDITION_FRUITS:
                if not self.level_up_flags.get(self.fruits_caught, False):
                    self.fall_speed += 0.2
                    self.level_up_flags[self.fruits_caught] = True
                    # SFX: Level up chime

        # --- Check Termination Conditions ---
        if self.fruits_caught >= self.WIN_CONDITION_FRUITS and not self.game_over:
            self.game_over = True
            self.win = True
            reward += 100
        
        if self.lives <= 0 and not self.game_over:
            self.game_over = True
            self.win = False
            reward -= 100
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_object(self):
        x = self.np_random.integers(20, self.WIDTH - 20)
        y = -20
        
        if self.np_random.random() < 0.25: # 25% chance of bomb
            self.falling_objects.append({'type': 'bomb', 'x': x, 'y': y, 'radius': 12})
        else: # 75% chance of fruit
            rand_fruit = self.np_random.random()
            if rand_fruit < 0.1: # 10% red
                fruit_type = 'red'
                radius = 12
            elif rand_fruit < 0.4: # 30% gold
                fruit_type = 'gold'
                radius = 10
            else: # 60% green
                fruit_type = 'green'
                radius = 8
            self.falling_objects.append({'type': fruit_type, 'x': x, 'y': y, 'radius': radius})

    def _create_explosion(self, x, y):
        for _ in range(40):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            color = self.np_random.choice([self.BOMB_COLOR, self.FUSE_COLOR, (128,128,128)])
            self.particles.append({
                'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40), 'color': color, 'radius': self.np_random.integers(2, 5)
            })

    def _create_catch_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.random() * math.pi + math.pi/2 # Upward spray
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': -math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30), 'color': color, 'radius': self.np_random.integers(2, 4)
            })
    
    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'])

        # Render falling objects
        for obj in self.falling_objects:
            x, y, r = int(obj['x']), int(obj['y']), obj['radius']
            if obj['type'] == 'bomb':
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.BOMB_COLOR)
                pygame.gfxdraw.aacircle(self.screen, x, y, r, self.BOMB_COLOR)
                # Animated fuse
                fuse_end_x = x + 5
                fuse_end_y = y - 5 - 2 * math.sin(self.steps * 0.5)
                pygame.draw.line(self.screen, self.FUSE_COLOR, (x, y - r), (fuse_end_x, fuse_end_y), 2)
                pygame.gfxdraw.filled_circle(self.screen, int(fuse_end_x), int(fuse_end_y), 3, self.SPARK_COLOR)
            else: # Fruit
                color = self.FRUIT_COLORS[obj['type']]
                highlight_color = (255, 255, 255, 150)
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
                pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
                # Highlight
                pygame.gfxdraw.filled_circle(self.screen, x - r//3, y - r//3, r//3, highlight_color)

        # Render player basket
        basket_rect = pygame.Rect(self.player_pos - self.player_width // 2, self.HEIGHT - self.player_height - 10, self.player_width, self.player_height)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, basket_rect, 3, border_radius=5)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, x, y, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, (x, y))

        # Score
        draw_text(f"Score: {self.score}", self.font_small, 10, 10)
        
        # Lives
        lives_text = "Lives: "
        text_surf = self.font_small.render(lives_text, True, self.COLOR_TEXT)
        text_shadow_surf = self.font_small.render(lives_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(text_shadow_surf, (self.WIDTH - 180 + 2, 12))
        self.screen.blit(text_surf, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            bomb_x = self.WIDTH - 80 + i * 30
            pygame.gfxdraw.filled_circle(self.screen, bomb_x, 25, 8, self.BOMB_COLOR)
            pygame.gfxdraw.aacircle(self.screen, bomb_x, 25, 8, self.BOMB_COLOR)

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            shadow_surf = self.font_large.render(message, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "lives_remaining": self.lives,
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

if __name__ == "__main__":
    # To run and visualize the game
    # This part is for human play and debugging
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting for human player
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering for display ---
        # The observation is already a rendered frame
        # Pygame uses (W, H) but our obs is (H, W), so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()