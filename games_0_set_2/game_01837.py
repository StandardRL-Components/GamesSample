
# Generated: 2025-08-28T02:52:09.166537
# Source Brief: brief_01837.md
# Brief Index: 1837

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use ← and → to move the basket and catch the falling fruit."
    )

    game_description = (
        "Catch the falling fruits to score points! Reach 100 catches to win, but be "
        "careful, if you miss 10 fruits, it's game over. Chain catches for a combo bonus!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WIN_SCORE = 100
        self.LOSE_MISSES = 10
        self.MAX_STEPS = 2000
        self.GROUND_HEIGHT = 30
        
        self.CATCHER_WIDTH = 80
        self.CATCHER_HEIGHT = 20
        self.CATCHER_SPEED = 10

        self.MAX_FRUITS = 10
        self.FRUIT_SPAWN_INTERVAL_MIN = 20
        self.FRUIT_SPAWN_INTERVAL_MAX = 40
        self.INITIAL_FRUIT_SPEED = 2.0
        self.FRUIT_SPEED_INCREASE = 0.05
        
        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (210, 240, 255) # Lighter Sky Blue
        self.COLOR_GROUND = (60, 179, 113)   # Medium Sea Green
        self.COLOR_CATCHER = (255, 255, 255) # White
        self.COLOR_CATCHER_OUTLINE = (100, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_OUTLINE = (50, 50, 50)
        self.FRUIT_PALETTE = {
            "apple": (220, 20, 60),   # Crimson
            "banana": (255, 255, 0),  # Yellow
            "orange": (255, 140, 0)   # DarkOrange
        }
        self.PARTICLE_COLORS = [
            (255, 255, 0), (255, 180, 0), (255, 215, 0) # Yellows and Golds
        ]

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
        
        # Fonts
        self.ui_font = pygame.font.Font(None, 36)
        self.combo_font = pygame.font.Font(None, 48)
        self.game_over_font = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.catcher_pos = None
        self.fruits = None
        self.particles = None
        self.steps = None
        self.score = None
        self.fruits_caught = None
        self.fruits_missed = None
        self.combo = None
        self.game_over = None
        self.win_condition = None
        self.fruit_spawn_timer = None
        self.base_fruit_speed = None
        self.combo_display_timer = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.catcher_pos = [
            self.SCREEN_WIDTH // 2 - self.CATCHER_WIDTH // 2,
            self.SCREEN_HEIGHT - self.GROUND_HEIGHT - self.CATCHER_HEIGHT
        ]
        self.fruits = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.combo = 0
        
        self.game_over = False
        self.win_condition = False
        
        self.fruit_spawn_timer = self.np_random.integers(10, 20)
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.combo_display_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.2  # Small penalty per step to encourage efficiency
        
        # Unpack factorized action
        movement = action[0]
        
        self._handle_input(movement)
        
        if not self.game_over:
            catch_reward = self._update_game_state()
            reward += catch_reward
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win_condition:
                reward += 100
            else: # Loss or max steps
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.catcher_pos[0] -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_pos[0] += self.CATCHER_SPEED
        
        # Clamp catcher position to screen boundaries
        self.catcher_pos[0] = max(0, min(self.catcher_pos[0], self.SCREEN_WIDTH - self.CATCHER_WIDTH))

    def _update_game_state(self):
        catch_reward = 0

        # --- Fruit Spawning ---
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0 and len(self.fruits) < self.MAX_FRUITS:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.np_random.integers(
                self.FRUIT_SPAWN_INTERVAL_MIN, self.FRUIT_SPAWN_INTERVAL_MAX
            )

        # --- Difficulty Scaling ---
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED + (self.fruits_caught // 25) * self.FRUIT_SPEED_INCREASE

        # --- Fruit & Collision Logic ---
        catcher_rect = pygame.Rect(self.catcher_pos[0], self.catcher_pos[1], self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        
        for fruit in self.fruits[:]:
            fruit['y'] += self.base_fruit_speed * fruit['speed_mod']
            fruit_rect = pygame.Rect(fruit['x'] - fruit['size'] // 2, fruit['y'] - fruit['size'] // 2, fruit['size'], fruit['size'])

            if catcher_rect.colliderect(fruit_rect):
                # --- Catch ---
                self.fruits_caught += 1
                self.score += 1
                self.combo += 1
                
                catch_reward += 1 + self.combo # Base reward + combo bonus
                self.combo_display_timer = 60 # Show combo text for 2 seconds (at 30fps)
                
                self._create_particles(fruit['x'], fruit['y'])
                self.fruits.remove(fruit)
                # Placeholder: # pygame.mixer.Sound('catch.wav').play()
            elif fruit['y'] > self.SCREEN_HEIGHT - self.GROUND_HEIGHT:
                # --- Miss ---
                self.fruits_missed += 1
                self.combo = 0
                self.fruits.remove(fruit)
                # Placeholder: # pygame.mixer.Sound('miss.wav').play()

        # --- Update Particles ---
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Update Combo Display Timer ---
        if self.combo_display_timer > 0:
            self.combo_display_timer -= 1
            
        return catch_reward

    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(list(self.FRUIT_PALETTE.keys()))
        size = self.np_random.integers(20, 31)
        self.fruits.append({
            'x': self.np_random.integers(size, self.SCREEN_WIDTH - size),
            'y': -size,
            'type': fruit_type,
            'size': size,
            'speed_mod': self.np_random.uniform(0.9, 1.2)
        })

    def _create_particles(self, x, y):
        for _ in range(20):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-4, -1),
                'lifespan': self.np_random.integers(20, 40),
                'color': random.choice(self.PARTICLE_COLORS),
                'size': self.np_random.integers(2, 5)
            })

    def _check_termination(self):
        if self.fruits_caught >= self.WIN_SCORE:
            self.game_over = True
            self.win_condition = True
            return True
        if self.fruits_missed >= self.LOSE_MISSES:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate between top and bottom colors
            ratio = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT))

        # Draw Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

        # Draw Catcher
        catcher_rect = (self.catcher_pos[0], self.catcher_pos[1], self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_OUTLINE, catcher_rect, width=2, border_radius=5)

        # Draw Fruits
        for fruit in self.fruits:
            pos = (int(fruit['x']), int(fruit['y']))
            color = self.FRUIT_PALETTE[fruit['type']]
            size = fruit['size']
            if fruit['type'] == 'apple':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size // 2, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size // 2, color)
            elif fruit['type'] == 'orange':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size // 2, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size // 2, color)
            elif fruit['type'] == 'banana':
                rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 4, size, size // 2)
                pygame.draw.arc(self.screen, color, rect, math.pi, 2 * math.pi, width=size // 4)

    def _render_ui(self):
        # Score and Misses display
        self._draw_text(f"Score: {self.score}", (10, 10), self.ui_font)
        misses_text = f"Missed: {self.fruits_missed}/{self.LOSE_MISSES}"
        misses_surf = self.ui_font.render(misses_text, True, self.COLOR_TEXT)
        self._draw_text(misses_text, (self.SCREEN_WIDTH - misses_surf.get_width() - 10, 10), self.ui_font)

        # Combo display
        if self.combo > 1 and self.combo_display_timer > 0:
            alpha = min(255, int(255 * (self.combo_display_timer / 30)))
            scale = 1 + 0.5 * (1 - self.combo_display_timer / 60)
            
            font = pygame.font.Font(None, int(48 * scale))
            text = f"{self.combo}x Combo!"
            self._draw_text(text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 3), font, alpha=alpha, center=True)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition else "GAME OVER"
            self._draw_text(message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.game_over_font, center=True)

    def _draw_text(self, text, pos, font, color=None, outline_color=None, alpha=255, center=False):
        color = color or self.COLOR_TEXT
        outline_color = outline_color or self.COLOR_TEXT_OUTLINE
        
        text_surf = font.render(text, True, color)
        outline_surf = font.render(text, True, outline_color)

        if alpha < 255:
            text_surf.set_alpha(alpha)
            outline_surf.set_alpha(alpha)

        x, y = pos
        if center:
            x -= text_surf.get_width() // 2
            y -= text_surf.get_height() // 2

        # Draw outline by blitting in 4 directions
        self.screen.blit(outline_surf, (x - 1, y - 1))
        self.screen.blit(outline_surf, (x + 1, y - 1))
        self.screen.blit(outline_surf, (x - 1, y + 1))
        self.screen.blit(outline_surf, (x + 1, y + 1))
        # Draw main text
        self.screen.blit(text_surf, (x, y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed,
            "combo": self.combo,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Example ---
    # To run this, you need to install pygame and gymnasium
    # And uncomment the pygame.display.set_mode line
    # And change the render_mode in the GameEnv constructor
    
    # env = GameEnv(render_mode="human")
    # screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # pygame.display.set_caption("Fruit Catcher")
    
    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()
    
    # while not done:
    #     movement = 0 # No-op
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         movement = 3
    #     if keys[pygame.K_RIGHT]:
    #         movement = 4
            
    #     action = [movement, 0, 0] # Movement, space, shift
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
        
    #     # Render to screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     clock.tick(30) # Run at 30 FPS
        
    # print(f"Game Over! Final Info: {info}")
    # env.close()
    
    # --- Basic RL Agent Loop Example ---
    obs, info = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = env.action_space.sample()  # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    env.close()