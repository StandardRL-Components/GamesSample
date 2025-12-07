
# Generated: 2025-08-27T17:12:22.738750
# Source Brief: brief_01460.md
# Brief Index: 1460

        
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
    A Gymnasium environment for a fruit-catching arcade game.

    The player controls a basket at the bottom of the screen and must catch
    fruits that fall from the top. The game rewards catching fruits and
    penalizes missing them. Difficulty increases as more fruits are caught.

    **Visuals:**
    - Bright, cartoonish aesthetic with a sky-blue gradient background.
    - Fruits are colorful circles with small stems.
    - The player's basket is a simple brown trapezoid.
    - UI elements for score, lives, and combo are clearly displayed.
    - Particle effects provide satisfying feedback for catching fruits.

    **Gameplay:**
    - Move the basket left and right to intercept falling fruits.
    - Catching fruits increases the score and combo multiplier.
    - Missing fruits depletes lives and resets the combo.
    - The game ends by catching 25 fruits (win) or missing 5 (loss).
    - Fruit falling speed increases progressively to scale the difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: ←→ to move the basket."

    # User-facing description of the game
    game_description = "Catch falling fruit in a top-down arcade game to achieve a high score before missing too many."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 25
        self.LOSS_CONDITION = 5
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 10.0
        self.FRUIT_RADIUS = 12
        self.INITIAL_FRUIT_SPEED = 2.0
        self.FRUIT_SPAWN_RATE = 45  # Lower is faster

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (240, 248, 255) # Alice Blue
        self.COLOR_BASKET = (139, 69, 19)    # Brown
        self.COLOR_TEXT = (30, 30, 30)
        self.COLOR_TEXT_SHADOW = (200, 200, 200)
        self.FRUIT_COLORS = [
            (220, 20, 60),  # Apple Red
            (255, 215, 0),  # Banana Yellow
            (255, 165, 0),  # Orange Orange
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize state variables
        # These are set properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.basket_x = 0
        self.lives = 0
        self.combo = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.base_fruit_speed = 0
        self.fruit_spawn_timer = 0
        self.fruits = []
        self.particles = []
        self.combo_flash_timer = 0

        # Call reset to ensure a valid initial state
        self.reset()
        
        # Validate the implementation against Gymnasium standards
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.basket_x = self.WIDTH / 2
        self.lives = self.LOSS_CONDITION
        self.combo = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.fruit_spawn_timer = 0
        self.combo_flash_timer = 0

        self.fruits.clear()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Actions & Continuous Reward ---
        movement = action[0]
        
        # Find closest fruit for continuous reward calculation
        closest_fruit = None
        if self.fruits:
            closest_fruit = min(self.fruits, key=lambda f: f['pos'][1])
            dist_before = abs(closest_fruit['pos'][0] - self.basket_x)

        # Update basket position
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        self.basket_x = np.clip(self.basket_x, self.BASKET_WIDTH / 2, self.WIDTH - self.BASKET_WIDTH / 2)

        # Calculate continuous reward
        if closest_fruit:
            dist_after = abs(closest_fruit['pos'][0] - self.basket_x)
            if dist_after < dist_before:
                reward += 1.0  # Moving towards fruit
            else:
                reward -= 0.1  # Moving away or staying still

        # --- 2. Update Game State ---
        
        # Update combo flash timer
        if self.combo_flash_timer > 0:
            self.combo_flash_timer -= 1

        # Update and remove old particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # Update fruits
        for fruit in reversed(self.fruits):
            fruit['pos'][1] += fruit['speed']
            
            # Check for catch
            basket_rect = pygame.Rect(self.basket_x - self.BASKET_WIDTH / 2, self.HEIGHT - self.BASKET_HEIGHT, self.BASKET_WIDTH, self.BASKET_HEIGHT)
            if basket_rect.collidepoint(fruit['pos']):
                # sfx: catch_sound.play()
                self.fruits.remove(fruit)
                self.fruits_caught += 1
                self.combo += 1
                self.score += 10 + (self.combo * 2)
                reward += 10 + (self.combo * 2)
                self.combo_flash_timer = 15 # Flash for 15 frames
                self._create_particles(fruit['pos'])

                # Increase difficulty
                if self.fruits_caught > 0 and self.fruits_caught % 5 == 0:
                    self.base_fruit_speed += 0.05
            
            # Check for miss
            elif fruit['pos'][1] > self.HEIGHT:
                # sfx: miss_sound.play()
                self.fruits.remove(fruit)
                self.fruits_missed += 1
                self.lives -= 1
                self.combo = 0
                reward -= 10

        # Spawn new fruits
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.FRUIT_SPAWN_RATE

        # --- 3. Check for Termination ---
        terminated = False
        if self.fruits_caught >= self.WIN_CONDITION:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.lives <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_fruit(self):
        fruit_type = self.np_random.integers(0, len(self.FRUIT_COLORS))
        pos = [self.np_random.uniform(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS), -self.FRUIT_RADIUS]
        speed_multiplier = self.np_random.uniform(0.9, 1.2)
        self.fruits.append({
            'pos': pos,
            'type': fruit_type,
            'speed': self.base_fruit_speed * speed_multiplier
        })
        
    def _create_particles(self, pos):
        # sfx: particle_burst.play()
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'size': size})

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            color_ratio = y / self.HEIGHT
            r = self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio
            g = self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio
            b = self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color = (255, 255, 100, alpha) # Yellow particle color
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Draw fruits
        for fruit in self.fruits:
            self._draw_fruit(fruit['pos'], fruit['type'])

        # Draw basket
        self._draw_basket()

        # Draw UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_fruit(self, pos, fruit_type):
        x, y = int(pos[0]), int(pos[1])
        color = self.FRUIT_COLORS[fruit_type]
        
        # Fruit body
        pygame.gfxdraw.aacircle(self.screen, x, y, self.FRUIT_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.FRUIT_RADIUS, color)
        
        # Simple gloss effect
        gloss_pos = (x + self.FRUIT_RADIUS // 3, y - self.FRUIT_RADIUS // 3)
        pygame.gfxdraw.filled_circle(self.screen, gloss_pos[0], gloss_pos[1], self.FRUIT_RADIUS // 4, (255, 255, 255, 100))
        
        # Stem
        stem_color = (139, 69, 19)
        pygame.draw.line(self.screen, stem_color, (x, y - self.FRUIT_RADIUS), (x + 2, y - self.FRUIT_RADIUS - 5), 3)

    def _draw_basket(self):
        x = int(self.basket_x)
        y = self.HEIGHT - self.BASKET_HEIGHT
        half_w = self.BASKET_WIDTH // 2
        
        points = [
            (x - half_w, y),
            (x + half_w, y),
            (x + half_w - 10, y + self.BASKET_HEIGHT),
            (x - half_w + 10, y + self.BASKET_HEIGHT)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_BASKET, points)
        pygame.draw.polygon(self.screen, (0,0,0), points, 2) # Outline

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow=False):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10), shadow=True)

        # Lives
        lives_text = f"Lives: {self.lives}"
        text_w = self.font_small.size(lives_text)[0]
        draw_text(lives_text, self.font_small, self.COLOR_TEXT, (self.WIDTH - text_w - 10, 10), shadow=True)

        # Combo
        if self.combo > 1:
            combo_text = f"{self.combo}x COMBO!"
            font = self.font_large if self.combo_flash_timer > 0 else self.font_small
            color = (255, 100, 0) if self.combo_flash_timer > 0 else self.COLOR_TEXT
            text_w, text_h = font.size(combo_text)
            draw_text(combo_text, font, color, (self.WIDTH / 2 - text_w / 2, 50), shadow=True)
            
        # Game Over Message
        if self.game_over:
            if self.fruits_caught >= self.WIN_CONDITION:
                msg = "YOU WIN!"
                color = (0, 200, 0)
            else:
                msg = "GAME OVER"
                color = (200, 0, 0)
            
            text_w, text_h = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), shadow=True)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage for interactive play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for rendering the environment ---
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(env.game_description)
    print(env.user_guide)

    running = True
    while running:
        # Default action is "no-op"
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            done = False

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()