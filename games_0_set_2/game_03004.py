
# Generated: 2025-08-27T22:04:51.610920
# Source Brief: brief_03004.md
# Brief Index: 3004

        
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
    A Gymnasium environment for a fast-paced arcade game.
    The player controls a basket to catch falling fruit while dodging bombs.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to move the basket. Catch fruit, avoid bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit to score points while dodging the bombs. Catch 30 fruits to win, but catch 3 bombs and you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Game constants
        self.WIN_SCORE = 30
        self.MAX_BOMBS = 3
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

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
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (220, 240, 255) # Lighter Blue
        self.COLOR_BASKET = (139, 69, 19) # SaddleBrown
        self.COLOR_BASKET_RIM = (101, 49, 0)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_SKULL = (240, 240, 240)
        self.COLOR_APPLE = (220, 20, 60) # Crimson
        self.COLOR_ORANGE = (255, 140, 0) # DarkOrange
        self.COLOR_LEMON = (255, 255, 0) # Yellow
        self.COLOR_STEM = (34, 139, 34) # ForestGreen
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_BG = (0, 0, 0)
        self.COLOR_OVERLAY = (0, 0, 0, 150)

        # Game state variables are initialized in reset()
        self.basket_pos_x = 0
        self.basket_width = 0
        self.basket_height = 0
        self.basket_speed = 0
        self.fruits = []
        self.bombs = []
        self.fall_speed_initial = 0
        self.fall_speed = 0
        self.spawn_prob = 0
        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.game_over_message = ""

        # Initialize state
        self.reset()
        
        # Run validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.game_over_message = ""

        self.basket_width = 100
        self.basket_height = 20
        self.basket_pos_x = self.WIDTH // 2
        self.basket_speed = 10

        self.fruits = []
        self.bombs = []

        self.fall_speed_initial = 2.0
        self.fall_speed = self.fall_speed_initial
        self.spawn_prob = 0.06

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game by one time step.
        """
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack action
            movement = action[0]  # 0-4: none/up/down/left/right

            # --- 1. Handle Player Input ---
            if movement == 3:  # Left
                self.basket_pos_x -= self.basket_speed
            elif movement == 4:  # Right
                self.basket_pos_x += self.basket_speed

            # Clamp basket position to screen bounds
            self.basket_pos_x = max(
                self.basket_width // 2,
                min(self.basket_pos_x, self.WIDTH - self.basket_width // 2)
            )

            # --- 2. Update Game State ---
            self.steps += 1

            # Difficulty scaling
            if self.steps > 0 and self.steps % 100 == 0:
                self.fall_speed += 0.1

            # Spawn new items
            if self.np_random.random() < self.spawn_prob:
                self._spawn_item()

            # Update fruits
            basket_rect = pygame.Rect(
                self.basket_pos_x - self.basket_width // 2,
                self.HEIGHT - self.basket_height - 10,
                self.basket_width,
                self.basket_height
            )

            for fruit in self.fruits[::-1]:
                fruit['pos'][1] += self.fall_speed
                fruit_rect = pygame.Rect(fruit['pos'][0] - fruit['size'], fruit['pos'][1] - fruit['size'], fruit['size']*2, fruit['size']*2)
                
                if basket_rect.colliderect(fruit_rect):
                    # SFX: Catch fruit sound
                    self.score += 1
                    reward += 1
                    self.fruits.remove(fruit)
                elif fruit['pos'][1] > self.HEIGHT + fruit['size']:
                    self.fruits.remove(fruit)

            # Update bombs
            for bomb in self.bombs[::-1]:
                bomb['pos'][1] += self.fall_speed
                bomb_rect = pygame.Rect(bomb['pos'][0] - bomb['size'], bomb['pos'][1] - bomb['size'], bomb['size']*2, bomb['size']*2)

                if basket_rect.colliderect(bomb_rect):
                    # SFX: Explosion sound
                    self.bombs_caught += 1
                    reward -= 5
                    self.bombs.remove(bomb)
                elif bomb['pos'][1] > self.HEIGHT + bomb['size']:
                    self.bombs.remove(bomb)

            # --- 3. Check for Termination ---
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100
                self.game_over_message = "YOU WIN!"
            elif self.bombs_caught >= self.MAX_BOMBS:
                self.game_over = True
                terminated = True
                reward -= 100
                self.game_over_message = "GAME OVER"
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
                self.game_over_message = "TIME'S UP!"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_item(self):
        """Spawns a new fruit or bomb at a random horizontal position."""
        x_pos = self.np_random.integers(20, self.WIDTH - 20)
        
        # 75% chance for fruit, 25% for bomb
        if self.np_random.random() < 0.75:
            fruit_type = self.np_random.choice(['apple', 'orange', 'lemon'])
            size = self.np_random.integers(12, 18)
            self.fruits.append({'pos': [x_pos, -size], 'type': fruit_type, 'size': size})
        else:
            size = 15
            self.bombs.append({'pos': [x_pos, -size], 'size': size})

    def _get_observation(self):
        """
        Renders the current game state to a numpy array.
        """
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Render all game elements
        self._render_game_elements()

        # Render UI overlay
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        """Renders the fruits, bombs, and basket."""
        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            size = fruit['size']
            if fruit['type'] == 'apple':
                color = self.COLOR_APPLE
            elif fruit['type'] == 'orange':
                color = self.COLOR_ORANGE
            else: # lemon
                color = self.COLOR_LEMON
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            # Draw stem
            pygame.draw.line(self.screen, self.COLOR_STEM, (pos[0], pos[1] - size), (pos[0]+2, pos[1] - size - 5), 3)

        # Draw bombs
        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            size = bomb['size']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
            self._draw_skull(pos, size)

        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.basket_width // 2,
            self.HEIGHT - self.basket_height - 10,
            self.basket_width,
            self.basket_height
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, basket_rect, width=3, border_radius=5)

    def _draw_skull(self, pos, size):
        """Draws a simple skull icon on a bomb."""
        skull_color = self.COLOR_BOMB_SKULL
        # Head
        head_radius = int(size * 0.6)
        pygame.draw.circle(self.screen, skull_color, pos, head_radius)
        # Eyes
        eye_radius = int(size * 0.15)
        eye_offset_x = int(size * 0.25)
        eye_offset_y = int(size * 0.1)
        pygame.draw.circle(self.screen, self.COLOR_BOMB, (pos[0] - eye_offset_x, pos[1] - eye_offset_y), eye_radius)
        pygame.draw.circle(self.screen, self.COLOR_BOMB, (pos[0] + eye_offset_x, pos[1] - eye_offset_y), eye_radius)
        # Jaw
        jaw_y = pos[1] + int(size * 0.3)
        for i in range(4):
            x = pos[0] - int(size*0.3) + i * int(size*0.2)
            pygame.draw.line(self.screen, self.COLOR_BOMB, (x, jaw_y), (x, jaw_y + int(size*0.2)), 1)


    def _render_ui(self):
        """Renders the score and bombs caught UI."""
        # Score display
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_bg_rect = pygame.Rect(5, 5, score_text.get_width() + 20, score_text.get_height() + 10)
        pygame.draw.rect(self.screen, self.COLOR_TEXT_BG, score_bg_rect, border_radius=5)
        self.screen.blit(score_text, (15, 10))

        # Bombs caught display
        bomb_ui_y = score_bg_rect.bottom + 10
        for i in range(self.MAX_BOMBS):
            pos = (30 + i * 35, bomb_ui_y)
            size = 12
            if i < self.bombs_caught:
                # Filled skull for caught bomb
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
                self._draw_skull(pos, size)
            else:
                # Outline for remaining lives
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)

    def _render_game_over(self):
        """Renders the game over screen."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        text_surface = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)
        
    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_caught": self.bombs_caught,
            "fall_speed": self.fall_speed,
        }

    def close(self):
        """
        Cleans up Pygame resources.
        """
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key state handling for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_q]:
             running = False
             
        if keys[pygame.K_r] and terminated:
             obs, info = env.reset()
             terminated = False

        # Step the environment
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(30)

    env.close()