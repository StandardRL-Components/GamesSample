
# Generated: 2025-08-28T06:06:57.215370
# Source Brief: brief_05790.md
# Brief Index: 5790

        
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
        "Controls: ←→ to move the basket. Catch the fruit and dodge the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must catch falling fruit while dodging bombs. "
        "Your goal is to collect 100 fruit before losing all your lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 3

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_BASKET = (139, 69, 19)
        self.COLOR_BASKET_RIM = (160, 82, 45)
        self.COLOR_BOMB = (200, 0, 0)
        self.COLOR_BOMB_FLASH = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEART = (255, 80, 80)
        self.FRUIT_COLORS = {
            "apple": (220, 40, 40),
            "lemon": (255, 230, 0),
            "grape": (128, 0, 128),
        }

        # Player settings
        self.PLAYER_WIDTH = 80
        self.PLAYER_HEIGHT = 20
        self.PLAYER_SPEED = 12

        # Item settings
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_INCREMENT = 0.5 # Increased from 0.05 for noticeable effect
        self.FRUIT_SPAWN_CHANCE = 0.04
        self.BOMB_SPAWN_CHANCE = 0.02
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 64, bold=True)

        # Initialize state variables
        self.np_random = None
        self.player_pos_x = 0
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.fall_speed = 0.0
        self.last_score_milestone = 0

        self.reset()
        
        # Self-validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.player_pos_x = self.WIDTH / 2
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.steps = 0
        self.game_over = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_score_milestone = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.TARGET_FPS)

        reward = 0
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # --- 1. Handle player input ---
            if movement == 3:  # Left
                self.player_pos_x -= self.PLAYER_SPEED
            elif movement == 4:  # Right
                self.player_pos_x += self.PLAYER_SPEED
            
            # Clamp player position
            self.player_pos_x = max(self.PLAYER_WIDTH / 2, min(self.WIDTH - self.PLAYER_WIDTH / 2, self.player_pos_x))

            # --- 2. Update game state ---
            self._update_difficulty()
            self._spawn_objects()
            
            # Update fruits
            reward += self._update_fruits()
            
            # Update bombs
            reward += self._update_bombs()

            # Update particles
            self._update_particles()
            
        # --- 3. Check for termination ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 50  # Goal-oriented reward for winning
            elif self.lives <= 0:
                reward -= 50  # Goal-oriented penalty for losing
            self.game_over = True
        
        # --- 4. Return observation and info ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_difficulty(self):
        current_milestone = self.score // 25
        if current_milestone > self.last_score_milestone:
            self.fall_speed += self.FALL_SPEED_INCREMENT
            self.last_score_milestone = current_milestone

    def _spawn_objects(self):
        # Spawn fruit
        if self.np_random.random() < self.FRUIT_SPAWN_CHANCE:
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            self.fruits.append({
                "pos": [self.np_random.integers(20, self.WIDTH - 20), -10],
                "color": self.FRUIT_COLORS[fruit_type],
                "radius": self.np_random.integers(8, 13),
            })
        
        # Spawn bomb
        if self.np_random.random() < self.BOMB_SPAWN_CHANCE:
            self.bombs.append({
                "pos": [self.np_random.integers(20, self.WIDTH - 20), -10],
                "radius": 12,
            })

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos_x - self.PLAYER_WIDTH / 2,
            self.HEIGHT - self.PLAYER_HEIGHT - 10,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )

    def _update_fruits(self):
        reward = 0
        player_rect = self._get_player_rect()
        fruits_to_keep = []
        for fruit in self.fruits:
            fruit["pos"][1] += self.fall_speed
            fruit_rect = pygame.Rect(fruit["pos"][0] - fruit["radius"], fruit["pos"][1] - fruit["radius"], fruit["radius"] * 2, fruit["radius"] * 2)
            
            if player_rect.colliderect(fruit_rect):
                # SFX: Fruit catch
                self.score += 1
                reward += 1
                self._create_particles(fruit["pos"], fruit["color"], 20)
            elif fruit["pos"][1] < self.HEIGHT + 20:
                fruits_to_keep.append(fruit)
        self.fruits = fruits_to_keep
        return reward

    def _update_bombs(self):
        reward = 0
        player_rect = self._get_player_rect()
        bombs_to_keep = []
        for bomb in self.bombs:
            bomb["pos"][1] += self.fall_speed
            bomb_rect = pygame.Rect(bomb["pos"][0] - bomb["radius"], bomb["pos"][1] - bomb["radius"], bomb["radius"] * 2, bomb["radius"] * 2)

            if player_rect.colliderect(bomb_rect):
                # SFX: Bomb explosion
                self.lives -= 1
                reward -= 5
                self._create_particles(bomb["pos"], self.COLOR_BOMB_FLASH, 40, 5)
                self._create_particles(bomb["pos"], self.COLOR_BOMB, 30, 3)
            elif bomb["pos"][1] < self.HEIGHT + 20:
                bombs_to_keep.append(bomb)
        self.bombs = bombs_to_keep
        return reward
    
    def _create_particles(self, pos, color, count, speed_mult=2.5):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * speed_mult + 0.5
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "lifetime": self.np_random.integers(15, 30),
                "radius": self.np_random.random() * 2 + 1
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], [int(p["pos"][0]), int(p["pos"][1])], int(p["radius"]))
            
        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit["pos"][0]), int(fruit["pos"][1]))
            radius = int(fruit["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, fruit["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, fruit["color"])
            
        # Draw bombs
        for bomb in self.bombs:
            pos = (int(bomb["pos"][0]), int(bomb["pos"][1]))
            radius = int(bomb["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            # Flashing core
            flash_alpha = (math.sin(self.steps * 0.4) + 1) / 2
            flash_color = (
                self.COLOR_BOMB[0] + (self.COLOR_BOMB_FLASH[0] - self.COLOR_BOMB[0]) * flash_alpha,
                self.COLOR_BOMB[1] + (self.COLOR_BOMB_FLASH[1] - self.COLOR_BOMB[1]) * flash_alpha,
                self.COLOR_BOMB[2] + (self.COLOR_BOMB_FLASH[2] - self.COLOR_BOMB[2]) * flash_alpha,
            )
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius // 2, flash_color)

        # Draw player basket
        player_rect = self._get_player_rect()
        points = [
            (player_rect.left, player_rect.top),
            (player_rect.right, player_rect.top),
            (player_rect.right - 10, player_rect.bottom),
            (player_rect.left + 10, player_rect.bottom)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_BASKET, points)
        pygame.draw.aalines(self.screen, self.COLOR_BASKET_RIM, True, points, True)

    def _render_ui(self):
        # Draw score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw lives
        for i in range(self.lives):
            self._draw_heart(self.WIDTH - 30 - (i * 35), 25, 15)

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            win_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surface = self.font_large.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)
    
    def _draw_heart(self, x, y, size):
        # Simple heart shape using polygons
        points = [
            (x, y + size // 4),
            (x - size // 2, y - size // 4),
            (x - size // 4, y - size // 2),
            (x, y - size // 4),
            (x + size // 4, y - size // 2),
            (x + size // 2, y - size // 4),
            (x, y + size // 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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
if __name__ == '__main__':
    # Set this to "dummy" to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # `pip install pygame` is required for this part
    
    try:
        pygame.display.set_caption(env.game_description)
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        display_screen = None

    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # --- Action mapping for human play ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        if display_screen:
            # The observation is (H, W, C), but pygame surface wants (W, H)
            # and the array is transposed. We need to get it back.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

        if done:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before closing
            if display_screen:
                pygame.time.wait(2000)

    env.close()