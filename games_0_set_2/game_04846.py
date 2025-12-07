import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the basket. Catch the fruit to score points!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a top-down arcade game to achieve the highest score before missing too many. Chain catches for combo bonuses!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 8
    FRUIT_SIZE = 16
    MAX_LIVES = 5
    WIN_SCORE = 50
    MAX_STEPS = 2000

    # Colors
    COLOR_BG_TOP = (20, 20, 40)
    COLOR_BG_BOTTOM = (40, 40, 80)
    COLOR_BASKET = (139, 69, 19)
    COLOR_BASKET_RIM = (160, 82, 45)
    COLOR_TEXT = (255, 255, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_MISS_MARKER = (255, 0, 0)
    FRUIT_COLORS = {
        "apple": (255, 60, 60),
        "lemon": (255, 255, 80),
        "orange": (255, 165, 0),
        "plum": (140, 80, 180),
        "lime": (50, 205, 50),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Impact", 48)
        self.font_game_over = pygame.font.SysFont("Impact", 64)

        # Initialize state variables
        self.basket_pos_x = 0
        self.fruits = []
        self.particles = []
        self.miss_markers = []
        self.steps = 0
        self.score = 0
        self.fruits_caught_total = 0
        self.lives = 0
        self.game_over = False
        self.combo_count = 0
        self.combo_display_timer = 0
        self.base_fruit_speed = 0.0
        self.np_random = None

        # self.reset() is called by the gym wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.basket_pos_x = self.SCREEN_WIDTH / 2
        self.fruits = []
        self.particles = []
        self.miss_markers = []
        self.steps = 0
        self.score = 0
        self.fruits_caught_total = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.combo_count = 0
        self.combo_display_timer = 0
        self.base_fruit_speed = 2.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1

        # 1. Handle player input
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos_x += self.BASKET_SPEED
        
        self.basket_pos_x = np.clip(
            self.basket_pos_x, self.BASKET_WIDTH / 2, self.SCREEN_WIDTH - self.BASKET_WIDTH / 2
        )

        # 2. Update game state
        # --- Difficulty scaling ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.base_fruit_speed += 0.05

        # --- Fruit spawning ---
        if self.np_random.random() < 0.04:
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            self.fruits.append({
                "x": self.np_random.integers(self.FRUIT_SIZE, self.SCREEN_WIDTH - self.FRUIT_SIZE),
                "y": -self.FRUIT_SIZE,
                "type": fruit_type,
                "speed": self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "rotation_speed": self.np_random.uniform(-0.1, 0.1)
            })

        # --- Fruit movement and collision ---
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.BASKET_WIDTH / 2,
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )

        for fruit in self.fruits[:]:
            fruit["y"] += fruit["speed"]
            fruit["angle"] += fruit["rotation_speed"]
            fruit_rect = pygame.Rect(fruit["x"] - self.FRUIT_SIZE / 2, fruit["y"] - self.FRUIT_SIZE / 2, self.FRUIT_SIZE, self.FRUIT_SIZE)

            # Check for catch
            if basket_rect.colliderect(fruit_rect):
                self.fruits.remove(fruit)
                self.fruits_caught_total += 1
                self.score += 1
                reward += 0.1
                
                self.combo_count += 1
                if self.combo_count > 1:
                    combo_reward = 1.0 if self.combo_count == 2 else 0.5
                    reward += combo_reward
                    self.score += int(combo_reward * 10) # Bonus score
                    self.combo_display_timer = 60 # 2 seconds at 30fps
                    self._create_particles(fruit['x'], fruit['y'], self.FRUIT_COLORS[fruit['type']])
            
            # Check for miss
            elif fruit["y"] > self.SCREEN_HEIGHT:
                self.fruits.remove(fruit)
                self.lives -= 1
                reward -= 0.5
                self.combo_count = 0
                self.miss_markers.append({"pos": (fruit["x"], self.SCREEN_HEIGHT - 10), "life": 45})

        # --- Update effects ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        for m in self.miss_markers[:]:
            m['life'] -= 1
            if m['life'] <= 0:
                self.miss_markers.remove(m)
        
        if self.combo_display_timer > 0:
            self.combo_display_timer -= 1

        # 3. Check for termination
        terminated = self.lives <= 0 or self.fruits_caught_total >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.fruits_caught_total >= self.WIN_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
        
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._draw_background()
        self._draw_miss_markers()
        self._draw_fruits()
        self._draw_basket()
        self._draw_particles()
        self._draw_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fruits_caught": self.fruits_caught_total,
            "combo": self.combo_count
        }

    def _draw_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_miss_markers(self):
        temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for marker in self.miss_markers:
            alpha = int(255 * (marker['life'] / 45))
            color = (*self.COLOR_MISS_MARKER, alpha)
            x, y = int(marker['pos'][0]), int(marker['pos'][1])
            size = 15
            pygame.draw.line(temp_surface, color, (x - size, y - size), (x + size, y + size), 3)
            pygame.draw.line(temp_surface, color, (x + size, y - size), (x - size, y + size), 3)
        self.screen.blit(temp_surface, (0,0))


    def _draw_fruits(self):
        for fruit in self.fruits:
            x, y = int(fruit["x"]), int(fruit["y"])
            color = self.FRUIT_COLORS[fruit["type"]]
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.FRUIT_SIZE // 2, color)
            darker_color = tuple(int(c * 0.8) for c in color)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.FRUIT_SIZE // 2, darker_color)
            
            # Add a small stem/leaf for visual flair
            angle = fruit['angle']
            leaf_x = x + int(math.cos(angle) * (self.FRUIT_SIZE // 2))
            leaf_y = y + int(math.sin(angle) * (self.FRUIT_SIZE // 2))
            pygame.draw.line(self.screen, (34, 139, 34), (x, y), (leaf_x, leaf_y), 2)


    def _draw_basket(self):
        x = int(self.basket_pos_x)
        y = self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 10
        rect = pygame.Rect(x - self.BASKET_WIDTH / 2, y, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, rect, width=3, border_radius=5)

    def _draw_particles(self):
        temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            size = int(p['life'] / 4)
            if size > 1:
                pygame.gfxdraw.filled_circle(temp_surface, pos[0], pos[1], size, color)
        self.screen.blit(temp_surface, (0,0))

    def _draw_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_ui, self.COLOR_TEXT, 20, 15, align="left")

        # Lives
        life_text_surf = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(life_text_surf, (self.SCREEN_WIDTH - 150, 15))
        for i in range(self.MAX_LIVES):
            color = (255, 80, 80) if i < self.lives else (50, 50, 50)
            pos = (self.SCREEN_WIDTH - 80 + i * 20, 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, color)
            darker_color = tuple(int(c * 0.8) for c in color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, darker_color)

        # Combo text
        if self.combo_display_timer > 0 and self.combo_count > 1:
            timer_progress = self.combo_display_timer / 60.0
            alpha = int(min(1.0, timer_progress * 4) * 255)
            scale = 1.0 + (1.0 - timer_progress) * 0.5
            
            text = f"{self.combo_count}x COMBO!"
            font = pygame.font.SysFont("Impact", int(48 * scale))
            
            color = (*self.COLOR_TEXT, alpha)
            
            self._draw_text(text, font, color, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 3, align="center", shadow=False)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.fruits_caught_total >= self.WIN_SCORE else "GAME OVER"
            self._draw_text(msg, self.font_game_over, self.COLOR_TEXT, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, align="center")

    def _draw_text(self, text, font, color, x, y, align="center", shadow=True):
        has_alpha = len(color) == 4
        base_color = color[:3] if has_alpha else color
        
        text_surface = font.render(text, True, base_color)
        if has_alpha:
            alpha = color[3]
            text_surface.set_alpha(alpha)

        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = (int(x), int(y))
        elif align == "left":
            text_rect.midleft = (int(x), int(y))
        elif align == "right":
            text_rect.midright = (int(x), int(y))

        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            if has_alpha:
                shadow_surface.set_alpha(alpha)
            shadow_rect = shadow_surface.get_rect(center=(text_rect.centerx + 2, text_rect.centery + 2))
            self.screen.blit(shadow_surface, shadow_rect)
            
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 25),
                'color': color
            })

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It's useful for testing and debugging.
    # To play, you might need to comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Re-initialize pygame with default video driver for display
    pygame.quit()
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.init()
    pygame.font.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Action defaults
        movement = 0 # No-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key presses for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        action = [movement, 0, 0] # Movement, space (unused), shift (unused)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Cap the frame rate
        clock.tick(30) # Run at 30 FPS

    pygame.quit()