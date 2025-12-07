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
        "Controls: ←→ to move the basket. Catch the falling fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a fast-paced isometric arcade game. "
        "Catch 50 fruits to win, but miss 10 and you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.width, self.height = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        
        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (200, 235, 255)
        self.COLOR_BASKET = (139, 69, 19) # SaddleBrown
        self.COLOR_BASKET_RIM = (160, 82, 45) # Sienna
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_MISS = (255, 0, 0)

        self.FRUIT_TYPES = [
            {"name": "Apple", "color": (255, 60, 60), "radius": 12},
            {"name": "Banana", "color": (255, 255, 100), "radius": 10},
            {"name": "Orange", "color": (255, 165, 0), "radius": 13},
            {"name": "Grape", "color": (140, 40, 160), "radius": 8},
            {"name": "Strawberry", "color": (255, 50, 100), "radius": 9},
            {"name": "Blueberry", "color": (80, 80, 220), "radius": 7},
            {"name": "Watermelon", "color": (60, 200, 60), "radius": 16},
            {"name": "Lemon", "color": (250, 250, 50), "radius": 11},
            {"name": "Cherry", "color": (220, 20, 60), "radius": 6},
            {"name": "Kiwi", "color": (150, 110, 80), "radius": 10},
        ]
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 36)
        except IOError:
            self.font_large = pygame.font.SysFont('sans-serif', 48)
            self.font_small = pygame.font.SysFont('sans-serif', 36)

        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50
        self.LOSE_MISSES = 10
        self.BASKET_WIDTH = 90
        self.BASKET_BOTTOM_WIDTH = 110
        self.BASKET_HEIGHT = 20
        self.BASKET_Y = self.height - 50
        self.BASKET_SPEED = 15 # Increased speed for better feel
        self.INITIAL_FRUIT_SPEED = 2.0
        
        # State variables are initialized in reset()
        self.basket_x = 0
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.game_over = False
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.fruit_spawn_timer = 0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_x = self.width / 2
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.game_over = False
        self.base_fruit_speed = self.INITIAL_FRUIT_SPEED
        self.fruit_spawn_timer = 20
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # When the game is over, we can just return the last observation
            # and signal termination.
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
        
        self.basket_x = np.clip(
            self.basket_x,
            self.BASKET_BOTTOM_WIDTH / 2,
            self.width - self.BASKET_BOTTOM_WIDTH / 2
        )

        # 2. Update game logic
        self.steps += 1
        
        # Update particles
        self._update_particles()
        
        # Spawn new fruits
        self._spawn_fruit()

        # Update fruits (movement, catch, miss)
        reward += self._update_fruits()
        
        # 3. Check for termination
        terminated = (
            self.score >= self.WIN_SCORE or
            self.missed_fruits >= self.LOSE_MISSES
        )
        truncated = self.steps >= self.MAX_STEPS
        
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif self.missed_fruits >= self.LOSE_MISSES:
                reward -= 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            type_index = self.np_random.integers(0, len(self.FRUIT_TYPES))
            fruit_info = self.FRUIT_TYPES[type_index]
            
            x_pos = self.np_random.uniform(
                fruit_info["radius"], self.width - fruit_info["radius"]
            )
            
            new_fruit = {
                "x": x_pos,
                "y": -fruit_info["radius"],
                "type": type_index,
                "speed": self.base_fruit_speed * self.np_random.uniform(0.9, 1.2),
            }
            self.fruits.append(new_fruit)
            self.fruit_spawn_timer = self.np_random.integers(15, 30)

    def _update_fruits(self):
        reward = 0.0
        fruits_to_remove = []
        old_score_tier = self.score // 50

        for fruit in self.fruits:
            fruit["y"] += fruit["speed"]
            
            fruit_info = self.FRUIT_TYPES[fruit["type"]]
            
            # Check for catch
            is_caught = (
                self.BASKET_Y < fruit["y"] + fruit_info["radius"] < self.BASKET_Y + self.BASKET_HEIGHT + fruit_info["radius"] and
                abs(fruit["x"] - self.basket_x) < self.BASKET_WIDTH / 2
            )

            if is_caught:
                self.score += 1
                reward += 0.1
                
                dist_from_center = abs(fruit["x"] - self.basket_x)
                if dist_from_center > self.BASKET_WIDTH * 0.4:
                    reward += 1.0
                
                self._create_splash(fruit["x"], fruit["y"], fruit_info["color"])
                fruits_to_remove.append(fruit)
                continue

            # Check for miss
            if fruit["y"] > self.height + fruit_info["radius"]:
                self.missed_fruits += 1
                reward -= 0.1
                fruits_to_remove.append(fruit)

        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        
        # Update difficulty
        new_score_tier = self.score // 50
        if new_score_tier > old_score_tier:
            self.base_fruit_speed += 0.05 * (new_score_tier - old_score_tier)
            
        return reward

    def _create_splash(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(20, 40),
                "color": color,
            }
            self.particles.append(particle)

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1  # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _render_text(self, text, font, x, y, color, shadow_color=None):
        if shadow_color:
            text_surface = font.render(text, True, shadow_color)
            self.screen.blit(text_surface, (x + 2, y + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.height):
            ratio = y / self.height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, color_with_alpha)

        # Render fruits
        for fruit in self.fruits:
            info = self.FRUIT_TYPES[fruit["type"]]
            x, y, r = int(fruit["x"]), int(fruit["y"]), info["radius"]
            color = info["color"]
            
            # Draw fruit body
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
            # FIX: The generator expression for color was not converted to a tuple, causing an error.
            darker_color = tuple(max(0, c - 30) for c in color)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, darker_color)
            
            # Draw highlight
            highlight_x = x - r // 3
            highlight_y = y - r // 3
            pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, r // 3, (255, 255, 255, 150))
        
        # Render basket
        bx, by = int(self.basket_x), int(self.BASKET_Y)
        top_w, bot_w, h = self.BASKET_WIDTH/2, self.BASKET_BOTTOM_WIDTH/2, self.BASKET_HEIGHT
        points = [
            (bx - top_w, by), (bx + top_w, by),
            (bx + bot_w, by + h), (bx - bot_w, by + h)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASKET)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BASKET_RIM)
        pygame.draw.line(self.screen, self.COLOR_BASKET_RIM, (bx - top_w, by), (bx + top_w, by), 3)

        # Render UI
        self._render_text(f"Score: {self.score}", self.font_small, 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        miss_text = "Misses: " + "X " * self.missed_fruits
        miss_text_width = self.font_small.size(miss_text)[0]
        self._render_text(miss_text, self.font_small, self.width - miss_text_width - 10, 10, self.COLOR_MISS, self.COLOR_TEXT_SHADOW)

        # Render game over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISS
            
            text_w, text_h = self.font_large.size(msg)
            self._render_text(msg, self.font_large, self.width/2 - text_w/2, self.height/2 - text_h/2, color, self.COLOR_TEXT_SHADOW)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
        }
        
    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    # The main loop needs a real display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need to set up a pygame window
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    truncated = False
    running = True
    
    # Simple human player loop
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_q]:
            running = False
        if keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False
            truncated = False

        if not terminated and not truncated:
            action = [movement, 0, 0] # Movement, no space, no shift
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we control the step rate here for human play
        clock.tick(30) 

    env.close()