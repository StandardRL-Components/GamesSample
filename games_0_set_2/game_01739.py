
# Generated: 2025-08-28T02:33:38.976082
# Source Brief: brief_01739.md
# Brief Index: 1739

        
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

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the target cursor. "
        "Position the cursor over the correct bin for the falling fruit."
    )

    game_description = (
        "A fast-paced sorting game. Catch falling fruits by positioning your cursor "
        "over the matching colored bin before the fruit reaches the sorting line. "
        "The fruits fall faster as you succeed!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CURSOR_SPEED = 15
        self.MAX_STEPS = 1500  # Increased to allow for more time
        self.WIN_SCORE = 15
        self.MAX_MISSES = 3
        self.INITIAL_FALL_SPEED = 2.0
        self.SPEED_INCREMENT = 0.75
        self.FRUITS_PER_LEVEL = 5 # Speed increases more frequently for better pacing

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.FRUIT_COLORS = {
            "apple": (220, 50, 50),
            "lemon": (250, 230, 80),
            "grape": (150, 80, 220)
        }
        self.BIN_COLORS = {
            "apple": (100, 20, 20),
            "lemon": (120, 110, 40),
            "grape": (70, 40, 100)
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.incorrect_sorts = None
        self.fruits_sorted_correctly = None
        self.total_fruits_sorted = None
        self.fruit_fall_speed = None
        self.cursor = None
        self.fruits = None
        self.particles = None
        self.flash_effect = None
        self.bins = None
        self.sort_line_y = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.incorrect_sorts = 0
        self.fruits_sorted_correctly = 0
        self.total_fruits_sorted = 0
        self.fruit_fall_speed = self.INITIAL_FALL_SPEED
        self.cursor = pygame.Rect(self.WIDTH // 2 - 15, self.HEIGHT - 150, 30, 30)
        self.fruits = []
        self.particles = []
        self.flash_effect = 0

        self.fruit_types = list(self.FRUIT_COLORS.keys())
        bin_width = 120
        bin_height = 80
        bin_gap = (self.WIDTH - len(self.fruit_types) * bin_width) / (len(self.fruit_types) + 1)
        self.bins = []
        for i, fruit_type in enumerate(self.fruit_types):
            x = bin_gap * (i + 1) + bin_width * i
            y = self.HEIGHT - bin_height
            self.bins.append({
                "rect": pygame.Rect(x, y, bin_width, bin_height),
                "type": fruit_type
            })

        self.sort_line_y = self.HEIGHT - bin_height - 30
        self._spawn_fruit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action  # space and shift are ignored

        continuous_reward = 0
        if self.fruits:
            current_fruit = self.fruits[0]
            target_bin = next(b for b in self.bins if b["type"] == current_fruit["type"])
            target_x = target_bin["rect"].centerx
            dist_before = abs(self.cursor.centerx - target_x)
        
        self._handle_input(movement)

        if self.fruits:
            dist_after = abs(self.cursor.centerx - target_x)
            if dist_after < dist_before:
                continuous_reward = 0.1
            elif dist_after > dist_before:
                continuous_reward = -0.1

        event_reward = self._update_game_state()
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.fruits_sorted_correctly >= self.WIN_SCORE:
                event_reward += 10
            else:
                event_reward -= 10
        
        reward = event_reward + continuous_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: self.cursor.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor.x += self.CURSOR_SPEED
        self.cursor.clamp_ip(self.screen.get_rect())

    def _update_game_state(self):
        event_reward = 0
        if self.game_over: return 0

        if self.flash_effect > 0: self.flash_effect -= 1
        
        self._update_particles()
        
        fruits_to_remove = []
        for fruit in self.fruits:
            fruit["pos"].y += self.fruit_fall_speed
            
            if fruit["pos"].y > self.sort_line_y and fruit not in fruits_to_remove:
                event_reward += self._sort_fruit(fruit)
                fruits_to_remove.append(fruit)
            elif fruit["pos"].y > self.HEIGHT and fruit not in fruits_to_remove:
                event_reward += self._miss_fruit()
                fruits_to_remove.append(fruit)

        for fruit in fruits_to_remove:
            self.fruits.remove(fruit)

        if not self.fruits and not self._check_termination():
            self._spawn_fruit()
            
        return event_reward

    def _sort_fruit(self, fruit):
        target_bin = None
        for bin_item in self.bins:
            if bin_item["rect"].colliderect(self.cursor):
                target_bin = bin_item
                break

        if target_bin and target_bin["type"] == fruit["type"]:
            self.score += 1
            self.fruits_sorted_correctly += 1
            self.total_fruits_sorted += 1
            self._create_particles(self.cursor.center, self.FRUIT_COLORS[fruit["type"]])
            if self.total_fruits_sorted % self.FRUITS_PER_LEVEL == 0:
                self.fruit_fall_speed += self.SPEED_INCREMENT
            return 1
        else:
            return self._miss_fruit()

    def _miss_fruit(self):
        self.score -= 1
        self.incorrect_sorts += 1
        self.flash_effect = 10
        return -1

    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(self.fruit_types)
        x = self.np_random.integers(50, self.WIDTH - 50)
        self.fruits.append({
            "pos": pygame.Vector2(x, -20),
            "type": fruit_type,
            "radius": 15
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over: return True
        if self.incorrect_sorts >= self.MAX_MISSES or \
           self.fruits_sorted_correctly >= self.WIN_SCORE or \
           self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for bin_item in self.bins:
            pygame.draw.rect(self.screen, self.BIN_COLORS[bin_item["type"]], bin_item["rect"], border_radius=10)
            pygame.draw.rect(self.screen, self.FRUIT_COLORS[bin_item["type"]], bin_item["rect"], 3, border_radius=10)
            self._draw_fruit(pygame.Vector2(bin_item["rect"].centerx, bin_item["rect"].centery + 10),
                             bin_item["type"], radius=12, alpha=100)

        sort_line_color = (100, 100, 100)
        dash_length = 10
        for x in range(0, self.WIDTH, dash_length * 2):
            pygame.draw.line(self.screen, sort_line_color, (x, self.sort_line_y), (x + dash_length, self.sort_line_y), 1)

        for fruit in self.fruits:
            self._draw_fruit(fruit["pos"], fruit["type"], fruit["radius"])

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        pygame.draw.rect(self.screen, (0,0,0,50), self.cursor.inflate(4,4), border_radius=5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor.left, self.cursor.centery), (self.cursor.right, self.cursor.centery), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor.centerx, self.cursor.top), (self.cursor.centerx, self.cursor.bottom), 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, self.cursor, 1, border_radius=5)

        if self.flash_effect > 0:
            alpha = int(120 * (self.flash_effect / 10))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 200, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        score_str = f"SCORE: {self.score}"
        shadow = self.font.render(score_str, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow, (12, 12))
        text = self.font.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(text, (10, 10))
        
        miss_text = self.small_font.render("MISSES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.WIDTH - 160, 15))
        for i in range(self.MAX_MISSES):
            pos = (self.WIDTH - 70 + i * 25, 25)
            if i < self.incorrect_sorts:
                pygame.draw.line(self.screen, (255, 80, 80), (pos[0]-8, pos[1]-8), (pos[0]+8, pos[1]+8), 4)
                pygame.draw.line(self.screen, (255, 80, 80), (pos[0]+8, pos[1]-8), (pos[0]-8, pos[1]+8), 4)
            else:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 8, (100, 100, 100))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            if self.fruits_sorted_correctly >= self.WIN_SCORE:
                end_text_str, color = "YOU WIN!", (100, 255, 100)
            else:
                end_text_str, color = "GAME OVER", (255, 100, 100)
            
            shadow = self.font.render(end_text_str, True, self.COLOR_TEXT_SHADOW)
            text_rect = shadow.get_rect(center=(self.WIDTH/2 + 2, self.HEIGHT/2 + 2))
            self.screen.blit(shadow, text_rect)
            
            text = self.font.render(end_text_str, True, color)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _draw_fruit(self, pos, fruit_type, radius, alpha=255):
        color = self.FRUIT_COLORS[fruit_type]
        surf = self.screen if alpha == 255 else pygame.Surface((radius*2+4, radius*2+4), pygame.SRCALPHA)
        temp_color = (*color, alpha) if alpha < 255 else color
        center_offset = 2 if alpha < 255 else 0
        
        px, py = int(pos.x), int(pos.y)
        cx, cy = radius + center_offset, radius + center_offset

        if fruit_type == "apple":
            pygame.gfxdraw.filled_circle(surf, cx, cy, radius, temp_color)
            pygame.gfxdraw.aacircle(surf, cx, cy, radius, temp_color)
        elif fruit_type == "lemon":
            pygame.gfxdraw.filled_ellipse(surf, cx, cy, int(radius * 0.8), radius, temp_color)
            pygame.gfxdraw.aaellipse(surf, cx, cy, int(radius * 0.8), radius, temp_color)
        elif fruit_type == "grape":
            r_small = int(radius * 0.6)
            offsets = [(-0.4, -0.3), (0.4, -0.3), (0, 0.3), (-0.8, 0.4), (0.8, 0.4), (0.2, 0.8)]
            for ox, oy in offsets:
                scx = cx + int(ox * radius)
                scy = cy + int(oy * radius)
                pygame.gfxdraw.filled_circle(surf, scx, scy, r_small, temp_color)
                pygame.gfxdraw.aacircle(surf, scx, scy, r_small, temp_color)

        if alpha < 255:
            self.screen.blit(surf, (px - radius - center_offset, py - radius - center_offset))
        else:
            # Need to shift the drawing position if drawing directly on screen
            surf.blit(surf, (px - radius, py - radius), area=pygame.Rect(0,0,radius*2,radius*2))


    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel,
                'life': self.np_random.integers(15, 30), 'max_life': 30,
                'color': color, 'size': self.np_random.integers(2, 5)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.incorrect_sorts}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Simple interactive loop
    pygame.display.set_caption("Fruit Sorter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and shift not used

        obs, reward, terminated, truncated, info = env.step(action)

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Match the intended FPS

    print(f"Game Over. Final Info: {info}")
    env.close()