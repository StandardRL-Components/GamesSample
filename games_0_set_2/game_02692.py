
# Generated: 2025-08-27T21:08:38.710906
# Source Brief: brief_02692.md
# Brief Index: 2692

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the basket. Catch fruits, dodge bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Maneuver a basket to catch falling fruits while dodging bombs in a vibrant, top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_FRUIT_COUNT = 50
        self.MAX_BOMB_HITS = 3

        # --- Colors ---
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (240, 230, 140) # Khaki
        self.COLOR_BASKET = (139, 69, 19) # Saddle Brown
        self.COLOR_BASKET_RIM = (160, 82, 45) # Sienna
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_FUSE = (255, 69, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (50, 50, 50)
        self.FRUIT_PROPS = {
            'small': {'color': (255, 0, 0), 'radius': 8, 'value': 1},   # Red Apple
            'medium': {'color': (255, 255, 0), 'radius': 12, 'value': 2}, # Yellow Lemon
            'large': {'color': (0, 128, 0), 'radius': 16, 'value': 3},   # Green Watermelon
        }

        # --- Player ---
        self.PLAYER_WIDTH = 80
        self.PLAYER_HEIGHT = 20
        self.PLAYER_SPEED = 12

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_ui_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_x = 0
        self.player_rect = None
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.fruit_collected_count = 0
        self.bomb_hit_count = 0
        self.base_fall_speed = 2.5
        self.current_fall_speed = 0.0
        self.spawn_prob = 0.03

        # Initialize state for the first time
        self.reset()
        
        # Validate the implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_x = self.SCREEN_WIDTH / 2
        self.player_rect = pygame.Rect(
            self.player_x - self.PLAYER_WIDTH / 2,
            self.SCREEN_HEIGHT - self.PLAYER_HEIGHT - 10,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.fruit_collected_count = 0
        self.bomb_hit_count = 0
        
        self.current_fall_speed = self.base_fall_speed

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement = action[0]
        reward = 0
        terminated = False

        # --- Update Game Logic ---
        self.steps += 1

        # 1. Player Movement
        if movement == 3:  # Left
            self.player_x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_x += self.PLAYER_SPEED
        
        self.player_x = np.clip(self.player_x, self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)
        self.player_rect.centerx = int(self.player_x)

        # 2. Difficulty Scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.current_fall_speed += 0.25
            self.spawn_prob = min(0.1, self.spawn_prob + 0.005)

        # 3. Spawn new items
        if self.np_random.random() < self.spawn_prob:
            if self.np_random.random() < 0.25: # 25% chance for a bomb
                self._spawn_bomb()
            else:
                self._spawn_fruit()

        # 4. Update and check falling items
        # Fruits
        for fruit in self.fruits[:]:
            fruit['pos'][1] += self.current_fall_speed * fruit['speed_mult']
            fruit['rect'].y = int(fruit['pos'][1])
            
            # Collision with basket
            if self.player_rect.colliderect(fruit['rect']):
                reward += fruit['value'] # Event reward
                self.score += fruit['value']
                self.fruit_collected_count += 1
                self._create_particles(fruit['rect'].center, fruit['color'], 20, 'sparkle')
                self.fruits.remove(fruit)
                # SFX: play_collect_sound()
                continue
            
            # Off-screen
            if fruit['rect'].top > self.SCREEN_HEIGHT:
                self.fruits.remove(fruit)
        
        # Bombs
        for bomb in self.bombs[:]:
            bomb['pos'][1] += self.current_fall_speed * bomb['speed_mult']
            bomb['rect'].y = int(bomb['pos'][1])

            # Collision with basket
            if self.player_rect.colliderect(bomb['rect']):
                reward -= 5 # Event reward
                self.bomb_hit_count += 1
                self._create_particles(bomb['rect'].center, self.COLOR_FUSE, 40, 'explosion')
                self.bombs.remove(bomb)
                # SFX: play_explosion_sound()
                continue

            # Off-screen
            if bomb['rect'].top > self.SCREEN_HEIGHT:
                self.bombs.remove(bomb)

        # 5. Continuous Rewards
        for fruit in self.fruits:
            if abs(fruit['rect'].centerx - self.player_rect.centerx) < self.player_rect.width / 2:
                reward += 0.1
        for bomb in self.bombs:
            if abs(bomb['rect'].centerx - self.player_rect.centerx) < self.player_rect.width / 2:
                reward -= 0.05
        
        # 6. Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['type'] == 'explosion':
                p['vel'][1] += 0.1 # Gravity for explosion debris
                p['radius'] = max(0, p['radius'] - 0.1)
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # 7. Check termination conditions
        if self.fruit_collected_count >= self.WIN_FRUIT_COUNT:
            reward += 100
            terminated = True
        elif self.bomb_hit_count >= self.MAX_BOMB_HITS:
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_collected": self.fruit_collected_count,
            "bombs_hit": self.bomb_hit_count,
        }

    # --- Spawning Methods ---
    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(list(self.FRUIT_PROPS.keys()))
        props = self.FRUIT_PROPS[fruit_type]
        x_pos = self.np_random.integers(props['radius'], self.SCREEN_WIDTH - props['radius'])
        speed_mult = self.np_random.uniform(0.8, 1.2)
        
        fruit = {
            'pos': [float(x_pos), float(-props['radius'])],
            'rect': pygame.Rect(x_pos - props['radius'], -props['radius'], props['radius']*2, props['radius']*2),
            'type': fruit_type,
            'color': props['color'],
            'value': props['value'],
            'radius': props['radius'],
            'speed_mult': speed_mult
        }
        self.fruits.append(fruit)

    def _spawn_bomb(self):
        radius = 12
        x_pos = self.np_random.integers(radius, self.SCREEN_WIDTH - radius)
        speed_mult = self.np_random.uniform(0.9, 1.4)
        
        bomb = {
            'pos': [float(x_pos), float(-radius)],
            'rect': pygame.Rect(x_pos - radius, -radius, radius*2, radius*2),
            'radius': radius,
            'speed_mult': speed_mult
        }
        self.bombs.append(bomb)

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'sparkle':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(10, 20)
                radius = self.np_random.uniform(2, 5)
            elif p_type == 'explosion':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 7)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(20, 40)
                radius = self.np_random.uniform(4, 8)
            else:
                continue

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': radius,
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'type': p_type,
            })

    # --- Rendering Methods ---
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate between top and bottom colors
            ratio = y / self.SCREEN_HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Render particles (behind other elements)
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color']
            if p['type'] == 'explosion':
                # Fade to orange/yellow
                r_fade = int(255 * (1 - (p['lifespan'] / p['max_lifespan'])))
                color = (min(255, p['color'][0] + r_fade), max(0, p['color'][1] - r_fade//2), 0)
            
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color + (alpha,))

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            # Fuse
            fuse_pos = (pos[0], pos[1] - bomb['radius'])
            pygame.draw.line(self.screen, self.COLOR_FUSE, fuse_pos, (fuse_pos[0], fuse_pos[1]-5), 2)
            # Spark
            if self.steps % 4 < 2:
                spark_color = (255, 255, 0) if self.steps % 8 < 4 else (255, 165, 0)
                pygame.gfxdraw.filled_circle(self.screen, fuse_pos[0], fuse_pos[1]-5, 2, spark_color)

        # Render player basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.player_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, self.player_rect, 3, border_radius=5)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text_shadow(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_UI_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)
            
        # Score
        score_text = f"Score: {self.score}"
        draw_text_shadow(score_text, self.font_ui, self.COLOR_UI_TEXT, (10, 10))

        # Fruit Count
        fruit_text = f"Fruits: {self.fruit_collected_count} / {self.WIN_FRUIT_COUNT}"
        draw_text_shadow(fruit_text, self.font_ui_small, self.COLOR_UI_TEXT, (10, 45))

        # Bomb Hits
        bomb_text = f"Hits: {self.bomb_hit_count} / {self.MAX_BOMB_HITS}"
        text_width = self.font_ui.render(bomb_text, True, (0,0,0)).get_width()
        draw_text_shadow(bomb_text, self.font_ui, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The brief specifies MultiDiscrete, but for human play we only need movement.
        # We simulate the MultiDiscrete action here.
        action = [movement, 0, 0]

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Fruits: {info['fruits_collected']}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            break

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(env.FPS)

    env.close()