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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit for points and combos in a fast-paced arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Colors
        self.COLOR_BG = (15, 23, 42) # Dark Slate Blue
        self.COLOR_BASKET = (255, 255, 255) # White
        self.COLOR_UI_TEXT = (226, 232, 240) # Slate 200
        self.COLOR_MISS_X = (239, 68, 68) # Red 500
        self.FRUIT_TYPES = {
            'apple': {'color': (220, 38, 38), 'value': 10, 'radius': 12},
            'banana': {'color': (250, 204, 21), 'value': 15, 'radius': 13},
            'orange': {'color': (249, 115, 22), 'value': 20, 'radius': 14},
        }
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game parameters
        self.FPS = 30
        self.WIN_SCORE = 500
        self.MAX_MISSES = 5
        self.MAX_STEPS = 1000
        self.BASKET_WIDTH = 100
        self.BASKET_HEIGHT = 20
        self.BASKET_Y_POS = self.HEIGHT - 40
        self.BASKET_SPEED = 12
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.combo = 0
        self.basket_pos_x = 0
        self.fruits = []
        self.particles = []
        self.fruit_fall_speed = 0.0
        self.fruit_spawn_rate = 0.0
        self.spawn_timer = 0.0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.combo = 0
        self.game_over = False
        
        self.basket_pos_x = self.WIDTH / 2 - self.BASKET_WIDTH / 2
        
        self.fruits.clear()
        self.particles.clear()
        
        self.fruit_fall_speed = 2.0
        self.fruit_spawn_rate = 1.0 # Fruits per second
        self.spawn_timer = self.FPS / self.fruit_spawn_rate
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        
        if not self.game_over:
            # --- Update Game Logic ---
            
            # 1. Handle player input
            if movement == 3: # Left
                self.basket_pos_x -= self.BASKET_SPEED
                reward -= 0.01 # Small penalty for movement to encourage efficiency
            elif movement == 4: # Right
                self.basket_pos_x += self.BASKET_SPEED
                reward -= 0.01

            self.basket_pos_x = np.clip(self.basket_pos_x, 0, self.WIDTH - self.BASKET_WIDTH)

            # 2. Update difficulty
            if self.steps > 0 and self.steps % 100 == 0:
                self.fruit_fall_speed += 0.1
                self.fruit_spawn_rate += 0.01
            
            # 3. Spawn new fruits
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_fruit()
                self.spawn_timer = self.FPS / self.fruit_spawn_rate

            # 4. Update fruits and check for catches/misses
            fruits_to_remove = []
            for fruit in self.fruits:
                fruit['pos'][1] += self.fruit_fall_speed
                
                # Check for catch
                basket_rect = pygame.Rect(self.basket_pos_x, self.BASKET_Y_POS, self.BASKET_WIDTH, self.BASKET_HEIGHT)
                fruit_rect = pygame.Rect(fruit['pos'][0] - fruit['radius'], fruit['pos'][1] - fruit['radius'], fruit['radius']*2, fruit['radius']*2)

                if basket_rect.colliderect(fruit_rect):
                    # SFX: Catch sound
                    self.score += fruit['value']
                    reward += 1.0
                    self.combo += 1
                    
                    if self.combo >= 2 and self.combo % 5 == 0:
                        reward += 10.0
                    elif self.combo >= 2 and self.combo % 3 == 0:
                        reward += 5.0

                    self._create_particles(fruit['pos'], self.FRUIT_TYPES[fruit['type']]['color'], 20)
                    fruits_to_remove.append(fruit)
                
                # Check for miss
                elif fruit['pos'][1] > self.HEIGHT:
                    # SFX: Splash/miss sound
                    self.missed_fruits += 1
                    self.combo = 0
                    self._create_particles([fruit['pos'][0], self.HEIGHT - 5], (100, 100, 200), 15, 'splash')
                    fruits_to_remove.append(fruit)

            self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
            
            # 5. Update particles
            self.particles = [p for p in self.particles if p['lifespan'] > 0]
            for p in self.particles:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # Gravity on particles
                p['lifespan'] -= 1
                p['radius'] = max(0, p['radius'] * 0.95)

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.missed_fruits >= self.MAX_MISSES:
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_fruit(self):
        fruit_name = random.choice(list(self.FRUIT_TYPES.keys()))
        fruit_info = self.FRUIT_TYPES[fruit_name]
        
        x_pos = random.uniform(fruit_info['radius'], self.WIDTH - fruit_info['radius'])
        
        self.fruits.append({
            'pos': [x_pos, -fruit_info['radius']],
            'type': fruit_name,
            'radius': fruit_info['radius'],
            'value': fruit_info['value']
        })

    def _create_particles(self, pos, color, count, p_type='catch'):
        for _ in range(count):
            if p_type == 'catch':
                vel = [random.uniform(-2, 2), random.uniform(-4, -1)]
                radius = random.uniform(2, 5)
                lifespan = random.randint(15, 25)
            else: # splash
                angle = random.uniform(math.pi * 1.1, math.pi * 1.9)
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                radius = random.uniform(3, 7)
                lifespan = random.randint(20, 30)

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': radius,
                'lifespan': lifespan,
                'color': color,
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw particles (behind other elements)
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color']
            )

        # Draw basket
        basket_rect = pygame.Rect(self.basket_pos_x, self.BASKET_Y_POS, self.BASKET_WIDTH, self.BASKET_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, (200, 200, 200), basket_rect, width=2, border_radius=5)

        # Draw fruits
        for fruit in self.fruits:
            f_type = self.FRUIT_TYPES[fruit['type']]
            pos_x, pos_y = int(fruit['pos'][0]), int(fruit['pos'][1])
            radius = int(fruit['radius'])
            
            # Fruit body
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, f_type['color'])
            # FIX: The generator expression was passed directly, it must be converted to a tuple.
            outline_color = tuple(max(0, c - 30) for c in f_type['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, outline_color)
            
            # Stem and leaf
            stem_rect = pygame.Rect(pos_x - 2, pos_y - radius - 4, 4, 5)
            pygame.draw.rect(self.screen, (139, 69, 19), stem_rect)
            pygame.draw.polygon(self.screen, (0, 150, 0), [(pos_x+2, pos_y-radius), (pos_x+8, pos_y-radius-2), (pos_x+2, pos_y-radius-4)])

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Misses
        miss_text_surf = self.font_ui.render("Misses:", True, self.COLOR_UI_TEXT)
        self.screen.blit(miss_text_surf, (self.WIDTH - 180, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS_X if i < self.missed_fruits else (80, 80, 80)
            pygame.draw.line(self.screen, color, (self.WIDTH - 40 - i*25, 15), (self.WIDTH - 20 - i*25, 35), 4)
            pygame.draw.line(self.screen, color, (self.WIDTH - 40 - i*25, 35), (self.WIDTH - 20 - i*25, 15), 4)
            
        # Combo
        if self.combo > 1:
            size = min(70, 24 + self.combo * 2)
            dynamic_font = pygame.font.Font(None, size)
            
            # Color shifts from white to yellow to orange with combo
            r = 255
            g = 255
            b = max(0, 255 - self.combo * 10)
            combo_color = (r, g, b)

            combo_surf = dynamic_font.render(f"x{self.combo}", True, combo_color)
            combo_rect = combo_surf.get_rect(center=(self.basket_pos_x + self.BASKET_WIDTH / 2, self.BASKET_Y_POS - 30))
            self.screen.blit(combo_surf, combo_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (134, 239, 172) # Green 300
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISS_X
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.missed_fruits,
            "combo": self.combo,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping for human play
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()