
# Generated: 2025-08-28T01:47:44.512219
# Source Brief: brief_04234.md
# Brief Index: 4234

        
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


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the basket. Catch the fruit and avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must catch falling fruits while dodging bombs. "
        "The longer you survive, the faster they fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (135, 206, 250)  # Light Sky Blue
    COLOR_BG_BOTTOM = (70, 130, 180)  # Steel Blue
    COLOR_BASKET = (139, 69, 19)  # Saddle Brown
    COLOR_BOMB = (40, 40, 40)
    COLOR_BOMB_SPIKE = (100, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    FRUIT_COLORS = {
        "apple": (220, 20, 60), # Crimson
        "lemon": (255, 250, 205), # Lemon Chiffon
        "lime": (50, 205, 50), # Lime Green
        "plum": (148, 0, 211) # Dark Violet
    }
    
    # Game Parameters
    MAX_STEPS = 1000
    WIN_SCORE = 25
    MAX_BOMBS_CAUGHT = 3
    
    BASKET_WIDTH = 80
    BASKET_HEIGHT = 20
    BASKET_SPEED = 12
    
    INITIAL_FALL_SPEED = 2.0
    SPEED_INCREASE_INTERVAL = 300 # steps (10 seconds at 30fps)
    SPEED_INCREASE_AMOUNT = 0.2
    
    INITIAL_SPAWN_RATE = 0.1 # items per frame
    SPAWN_RATE_INCREASE_PER_SEC = 0.01
    BOMB_SPAWN_PROB = 0.25 # 25% chance an item is a bomb

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.win = False
        
        self.basket_pos_x = 0
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.fall_speed = 0
        self.spawn_rate = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bombs_caught = 0
        self.game_over = False
        self.win = False
        
        self.basket_pos_x = self.SCREEN_WIDTH // 2
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_rate = self.INITIAL_SPAWN_RATE
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Update Game Logic ---
        self._handle_input(action)
        
        # Update items before calculating rewards
        reward += self._update_items()
        
        # Spawn new items
        self._spawn_items()
        
        # Update particles
        self._update_particles()
        
        # Update difficulty
        self._update_difficulty()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.bombs_caught >= self.MAX_BOMBS_CAUGHT:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.basket_pos_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos_x += self.BASKET_SPEED
            
        # Clamp basket position
        self.basket_pos_x = np.clip(
            self.basket_pos_x,
            self.BASKET_WIDTH // 2,
            self.SCREEN_WIDTH - self.BASKET_WIDTH // 2
        )

    def _update_items(self):
        step_reward = 0
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.BASKET_WIDTH // 2,
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )

        # Update fruits
        for fruit in self.fruits[:]:
            fruit['pos'][1] += self.fall_speed
            if fruit['pos'][1] > self.SCREEN_HEIGHT:
                self.fruits.remove(fruit)
            elif basket_rect.collidepoint(fruit['pos']):
                self.fruits.remove(fruit)
                self.score += 1
                step_reward += 1
                # Sound: CATCH_FRUIT
                self._create_particles(fruit['pos'], fruit['color'], 20)
                
                # Bonus for risky catch
                is_risky = False
                for bomb in self.bombs:
                    dist = math.hypot(self.basket_pos_x - bomb['pos'][0], basket_rect.centery - bomb['pos'][1])
                    if dist < self.BASKET_WIDTH / 2 + bomb['radius'] + 5:
                        is_risky = True
                        break
                if is_risky:
                    step_reward += 5

        # Update bombs
        for bomb in self.bombs[:]:
            bomb['pos'][1] += self.fall_speed
            if bomb['pos'][1] > self.SCREEN_HEIGHT:
                self.bombs.remove(bomb)
            elif basket_rect.collidepoint(bomb['pos']):
                self.bombs.remove(bomb)
                self.bombs_caught += 1
                # Sound: EXPLOSION
                self._create_particles(bomb['pos'], self.COLOR_BOMB_SPIKE, 30, is_explosion=True)

        # Penalty for near miss with a bomb
        for bomb in self.bombs:
            dist = math.hypot(bomb['pos'][0] - self.basket_pos_x, bomb['pos'][1] - basket_rect.centery)
            if self.BASKET_WIDTH / 2 + bomb['radius'] < dist < self.BASKET_WIDTH / 2 + bomb['radius'] + 5:
                step_reward -= 1
                
        return step_reward

    def _spawn_items(self):
        if self.np_random.random() < self.spawn_rate:
            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            
            if self.np_random.random() < self.BOMB_SPAWN_PROB:
                # Spawn bomb
                self.bombs.append({
                    'pos': [x_pos, -20],
                    'radius': 12,
                })
            else:
                # Spawn fruit
                fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
                self.fruits.append({
                    'pos': [x_pos, -20],
                    'radius': self.np_random.integers(10, 15),
                    'type': fruit_type,
                    'color': self.FRUIT_COLORS[fruit_type]
                })

    def _update_difficulty(self):
        self.fall_speed = self.INITIAL_FALL_SPEED + (self.steps // self.SPEED_INCREASE_INTERVAL) * self.SPEED_INCREASE_AMOUNT
        spawn_rate_increase = (self.steps / 30) * self.SPAWN_RATE_INCREASE_PER_SEC
        self.spawn_rate = min(0.5, self.INITIAL_SPAWN_RATE + spawn_rate_increase)

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(20, 40)
                p_color = self.np_random.choice([self.COLOR_BOMB, (255,69,0), (255,165,0)])
            else:
                vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)]
                lifespan = self.np_random.integers(15, 30)
                p_color = color
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': lifespan,
                'color': p_color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self._draw_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw particles (behind other elements)
        for p in self.particles:
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)

        # Draw fruits
        for fruit in self.fruits:
            self._draw_fruit(fruit)
            
        # Draw bombs
        for bomb in self.bombs:
            self._draw_bomb(bomb)
            
        # Draw basket
        basket_rect = pygame.Rect(
            self.basket_pos_x - self.BASKET_WIDTH // 2,
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_BASKET), basket_rect.inflate(-8, -8), border_radius=5)

    def _draw_fruit(self, fruit):
        pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
        radius = fruit['radius']
        color = fruit['color']
        
        # Fruit body with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        
        # Stem
        stem_color = (139, 69, 19)
        pygame.draw.line(self.screen, stem_color, (pos[0], pos[1] - radius), (pos[0], pos[1] - radius - 5), 2)
        
        # Leaf
        leaf_color = (0, 128, 0)
        leaf_points = [(pos[0], pos[1] - radius - 3), (pos[0] + 5, pos[1] - radius - 8), (pos[0] + 2, pos[1] - radius - 2)]
        pygame.draw.polygon(self.screen, leaf_color, leaf_points)

    def _draw_bomb(self, bomb):
        pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
        radius = bomb['radius']
        
        # Spikes
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            spike_pos = (
                int(pos[0] + math.cos(angle) * (radius * 0.8)),
                int(pos[1] + math.sin(angle) * (radius * 0.8))
            )
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SPIKE, spike_pos, radius // 2)

        # Main body with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
        
        # Glint
        glint_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
        pygame.draw.circle(self.screen, (150, 150, 150), glint_pos, radius // 4)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, self.font_small, (10, 10))
        
        # Bombs caught
        bombs_text = f"Bombs: {self.bombs_caught} / {self.MAX_BOMBS_CAUGHT}"
        text_width = self.font_small.size(bombs_text)[0]
        self._draw_text(bombs_text, self.font_small, (self.SCREEN_WIDTH - text_width - 10, 10))

        # Game Over message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text_width, text_height = self.font_large.size(message)
            pos = ((self.SCREEN_WIDTH - text_width) // 2, (self.SCREEN_HEIGHT - text_height) // 2)
            self._draw_text(message, self.font_large, pos, color)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        text_surf_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surf_shadow, shadow_pos)
        
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_caught": self.bombs_caught,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")