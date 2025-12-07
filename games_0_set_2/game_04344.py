
# Generated: 2025-08-28T02:07:15.884151
# Source Brief: brief_04344.md
# Brief Index: 4344

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the slicer. Hold space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points, but be careful to avoid the bombs! Reach 5000 points to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 5000
        self.MAX_LIVES = 3
        self.MAX_STEPS = 5000
        self.SLICER_SPEED = 15
        
        # Colors
        self.COLOR_BG_TOP = (10, 10, 20)
        self.COLOR_BG_BOTTOM = (20, 20, 50)
        self.COLOR_SLICER_INACTIVE = (200, 200, 255)
        self.COLOR_SLICER_ACTIVE = (255, 50, 50)
        self.COLOR_SLICER_GLOW = (255, 100, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_SKULL = (220, 220, 220)

        # Fruit definitions
        self.FRUIT_TYPES = [
            {'color': (220, 0, 0), 'points': 100, 'rarity': 0.6, 'name': 'apple'},
            {'color': (50, 200, 50), 'points': 120, 'rarity': 0.25, 'name': 'lime'},
            {'color': (255, 230, 0), 'points': 150, 'rarity': 0.1, 'name': 'lemon'},
            {'color': (255, 140, 0), 'points': 200, 'rarity': 0.04, 'name': 'orange'},
            {'color': (255, 215, 0), 'points': 500, 'rarity': 0.01, 'name': 'golden_apple'}
        ]
        self.fruit_rarity_cdf = np.cumsum([f['rarity'] for f in self.FRUIT_TYPES])

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.slicer_x = 0
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed_multiplier = 1.0
        self.last_slice_active = False
        self.np_random = None

        # Validate implementation after setup
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.slicer_x = self.WIDTH // 2
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.fall_speed_multiplier = 1.0
        self.last_slice_active = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # Unpack factorized action
        movement, space_held, _ = action
        is_slicing = space_held == 1
        
        # 1. Update Slicer Position
        if movement == 3:  # Left
            self.slicer_x -= self.SLICER_SPEED
        elif movement == 4:  # Right
            self.slicer_x += self.SLICER_SPEED
        self.slicer_x = np.clip(self.slicer_x, 0, self.WIDTH)

        # 2. Update and Clean Objects
        self.fruits = [f for f in self.fruits if self._update_object(f)]
        self.bombs = [b for b in self.bombs if self._update_object(b)]
        self.particles = [p for p in self.particles if self._update_particle(p)]

        # 3. Handle Slicing
        if is_slicing:
            # Sliced Fruits
            sliced_this_frame = False
            remaining_fruits = []
            for fruit in self.fruits:
                if abs(fruit['pos'][0] - self.slicer_x) < fruit['size']:
                    self.score += fruit['points']
                    reward += 1.0  # Base reward for any fruit
                    if fruit['name'] == 'golden_apple':
                        reward += 9.0  # Bonus for rare fruit
                    self._create_fruit_particles(fruit)
                    sliced_this_frame = True
                else:
                    remaining_fruits.append(fruit)
            self.fruits = remaining_fruits
            
            # Sliced Bombs
            remaining_bombs = []
            for bomb in self.bombs:
                if abs(bomb['pos'][0] - self.slicer_x) < bomb['size']:
                    self.lives -= 1
                    reward -= 5.0
                    self._create_explosion_particles(bomb)
                    sliced_this_frame = True
                else:
                    remaining_bombs.append(bomb)
            self.bombs = remaining_bombs
            
            if sliced_this_frame:
                # Placeholder for sound effect
                # print("SLICE_SOUND")
                pass

        self.last_slice_active = is_slicing

        # 4. Spawn New Objects
        if self.np_random.random() < 0.05 + (self.steps / 20000):
            if self.np_random.random() < 0.7:
                self._spawn_fruit()
            else:
                self._spawn_bomb()

        # 5. Update Difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed_multiplier = min(3.0, self.fall_speed_multiplier + 0.05)

        # 6. Check Termination
        terminated = False
        win = self.score >= self.WIN_SCORE
        loss = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS

        if win:
            reward += 100.0
            terminated = True
        elif loss:
            reward -= 50.0
            terminated = True
        elif timeout:
            terminated = True
        
        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_object(self, obj):
        obj['pos'][1] += obj['vel'][1] * self.fall_speed_multiplier
        return obj['pos'][1] < self.HEIGHT + obj['size']

    def _update_particle(self, p):
        p['pos'] = [p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1]]
        p['vel'][1] += 0.1 # Gravity
        p['lifespan'] -= 1
        return p['lifespan'] > 0

    def _spawn_fruit(self):
        rand_val = self.np_random.random()
        type_idx = np.searchsorted(self.fruit_rarity_cdf, rand_val)
        fruit_type = self.FRUIT_TYPES[type_idx]
        
        size = self.np_random.integers(15, 30)
        self.fruits.append({
            'pos': [self.np_random.integers(size, self.WIDTH - size), -size],
            'vel': [0, self.np_random.uniform(1.5, 2.5)],
            'size': size,
            **fruit_type
        })

    def _spawn_bomb(self):
        size = 20
        self.bombs.append({
            'pos': [self.np_random.integers(size, self.WIDTH - size), -size],
            'vel': [0, self.np_random.uniform(1.8, 2.8)],
            'size': size
        })
        
    def _create_fruit_particles(self, fruit):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(fruit['pos']),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'color': fruit['color'],
                'size': self.np_random.integers(2, 5),
                'lifespan': self.np_random.integers(20, 40)
            })

    def _create_explosion_particles(self, bomb):
        # Flash
        self.particles.append({
            'pos': list(bomb['pos']), 'vel': [0,0], 'color': (255, 255, 255),
            'size': 100, 'lifespan': 5, 'type': 'flash'
        })
        # Smoke
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 3)
            color_val = self.np_random.integers(50, 150)
            self.particles.append({
                'pos': list(bomb['pos']),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': (color_val, color_val, color_val),
                'size': self.np_random.integers(5, 15),
                'lifespan': self.np_random.integers(30, 60),
                'type': 'smoke'
            })

    def _get_observation(self):
        # 1. Draw Background Gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # 2. Draw Particles
        for p in self.particles:
            if p.get('type') == 'flash':
                alpha = int(255 * (p['lifespan'] / 5))
                flash_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(flash_surf, p['color'] + (alpha,), (p['size'], p['size']), p['size'])
                self.screen.blit(flash_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)
            else:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), p['size'])

        # 3. Draw Game Objects
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['size'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['size'], fruit['color'])
            # Shine effect
            shine_pos = (pos[0] + fruit['size'] // 3, pos[1] - fruit['size'] // 3)
            shine_color = (255, 255, 255, 100)
            pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], fruit['size'] // 4, shine_color)

        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bomb['size'], self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb['size'], self.COLOR_BOMB)
            # Skull
            s = bomb['size']
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (pos[0] - s//3, pos[1] - s//4), s//6)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (pos[0] + s//3, pos[1] - s//4), s//6)
            pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL, (pos[0]-s//8, pos[1]+s//8, s//4, s//3))

        # 4. Draw Slicer
        if self.last_slice_active:
            glow_surf = pygame.Surface((40, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(glow_surf, self.COLOR_SLICER_GLOW, (20, 0), (20, self.HEIGHT), 20)
            self.screen.blit(glow_surf, (int(self.slicer_x) - 20, 0), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.line(self.screen, self.COLOR_SLICER_ACTIVE, (int(self.slicer_x), 0), (int(self.slicer_x), self.HEIGHT), 3)
        else:
            pygame.draw.line(self.screen, self.COLOR_SLICER_INACTIVE, (int(self.slicer_x), 0), (int(self.slicer_x), self.HEIGHT), 2)
        
        # 5. Draw UI
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.lives):
            bomb_pos = (self.WIDTH - 30 - i * 35, 25)
            pygame.gfxdraw.filled_circle(self.screen, bomb_pos[0], bomb_pos[1], 10, self.COLOR_BOMB)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (bomb_pos[0]-3, bomb_pos[1]-2), 2)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (bomb_pos[0]+3, bomb_pos[1]-2), 2)

        # 6. Draw Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
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
        print("Running implementation validation...")
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # For human play
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    
    while not done:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Maintain 30 FPS
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()