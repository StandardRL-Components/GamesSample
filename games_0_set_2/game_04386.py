
# Generated: 2025-08-28T02:14:12.180748
# Source Brief: brief_04386.md
# Brief Index: 4386

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points, but be careful to avoid the bombs!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Colors and Constants ---
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (144, 238, 144)  # Light Green
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0, 128)
    COLOR_BOMB = (30, 30, 30)
    COLOR_BOMB_FUSE = (255, 165, 0)
    COLOR_APPLE = (220, 20, 60)
    COLOR_BANANA = (255, 223, 0)
    COLOR_ORANGE = (255, 140, 0)
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_SCORE = 500
    MAX_BOMBS_SLICED = 3
    
    CURSOR_SPEED = 10
    SLICE_RADIUS = 30
    BOMB_PROXIMITY_BONUS_RADIUS = 75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bombs_sliced = 0
        self.cursor_pos = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.slice_effects = None
        self.spawn_prob = None
        self.object_speed_multiplier = None
        self.np_random = None

        # Create a static background surface for performance
        self.background = self._create_background()
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bombs_sliced = 0
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_effects = []
        
        self.spawn_prob = 0.03  # Initial probability of spawning an object per frame
        self.object_speed_multiplier = 1.0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- 1. Handle Actions ---
        movement, space_held, _ = action
        
        # Movement
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        else: # No-op
            reward -= 0.01 # Small penalty for inactivity

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        # Slicing
        if space_held:
            # SFX: whoosh.wav
            self._handle_slice(self.cursor_pos, reward)
            self.slice_effects.append({'pos': self.cursor_pos.copy(), 'timer': 10, 'radius': self.SLICE_RADIUS})

        # --- 2. Update Game State ---
        self._update_objects()
        self._update_particles()
        self._update_slice_effects()
        
        # --- 3. Spawn New Objects ---
        self._spawn_objects()
        
        # --- 4. Difficulty Scaling ---
        self.spawn_prob += (0.001 / self.FPS) # Increase spawn rate by 0.001 per second
        if self.steps > 0 and self.steps % 1000 == 0:
            self.object_speed_multiplier += 0.1
            
        # --- 5. Calculate Rewards & Termination ---
        current_reward, missed_fruits = self._calculate_reward()
        reward += current_reward
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.SCREEN_HEIGHT + f['radius']]
        reward -= 1 * missed_fruits
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
            if self.bombs_sliced >= self.MAX_BOMBS_SLICED:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_slice(self, pos, reward_ref):
        sliced_something = False
        
        # Check for bomb slices first
        for bomb in self.bombs[:]:
            dist = math.hypot(pos[0] - bomb['pos'][0], pos[1] - bomb['pos'][1])
            if dist < self.SLICE_RADIUS + bomb['radius']:
                self.bombs.remove(bomb)
                self.bombs_sliced += 1
                self._create_explosion(bomb['pos'])
                # SFX: explosion.wav
                sliced_something = True

        # Check for fruit slices
        for fruit in self.fruits[:]:
            dist = math.hypot(pos[0] - fruit['pos'][0], pos[1] - fruit['pos'][1])
            if dist < self.SLICE_RADIUS + fruit['radius']:
                self.fruits.remove(fruit)
                self.score += fruit['points']
                self._create_juice_splash(fruit['pos'], fruit['color'])
                # SFX: slice.wav
                reward_ref += 1
                
                # Bonus for slicing near a bomb
                for bomb in self.bombs:
                    if math.hypot(fruit['pos'][0] - bomb['pos'][0], fruit['pos'][1] - bomb['pos'][1]) < self.BOMB_PROXIMITY_BONUS_RADIUS:
                        reward_ref += 5
                        break
                sliced_something = True
                
    def _update_objects(self):
        for fruit in self.fruits:
            fruit['pos'][1] += fruit['vel'] * self.object_speed_multiplier
            fruit['angle'] += fruit['rot_speed']
        
        for bomb in self.bombs:
            bomb['pos'][1] += bomb['vel'] * self.object_speed_multiplier
            bomb['fuse_timer'] -= 1
            if bomb['fuse_timer'] <= 0:
                # SFX: fizz.wav
                fuse_end_pos = bomb['pos'] + np.array([0, -bomb['radius']])
                spark_vel = self.np_random.uniform(-1, 1, 2) * 2
                self.particles.append({'pos': fuse_end_pos, 'vel': spark_vel, 'color': self.COLOR_BOMB_FUSE, 'timer': 10, 'size': 3})
                bomb['fuse_timer'] = self.np_random.integers(5, 15)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['timer'] -= 1
            if p['timer'] <= 0:
                self.particles.remove(p)

    def _update_slice_effects(self):
        for effect in self.slice_effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.slice_effects.remove(effect)

    def _spawn_objects(self):
        if self.np_random.random() < self.spawn_prob:
            x_pos = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            base_speed = self.np_random.uniform(2, 4)
            
            if self.np_random.random() < 0.25: # 25% chance of bomb
                self.bombs.append({
                    'pos': np.array([x_pos, -20], dtype=float),
                    'vel': base_speed,
                    'radius': 20,
                    'fuse_timer': self.np_random.integers(5, 15),
                })
            else: # 75% chance of fruit
                fruit_type = self.np_random.choice(['apple', 'banana', 'orange'])
                if fruit_type == 'apple':
                    self.fruits.append({
                        'type': 'apple', 'color': self.COLOR_APPLE, 'points': 10,
                        'radius': 20, 'pos': np.array([x_pos, -20], dtype=float),
                        'vel': base_speed, 'angle': 0, 'rot_speed': self.np_random.uniform(-0.1, 0.1)
                    })
                elif fruit_type == 'banana':
                     self.fruits.append({
                        'type': 'banana', 'color': self.COLOR_BANANA, 'points': 15,
                        'radius': 25, 'pos': np.array([x_pos, -25], dtype=float),
                        'vel': base_speed * 0.9, 'angle': self.np_random.uniform(0, 2*math.pi), 'rot_speed': self.np_random.uniform(-0.1, 0.1)
                    })
                elif fruit_type == 'orange':
                     self.fruits.append({
                        'type': 'orange', 'color': self.COLOR_ORANGE, 'points': 20,
                        'radius': 22, 'pos': np.array([x_pos, -22], dtype=float),
                        'vel': base_speed * 1.1, 'angle': 0, 'rot_speed': self.np_random.uniform(-0.1, 0.1)
                    })
    
    def _create_juice_splash(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'color': color, 'timer': self.np_random.integers(15, 30), 'size': self.np_random.integers(2, 5)})
    
    def _create_explosion(self, pos):
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(3, 8)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            color = random.choice([(255, 0, 0), (255, 165, 0), (255, 69, 0)])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'color': color, 'timer': self.np_random.integers(20, 40), 'size': self.np_random.integers(2, 6)})
        self.slice_effects.append({'pos': pos.copy(), 'timer': 15, 'radius': 60, 'color': (255, 0, 0, 100)})

    def _calculate_reward(self):
        reward = 0
        missed_fruits = 0
        for fruit in self.fruits:
            if fruit['pos'][1] > self.SCREEN_HEIGHT + fruit['radius']:
                missed_fruits += 1
        return reward, missed_fruits

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.bombs_sliced >= self.MAX_BOMBS_SLICED
            or self.steps >= self.MAX_STEPS
        )
    
    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_sliced": self.bombs_sliced,
        }

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        # Render slice effects
        for effect in self.slice_effects:
            alpha = int(255 * (effect['timer'] / (15 if 'color' in effect else 10)))
            color = effect.get('color', (255, 255, 255, alpha))
            if len(color) == 3: color = (*color, alpha)
            s = pygame.Surface((effect['radius']*2, effect['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (effect['radius'], effect['radius']), effect['radius'])
            self.screen.blit(s, (int(effect['pos'][0] - effect['radius']), int(effect['pos'][1] - effect['radius'])))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['timer'] / 30))
            color = (*p['color'], alpha)
            r = int(p['size'] * (p['timer'] / 30))
            if r > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), r, color)

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            if fruit['type'] == 'apple':
                pygame.draw.circle(self.screen, fruit['color'], pos, fruit['radius'])
            elif fruit['type'] == 'orange':
                pygame.draw.circle(self.screen, fruit['color'], pos, fruit['radius'])
            elif fruit['type'] == 'banana':
                rect = pygame.Rect(pos[0] - fruit['radius'], pos[1] - fruit['radius'], fruit['radius']*2, fruit['radius']*2)
                pygame.draw.arc(self.screen, fruit['color'], rect, math.pi/4 + fruit['angle'], 3*math.pi/4 + fruit['angle'], 8)
                pygame.draw.arc(self.screen, fruit['color'], rect, 5*math.pi/4 + fruit['angle'], 7*math.pi/4 + fruit['angle'], 8)

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_BOMB, pos, bomb['radius'])
            pygame.draw.circle(self.screen, (100,100,100), pos, bomb['radius'], 1)
            # Fuse
            fuse_start = (pos[0], pos[1] - bomb['radius'])
            fuse_end = (pos[0] + 3, pos[1] - bomb['radius'] - 5)
            pygame.draw.line(self.screen, (139, 69, 19), fuse_start, fuse_end, 3)
            pygame.gfxdraw.filled_circle(self.screen, fuse_end[0], fuse_end[1], 2, self.COLOR_BOMB_FUSE)

    def _render_ui(self):
        # Render score with a shadow
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        shadow_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_text, (12, 12))
        self.screen.blit(score_text, (10, 10))

        # Render bomb counter
        for i in range(self.MAX_BOMBS_SLICED):
            pos = (self.SCREEN_WIDTH - 40 - i * 40, 15)
            color = self.COLOR_BOMB if i < self.bombs_sliced else (100, 100, 100, 128)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, color)
            if i >= self.bombs_sliced:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, color)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # shift is not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    pygame.quit()