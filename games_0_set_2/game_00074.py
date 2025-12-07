
# Generated: 2025-08-27T12:31:26.833939
# Source Brief: brief_00074.md
# Brief Index: 74

        
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
        "Controls: Use arrow keys to move the blade. Hold spacebar and move to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit with a virtual blade to reach a target score before too many fruits splat on the ground."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for interpolation and game speed
        self.BLADE_SPEED = 15
        self.WIN_SCORE = 25
        self.MAX_MISSES = 5
        self.MAX_STEPS = 1500 # Extended to allow more time to win/lose

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (70, 130, 180) # Steel Blue
        self.COLOR_BLADE = (255, 255, 255)
        self.COLOR_BLADE_OUTLINE = (0, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.FRUIT_COLORS = {
            "apple": (220, 20, 60),    # Crimson Red
            "pear": (50, 205, 50),     # Lime Green
            "lemon": (255, 255, 0),    # Yellow
            "orange": (255, 140, 0),   # Dark Orange
            "plum": (138, 43, 226),    # Blue Violet
        }
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Pre-render background for performance
        self.background = self._create_gradient_background()

        # Initialize state variables
        self.blade_pos = None
        self.last_blade_pos = None
        self.fruits = None
        self.particles = None
        self.swipe_trails = None
        self.steps = None
        self.score = None
        self.fruits_sliced = None
        self.fruits_missed = None
        self.game_over = None
        self.base_fruit_speed = None
        self.fruit_spawn_timer = None
        self.fruit_spawn_interval = None

        self.reset()
        self.validate_implementation()

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color = [
                self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * (y / self.HEIGHT)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.blade_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.last_blade_pos = self.blade_pos.copy()
        
        self.fruits = []
        self.particles = []
        self.swipe_trails = deque(maxlen=10) # Store last 10 trail points for a smooth curve

        self.steps = 0
        self.score = 0
        self.fruits_sliced = 0
        self.fruits_missed = 0
        self.game_over = False
        
        self.base_fruit_speed = 3.0
        self.fruit_spawn_timer = 0
        self.fruit_spawn_interval = 60 # Spawn a fruit every 2 seconds at 30fps

        # Spawn a few initial fruits
        for _ in range(3):
            self._spawn_fruit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is ignored as per brief
        
        self.steps += 1
        reward = -0.01 # Small penalty for time passing to encourage action

        self.last_blade_pos = self.blade_pos.copy()
        self._handle_input(movement)

        if space_held:
            self.swipe_trails.append(self.blade_pos.copy())
            swipe_segment = (self.last_blade_pos, self.blade_pos)
        else:
            self.swipe_trails.clear()
            swipe_segment = None
        
        slice_reward, miss_penalty = self._update_fruits(swipe_segment)
        reward += slice_reward + miss_penalty

        self._update_particles()
        self._spawn_fruit_logic()
        self._update_difficulty()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.blade_pos[1] -= self.BLADE_SPEED
        elif movement == 2: # Down
            self.blade_pos[1] += self.BLADE_SPEED
        elif movement == 3: # Left
            self.blade_pos[0] -= self.BLADE_SPEED
        elif movement == 4: # Right
            self.blade_pos[0] += self.BLADE_SPEED
        
        # Clamp blade position to screen bounds
        self.blade_pos[0] = max(0, min(self.WIDTH, self.blade_pos[0]))
        self.blade_pos[1] = max(0, min(self.HEIGHT, self.blade_pos[1]))

    def _update_fruits(self, swipe_segment):
        slice_reward = 0
        miss_penalty = 0
        
        for fruit in self.fruits[:]:
            # Movement
            fruit['pos'][0] += fruit['vel'][0]
            fruit['pos'][1] += fruit['vel'][1]
            fruit['angle'] += fruit['rot_speed']

            # Slicing
            if swipe_segment and self._line_circle_collision(swipe_segment[0], swipe_segment[1], fruit['pos'], fruit['radius']):
                # SFX: Slice sound
                self.fruits.remove(fruit)
                self.fruits_sliced += 1
                slice_reward += 1
                self.score += 10
                
                # Risky slice bonus
                if fruit['pos'][1] > self.HEIGHT * 0.85:
                    slice_reward += 5
                    self.score += 25

                self._create_juice_splash(fruit['pos'], fruit['color'])
                continue

            # Missed fruit
            if fruit['pos'][1] > self.HEIGHT + fruit['radius']:
                # SFX: Splat sound
                self.fruits.remove(fruit)
                self.fruits_missed += 1
                miss_penalty -= 1
                self._create_splat_effect(fruit['pos'][0], fruit['color'])

        return slice_reward, miss_penalty

    def _create_juice_splash(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': pos[:],
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            }
            self.particles.append(particle)

    def _create_splat_effect(self, x_pos, color):
        splat_color = tuple(max(0, c - 50) for c in color)
        for _ in range(15):
            angle = self.np_random.uniform(math.pi, 2 * math.pi) # Upward splash
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': [x_pos, self.HEIGHT-5],
                'vel': vel,
                'radius': self.np_random.uniform(3, 6),
                'lifespan': self.np_random.integers(20, 40),
                'color': splat_color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
    
    def _spawn_fruit_logic(self):
        self.fruit_spawn_timer += 1
        if self.fruit_spawn_timer >= self.fruit_spawn_interval:
            self.fruit_spawn_timer = 0
            self._spawn_fruit()

    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
        fruit = {
            'pos': [self.np_random.uniform(self.WIDTH * 0.1, self.WIDTH * 0.9), -20],
            'vel': [self.np_random.uniform(-1, 1), self.base_fruit_speed + self.np_random.uniform(-0.5, 0.5)],
            'radius': self.np_random.integers(15, 25),
            'color': self.FRUIT_COLORS[fruit_type],
            'type': fruit_type,
            'angle': 0,
            'rot_speed': self.np_random.uniform(-0.1, 0.1)
        }
        self.fruits.append(fruit)

    def _update_difficulty(self):
        # Increase fruit speed over time
        if self.steps > 0 and self.steps % 250 == 0:
            self.base_fruit_speed += 0.1
        # Decrease spawn interval
        self.fruit_spawn_interval = max(15, 60 - (self.steps // 100))

    def _check_termination(self):
        if self.fruits_sliced >= self.WIN_SCORE:
            return True, 100 # Win
        if self.fruits_missed >= self.MAX_MISSES:
            return True, -100 # Loss
        if self.steps >= self.MAX_STEPS:
            return True, 0 # Timeout
        return False, 0

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            alpha = max(0, min(255, alpha))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render fruits
        for fruit in self.fruits:
            pygame.gfxdraw.aacircle(self.screen, int(fruit['pos'][0]), int(fruit['pos'][1]), fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(fruit['pos'][0]), int(fruit['pos'][1]), fruit['radius'], fruit['color'])

        # Render swipe trail
        if len(self.swipe_trails) > 1:
            points = [tuple(map(int, p)) for p in self.swipe_trails]
            pygame.draw.lines(self.screen, self.COLOR_BLADE, False, points, 8)
            pygame.draw.lines(self.screen, (230, 230, 250), False, points, 4)

        # Render blade cursor
        x, y = int(self.blade_pos[0]), int(self.blade_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_BLADE_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 6, self.COLOR_BLADE)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        shadow_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_text, (12, 12))
        self.screen.blit(score_text, (10, 10))

        # Misses
        for i in range(self.MAX_MISSES):
            x = self.WIDTH - 25 - (i * 30)
            y = 25
            color = (100, 0, 0) if i < self.fruits_missed else (220, 20, 60)
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, (50, 50, 50))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, (50, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_sliced": self.fruits_sliced,
            "fruits_missed": self.fruits_missed,
        }

    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Simple approximation by checking points along the line segment
        num_checks = 10
        for i in range(num_checks + 1):
            t = i / num_checks
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            dist_sq = (px - circle_center[0])**2 + (py - circle_center[1])**2
            if dist_sq < circle_radius**2:
                return True
        return False

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Action mapping for human play ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Unused
        
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
            
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Frame rate ---
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Info: {info}")
    env.close()