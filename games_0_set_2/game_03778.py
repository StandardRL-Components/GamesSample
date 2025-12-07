
# Generated: 2025-08-28T00:23:53.703110
# Source Brief: brief_03778.md
# Brief Index: 3778

        
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
        "Controls: Use arrow keys to move the slicing cursor. Hold Space to slice between your last position and current position."
    )

    game_description = (
        "A fast-paced arcade game. Slice the falling fruit to score points, but be careful to avoid the bombs! Hitting three bombs ends the game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CURSOR_SPEED = 20
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 1000
        self.MAX_BOMBS_HIT = 3
        
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
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_SCORE = (255, 255, 255)
        self.COLOR_SLICE_TRAIL = (255, 255, 255)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_FUSE = (200, 150, 0)
        self.COLOR_BOMB_SPARK = (255, 255, 100)
        
        self.FRUIT_TYPES = {
            "apple": {"color": (255, 50, 50), "radius": 18, "score": 10},
            "orange": {"color": (255, 165, 0), "radius": 20, "score": 10},
            "banana": {"color": (255, 255, 80), "radius": 15, "score": 15}, # Represented as a circle for simplicity
            "kiwi": {"color": (100, 200, 50), "radius": 16, "score": 12},
        }
        
        # Pre-render background gradient
        self.bg_surface = self._create_gradient_background()
        
        # Initialize state variables
        self.cursor_pos = None
        self.prev_cursor_pos = None
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = []
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.fuse_anim_timer = 0
        
        self.base_fruit_spawn_rate = 0.03
        self.base_bomb_spawn_rate = 0.01
        self.base_fruit_speed = 2.0
        
        # This will be properly initialized in reset()
        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.game_over = False
        self.fuse_anim_timer = 0
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.prev_cursor_pos = self.cursor_pos.copy()
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = []
        
        # Reset difficulty
        self.current_fruit_spawn_rate = self.base_fruit_spawn_rate
        self.current_bomb_spawn_rate = self.base_bomb_spawn_rate
        self.current_fruit_speed = self.base_fruit_speed
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        self.prev_cursor_pos = self.cursor_pos.copy()
        
        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_input(movement)
        
        if space_held:
            # Only slice if the cursor has moved
            if np.linalg.norm(self.cursor_pos - self.prev_cursor_pos) > 1.0:
                reward += self._handle_slicing()
                # Add a new slice trail
                self.slice_trails.append({
                    "start": self.prev_cursor_pos.copy(),
                    "end": self.cursor_pos.copy(),
                    "lifetime": 10  # frames
                })

        self._update_game_state()
        self._spawn_entities()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif self.bombs_hit >= self.MAX_BOMBS_HIT:
                reward += -100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
            
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _handle_slicing(self):
        slice_reward = 0.0
        p1 = self.prev_cursor_pos
        p2 = self.cursor_pos
        
        fruits_to_remove = []
        for i, fruit in enumerate(self.fruits):
            if self._line_segment_circle_intersection(p1, p2, fruit['pos'], fruit['radius']):
                fruits_to_remove.append(i)
                # SFX: Fruit slice
                
                # Base reward for slicing
                slice_reward += 1.0
                self.score += fruit['score']
                
                # Check for bonus reward (slicing near a bomb)
                for bomb in self.bombs:
                    if np.linalg.norm(fruit['pos'] - bomb['pos']) < 75:
                        slice_reward += 5.0
                        break # Only one bonus per fruit
                
                self._create_fruit_particles(fruit)

        bombs_to_remove = []
        for i, bomb in enumerate(self.bombs):
            if self._line_segment_circle_intersection(p1, p2, bomb['pos'], bomb['radius']):
                bombs_to_remove.append(i)
                # SFX: Bomb fuse sizzle, then explosion
                self.bombs_hit += 1
                slice_reward -= 5.0
                self._create_bomb_explosion(bomb)

        # Remove sliced items (in reverse to avoid index errors)
        for i in sorted(fruits_to_remove, reverse=True):
            del self.fruits[i]
        for i in sorted(bombs_to_remove, reverse=True):
            del self.bombs[i]
            
        return slice_reward

    def _update_game_state(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_fruit_speed += 0.05
            self.current_fruit_spawn_rate = min(0.1, self.current_fruit_spawn_rate + 0.02)
            self.current_bomb_spawn_rate = min(0.05, self.current_bomb_spawn_rate + 0.01)

        # Update fruits
        for fruit in self.fruits:
            fruit['pos'] += fruit['vel']
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.HEIGHT + f['radius']]
        
        # Update bombs
        for bomb in self.bombs:
            bomb['pos'] += bomb['vel']
        self.bombs = [b for b in self.bombs if b['pos'][1] < self.HEIGHT + b['radius']]
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        
        # Update slice trails
        for s in self.slice_trails:
            s['lifetime'] -= 1
        self.slice_trails = [s for s in self.slice_trails if s['lifetime'] > 0]
        
        # Update bomb fuse animation
        self.fuse_anim_timer = (self.fuse_anim_timer + 1) % 60

    def _spawn_entities(self):
        # Spawn fruits
        if self.np_random.random() < self.current_fruit_spawn_rate:
            fruit_type_name = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
            fruit_type = self.FRUIT_TYPES[fruit_type_name]
            
            x_pos = self.np_random.uniform(fruit_type['radius'], self.WIDTH - fruit_type['radius'])
            y_pos = -fruit_type['radius']
            
            angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6)
            speed = self.current_fruit_speed + self.np_random.uniform(-0.5, 0.5)
            vx = math.cos(angle) * speed * self.np_random.choice([-1, 1]) * 0.5
            vy = math.sin(angle) * speed
            
            self.fruits.append({
                'pos': np.array([x_pos, y_pos], dtype=float),
                'vel': np.array([vx, vy], dtype=float),
                'radius': fruit_type['radius'],
                'color': fruit_type['color'],
                'score': fruit_type['score']
            })
            
        # Spawn bombs
        if self.np_random.random() < self.current_bomb_spawn_rate:
            radius = 22
            x_pos = self.np_random.uniform(radius, self.WIDTH - radius)
            y_pos = -radius
            
            self.bombs.append({
                'pos': np.array([x_pos, y_pos], dtype=float),
                'vel': np.array([0.0, self.current_fruit_speed * 0.8], dtype=float),
                'radius': radius
            })

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.bombs_hit >= self.MAX_BOMBS_HIT or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw slice trails
        for s in self.slice_trails:
            alpha = int(255 * (s['lifetime'] / 10))
            color = (*self.COLOR_SLICE_TRAIL, alpha)
            pygame.draw.line(self.screen, color, s['start'], s['end'], width=max(1, int(s['lifetime']/2)))

        # Draw particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['lifetime'] / p['max_lifetime'])))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw fruits
        for fruit in self.fruits:
            pos = fruit['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            # Add a simple shine effect
            shine_pos = (pos[0] - fruit['radius'] // 2, pos[1] - fruit['radius'] // 2)
            pygame.gfxdraw.aacircle(self.screen, shine_pos[0], shine_pos[1], fruit['radius'] // 4, (255,255,255,100))
            pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], fruit['radius'] // 4, (255,255,255,100))

        # Draw bombs
        for bomb in self.bombs:
            pos = bomb['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            
            # Fuse
            fuse_angle = math.pi * 1.5
            fuse_start = pos + np.array([math.cos(fuse_angle), math.sin(fuse_angle)]) * bomb['radius']
            fuse_end = fuse_start + np.array([0, -10])
            pygame.draw.line(self.screen, self.COLOR_BOMB_FUSE, fuse_start.astype(int), fuse_end.astype(int), 3)

            # Spark
            spark_progress = (self.fuse_anim_timer / 60)
            spark_pos = fuse_end + np.array([0, 8 * (1 - spark_progress)])
            spark_size = int(3 * (1 + math.sin(self.fuse_anim_timer * 0.5)))
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SPARK, spark_pos.astype(int), spark_size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))
        
        # Bomb counter
        for i in range(self.MAX_BOMBS_HIT):
            skull_pos = (self.WIDTH - 40 - i * 45, 20)
            color = (150, 0, 0) if i < self.bombs_hit else (50, 50, 50)
            self._draw_skull(self.screen, skull_pos, color)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_hit": self.bombs_hit,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    # --- Helper Functions ---
    
    def _create_gradient_background(self):
        gradient_surface = pygame.Surface((1, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            gradient_surface.set_at((0, y), color)
        return pygame.transform.scale(gradient_surface, (self.WIDTH, self.HEIGHT))

    def _line_segment_circle_intersection(self, p1, p2, circle_center, radius):
        # Simplified check: Check if endpoints are in circle, or if circle center is near segment
        if np.linalg.norm(p1 - circle_center) < radius or np.linalg.norm(p2 - circle_center) < radius:
            return True
        
        d = p2 - p1
        len_sq = np.dot(d, d)
        if len_sq == 0: return False
        
        t = max(0, min(1, np.dot(circle_center - p1, d) / len_sq))
        closest_point = p1 + t * d
        
        return np.linalg.norm(circle_center - closest_point) < radius

    def _create_fruit_particles(self, fruit):
        # Juice spray
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': fruit['pos'].copy(),
                'vel': vel,
                'color': fruit['color'],
                'size': self.np_random.integers(2, 5),
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })

    def _create_bomb_explosion(self, bomb):
        # Smoke and fire ring
        for i in range(50):
            angle = (i / 50) * 2 * math.pi
            speed = self.np_random.uniform(3, 7)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(20, 40)
            color = self.np_random.choice([(255, 100, 0), (100, 100, 100), (255, 200, 0)])
            self.particles.append({
                'pos': bomb['pos'].copy(),
                'vel': vel,
                'color': color,
                'size': self.np_random.integers(3, 7),
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })

    def _draw_skull(self, surface, pos, color):
        x, y = pos
        # Main head
        pygame.draw.rect(surface, color, (x, y, 30, 25), border_radius=8)
        # Eyes
        pygame.draw.circle(surface, self.COLOR_BG_BOTTOM, (x + 9, y + 10), 5)
        pygame.draw.circle(surface, self.COLOR_BG_BOTTOM, (x + 21, y + 10), 5)
        # Jaw
        pygame.draw.rect(surface, color, (x + 5, y + 20, 20, 10))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import sys
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()
    sys.exit()