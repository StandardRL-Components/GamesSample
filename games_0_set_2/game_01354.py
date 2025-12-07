
# Generated: 2025-08-27T16:51:52.107118
# Source Brief: brief_01354.md
# Brief Index: 1354

        
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
        "Controls: ←→ to move the catcher. Press space to attempt a catch."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch the falling fruit! Move your catcher and time your catches to score points. Miss five fruits and the game is over. Reach 50 catches to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 50
        self.LOSE_MISSES = 5
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 50, 80)
        self.COLOR_CATCHER = (200, 200, 255)
        self.COLOR_CATCHER_ACTION = (100, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SCORE = (255, 220, 0)
        self.COLOR_MISS = (255, 50, 50)
        self.FRUIT_COLORS = [
            (255, 80, 80), (80, 255, 80), (255, 255, 80), 
            (255, 80, 255), (80, 255, 255)
        ]
        
        # Player/Catcher properties
        self.CATCHER_WIDTH = 80
        self.CATCHER_HEIGHT = 20
        self.CATCHER_SPEED = 12

        # Fruit properties
        self.INITIAL_FRUIT_FALL_SPEED = 2.0
        self.FRUIT_SPAWN_PROB = 0.06
        self.DIFFICULTY_INTERVAL = 100
        self.DIFFICULTY_INCREASE = 0.05

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Pre-render background gradient for performance
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self._draw_gradient()
        
        # Initialize state variables
        self.catcher_pos = None
        self.fruits = None
        self.particles = None
        self.steps = None
        self.score = None
        self.missed_fruits = None
        self.fruit_fall_speed = None
        self.is_catching = None
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.catcher_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - self.CATCHER_HEIGHT - 5)
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.fruit_fall_speed = self.INITIAL_FRUIT_FALL_SPEED
        self.is_catching = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        self.is_catching = action[1] == 1
        
        # --- Continuous Reward Calculation (pre-movement) ---
        old_dist_to_fruit = float('inf')
        nearest_fruit = self._get_nearest_fruit()
        if nearest_fruit:
            old_dist_to_fruit = abs(self.catcher_pos.x - nearest_fruit['pos'].x)

        # --- Update Game Logic ---
        # 1. Move Catcher
        if movement == 3:  # Left
            self.catcher_pos.x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_pos.x += self.CATCHER_SPEED
        self.catcher_pos.x = np.clip(self.catcher_pos.x, self.CATCHER_WIDTH / 2, self.WIDTH - self.CATCHER_WIDTH / 2)

        # --- Continuous Reward Calculation (post-movement) ---
        if nearest_fruit:
            new_dist_to_fruit = abs(self.catcher_pos.x - nearest_fruit['pos'].x)
            if new_dist_to_fruit < old_dist_to_fruit:
                reward += 0.1  # Reward for moving closer
            else:
                reward -= 0.1  # Penalty for moving away or staying still when not aligned
        
        # 2. Update Particles
        self._update_particles()
        
        # 3. Update Fruits (Move, Catch, Miss)
        fruits_to_remove = []
        catcher_rect = self._get_catcher_rect()
        
        for fruit in self.fruits:
            fruit['pos'].y += self.fruit_fall_speed
            
            fruit_rect = pygame.Rect(fruit['pos'].x - fruit['size'], fruit['pos'].y - fruit['size'], fruit['size']*2, fruit['size']*2)

            # Check for catch
            if self.is_catching and catcher_rect.colliderect(fruit_rect):
                # sfx: catch_sound.play()
                self.score += 1
                reward += 2.0
                fruits_to_remove.append(fruit)
                self._create_particles(fruit['pos'], self.COLOR_SCORE, 20)
                continue

            # Check for miss
            if fruit['pos'].y > self.HEIGHT + fruit['size']:
                # sfx: miss_sound.play()
                self.missed_fruits += 1
                reward -= 1.0
                fruits_to_remove.append(fruit)
                self._create_particles(pygame.Vector2(fruit['pos'].x, self.HEIGHT - 10), self.COLOR_MISS, 15, is_miss=True)

        # Remove caught/missed fruits
        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        
        # 4. Spawn new fruit
        if self.np_random.random() < self.FRUIT_SPAWN_PROB:
            self._spawn_fruit()

        # 5. Increase difficulty
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fruit_fall_speed += self.DIFFICULTY_INCREASE
            
        # 6. Update step counter
        self.steps += 1
        
        # 7. Check termination conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        elif self.missed_fruits >= self.LOSE_MISSES:
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Base background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed": self.missed_fruits
        }

    def _get_catcher_rect(self):
        return pygame.Rect(
            self.catcher_pos.x - self.CATCHER_WIDTH / 2,
            self.catcher_pos.y - self.CATCHER_HEIGHT / 2,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )

    def _get_nearest_fruit(self):
        if not self.fruits:
            return None
        
        catcher_x = self.catcher_pos.x
        return min(self.fruits, key=lambda f: abs(f['pos'].x - catcher_x))

    def _spawn_fruit(self):
        size = self.np_random.integers(10, 16)
        fruit_type = self.np_random.choice(['circle', 'square', 'triangle'])
        new_fruit = {
            'pos': pygame.Vector2(self.np_random.integers(size, self.WIDTH - size), -size),
            'size': size,
            'color': random.choice(self.FRUIT_COLORS),
            'type': fruit_type,
        }
        self.fruits.append(new_fruit)

    def _create_particles(self, pos, color, count, is_miss=False):
        for _ in range(count):
            if is_miss:
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                speed = self.np_random.uniform(2, 5)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
            
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_gradient(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 3, (*p['color'], alpha))

        # Draw fruits
        for fruit in self.fruits:
            pos_x, pos_y = int(fruit['pos'].x), int(fruit['pos'].y)
            size = fruit['size']
            color = fruit['color']
            outline_color = tuple(max(0, c-50) for c in color)

            if fruit['type'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, size, color)
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, size, color)
            elif fruit['type'] == 'square':
                rect = pygame.Rect(pos_x - size, pos_y - size, size*2, size*2)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                pygame.draw.rect(self.screen, outline_color, rect, width=2, border_radius=3)
            elif fruit['type'] == 'triangle':
                points = [
                    (pos_x, pos_y - size),
                    (pos_x - size, pos_y + size),
                    (pos_x + size, pos_y + size)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw catcher
        catcher_rect = self._get_catcher_rect()
        color = self.COLOR_CATCHER_ACTION if self.is_catching else self.COLOR_CATCHER
        
        # Glow effect when catching
        if self.is_catching:
            glow_rect = catcher_rect.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*color, 50), glow_surf.get_rect(), border_radius=12)
            pygame.draw.rect(glow_surf, (*color, 30), glow_surf.get_rect().inflate(-5,-5), border_radius=8)
            self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, color, catcher_rect, border_radius=5)
        pygame.draw.rect(self.screen, (255,255,255), catcher_rect, width=2, border_radius=5)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Misses display
        miss_text = self.font_small.render("MISSES:", True, self.COLOR_MISS)
        self.screen.blit(miss_text, (self.WIDTH - 150, 15))
        for i in range(self.LOSE_MISSES):
            pos_x = self.WIDTH - 60 + (i * 20)
            pos_y = 25
            color = self.COLOR_MISS if i < self.missed_fruits else (80, 80, 80)
            pygame.draw.line(self.screen, color, (pos_x - 5, pos_y - 5), (pos_x + 5, pos_y + 5), 3)
            pygame.draw.line(self.screen, color, (pos_x + 5, pos_y - 5), (pos_x - 5, pos_y + 5), 3)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            
        clock.tick(30) # Match the intended frame rate
        
    env.close()