
# Generated: 2025-08-27T22:24:00.748967
# Source Brief: brief_03112.md
# Brief Index: 3112

        
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
        "Controls: Use the Left and Right arrow keys to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit while dodging bombs in this fast-paced, top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 50
        self.MAX_LIVES = 3
        self.MAX_STEPS = 10000

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
        self.font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 28)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_BASKET = (160, 82, 45)
        self.COLOR_BASKET_BORDER = (139, 69, 19)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_HIGHLIGHT = (60, 60, 60)
        self.COLOR_FUSE = (200, 200, 200)
        self.COLOR_SPARK = (255, 220, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.FRUIT_COLORS = {
            "apple": (220, 50, 50),
            "orange": (255, 165, 0),
            "pear": (154, 205, 50),
        }
        self.FRUIT_TYPES = list(self.FRUIT_COLORS.keys())
        
        # Game Parameters
        self.BASKET_WIDTH = 80
        self.BASKET_HEIGHT = 20
        self.BASKET_SPEED = 12
        self.INITIAL_FALL_SPEED = 2.0
        self.SPEED_INCREASE_INTERVAL = 10
        self.SPEED_INCREASE_AMOUNT = 0.25
        self.OBJECT_SPAWN_RATE = 0.04
        self.BOMB_PROBABILITY = 0.25

        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.basket_rect = None
        self.fall_speed = 0.0
        self.objects = []
        self.particles = []
        self.screen_shake = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        
        self.basket_rect = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) / 2,
            self.HEIGHT - self.BASKET_HEIGHT - 10,
            self.BASKET_WIDTH,
            self.BASKET_HEIGHT,
        )
        self.objects = []
        self.particles = []
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_objects()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._spawn_objects()
            self._update_particles()
            self._update_difficulty()

            if self.screen_shake > 0:
                self.screen_shake -= 1

            self.steps += 1
            
            if self.lives <= 0:
                self.game_over = True
                terminated = True
                reward += -100  # Terminal penalty
                # sound: game_over
            elif self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100  # Terminal reward
                # sound: game_win
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
            self.basket_rect.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_rect.x += self.BASKET_SPEED
        
        self.basket_rect.x = np.clip(self.basket_rect.x, 0, self.WIDTH - self.BASKET_WIDTH)

    def _update_objects(self):
        for obj in self.objects:
            obj['rect'].y += self.fall_speed
            if obj['type'] == 'bomb':
                obj['fuse_timer'] = (obj['fuse_timer'] + 1) % 60
        self.objects = [obj for obj in self.objects if obj['rect'].top < self.HEIGHT]

    def _handle_collisions(self):
        reward = 0
        collided_indices = []
        for i, obj in enumerate(self.objects):
            if self.basket_rect.colliderect(obj['rect']):
                collided_indices.append(i)
                if obj['type'] == 'bomb':
                    # sound: explosion
                    self.lives -= 1
                    reward += -5
                    self._create_particles(obj['rect'].center, (100, 100, 100), 50, 'explosion')
                    self.screen_shake = 15
                else:  # Fruit
                    # sound: catch_fruit
                    self.score += 1
                    reward += 1
                    self._create_particles(obj['rect'].center, self.FRUIT_COLORS[obj['type']], 25, 'catch')
        
        # Remove collided objects by creating a new list
        self.objects = [obj for i, obj in enumerate(self.objects) if i not in collided_indices]
        return reward

    def _spawn_objects(self):
        if self.np_random.random() < self.OBJECT_SPAWN_RATE:
            x = self.np_random.integers(20, self.WIDTH - 20)
            is_bomb = self.np_random.random() < self.BOMB_PROBABILITY
            
            obj_rect = pygame.Rect(x, -30, 24, 24)
            # Prevent overlaps when spawning
            if any(obj_rect.colliderect(existing['rect']) for existing in self.objects if existing['rect'].y < 50):
                return

            if is_bomb:
                self.objects.append({'type': 'bomb', 'rect': obj_rect, 'fuse_timer': 0})
            else:
                fruit_type = self.np_random.choice(self.FRUIT_TYPES)
                self.objects.append({'type': fruit_type, 'rect': obj_rect})

    def _update_difficulty(self):
        # Deterministically calculate fall speed based on score
        speed_level = self.score // self.SPEED_INCREASE_INTERVAL
        self.fall_speed = self.INITIAL_FALL_SPEED + speed_level * self.SPEED_INCREASE_AMOUNT

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.97)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'catch':
                angle = self.np_random.uniform(-math.pi / 2 - 0.5, -math.pi / 2 + 0.5)
                speed = self.np_random.uniform(2, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                radius = self.np_random.uniform(2, 5)
                life = 30
            elif p_type == 'explosion':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                radius = self.np_random.uniform(1, 4)
                life = 50
                color = self.np_random.choice([(255,165,0), (100,100,100), (255, 69, 0), self.COLOR_SPARK])
            
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'color': color, 'life': life})

    def _get_observation(self):
        render_offset = (0, 0)
        if self.screen_shake > 0:
            render_offset = (self.np_random.integers(-6, 7), self.np_random.integers(-6, 7))

        self.screen.fill(self.COLOR_BG)
        
        self._render_particles(render_offset)
        self._render_game(render_offset)
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Render objects
        for obj in self.objects:
            r = obj['rect'].move(offset)
            if obj['type'] == 'bomb':
                self._draw_bomb(r, obj['fuse_timer'])
            else:
                self._draw_fruit(r, obj['type'])
        
        # Render basket
        r = self.basket_rect.move(offset)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, r, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_BORDER, r, width=3, border_radius=5)

    def _render_particles(self, offset):
        ox, oy = offset
        for p in self.particles:
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _draw_fruit(self, rect, fruit_type):
        color = self.FRUIT_COLORS[fruit_type]
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, rect.width // 2, color)
        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, rect.width // 2, color)
        # Add a little shine
        shine_pos = (rect.centerx + 5, rect.centery - 5)
        pygame.gfxdraw.aacircle(self.screen, shine_pos[0], shine_pos[1], 3, (255, 255, 255, 100))
        pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], 3, (255, 255, 255, 100))

    def _draw_bomb(self, rect, fuse_timer):
        # Bomb body
        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, rect.width // 2, self.COLOR_BOMB)
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, rect.width // 2, self.COLOR_BOMB_HIGHLIGHT)
        # Fuse
        fuse_end = (rect.centerx + 6, rect.top + 3)
        pygame.draw.line(self.screen, self.COLOR_FUSE, (rect.centerx, rect.top + 8), fuse_end, 3)
        # Animated Spark
        if (fuse_timer // 8) % 2 == 0:
            spark_pos = (fuse_end[0] + 1, fuse_end[1] - 1)
            pygame.gfxdraw.filled_circle(self.screen, spark_pos[0], spark_pos[1], 3, self.COLOR_SPARK)

    def _render_ui(self):
        # Score
        score_surf = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Lives
        for i in range(self.MAX_LIVES):
            is_alive = i < self.lives
            color = self.FRUIT_COLORS["apple"] if is_alive else (80, 20, 20)
            center_x = self.WIDTH - 30 - (i * 35)
            pygame.gfxdraw.aacircle(self.screen, center_x, 25, 12, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, 25, 12, color)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 10, 20, 200))
        self.screen.blit(overlay, (0, 0))

        status = "You Win!" if self.score >= self.WIN_SCORE else "Game Over"
        color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
        
        title_surf = self.font.render(status, True, color)
        title_rect = title_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 25))
        self.screen.blit(title_surf, title_rect)

        score_surf = self.small_font.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 15))
        self.screen.blit(score_surf, score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "fall_speed": self.fall_speed,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    # Keep the game over screen for a few seconds
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        else:
            continue
        break
        
    env.close()
    pygame.quit()