
# Generated: 2025-08-27T16:18:39.457414
# Source Brief: brief_01186.md
# Brief Index: 1186

        
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
        "Controls: Use arrow keys (↑↓←→) to move the ninja. "
        "Collect the yellow numbers while avoiding the red enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Number Ninja is a top-down arcade game where the player collects "
        "target numbers while dodging enemies to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (16, 16, 32)  # Dark Blue
    COLOR_WALL = (48, 48, 64)
    COLOR_PLAYER = (0, 255, 0)
    COLOR_PLAYER_GLOW = (0, 255, 0, 50)
    COLOR_ENEMY = (255, 0, 0)
    COLOR_NUMBER = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Game parameters
    PLAYER_SIZE = 16
    PLAYER_SPEED = 4
    WALL_MARGIN = 20
    NUM_ENEMIES = 4
    ENEMY_SIZE = 12
    ENEMY_BASE_SPEED = 0.02 # Radians per frame
    NUMBER_SIZE = 12
    WIN_CONDITION_SCORE = 50
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    DIFFICULTY_INTERVAL = 500 # Steps to increase enemy speed

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
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...
        self.ninja_pos = None
        self.enemies = []
        self.number = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.numbers_collected = 0
        self.time_left = 0
        self.enemy_current_speed = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.ninja_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            path_radius = self.np_random.uniform(50, 150)
            path_center = np.array([
                self.np_random.uniform(self.WALL_MARGIN + path_radius, self.SCREEN_WIDTH - self.WALL_MARGIN - path_radius),
                self.np_random.uniform(self.WALL_MARGIN + path_radius, self.SCREEN_HEIGHT - self.WALL_MARGIN - path_radius)
            ])
            start_angle = self.np_random.uniform(0, 2 * math.pi)
            self.enemies.append({
                "center": path_center,
                "radius": path_radius,
                "angle": start_angle,
                "pos": path_center + np.array([math.cos(start_angle), math.sin(start_angle)]) * path_radius
            })

        self._spawn_number()

        self.particles = []
        self.steps = 0
        self.score = 0
        self.numbers_collected = 0
        self.time_left = self.MAX_STEPS
        self.enemy_current_speed = self.ENEMY_BASE_SPEED
        self.game_over = False
        self.win = False

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01  # Small reward for surviving

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            # space_held and shift_held are not used in this game
            
            # Update game logic
            self._move_player(movement)
            self._update_enemies()
            self._update_particles()
            
            # Handle collisions and events
            reward += self._handle_collisions()
            
            # Update timers and difficulty
            self.steps += 1
            self.time_left -= 1

            if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
                self.enemy_current_speed += 0.001

        # Check for termination conditions
        terminated = False
        if self.win:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.time_left <= 0 and not self.game_over:
            # SFX: Time-up sound
            reward -= 50
            self.game_over = True
            terminated = True
        elif self.game_over: # Covers enemy collision case
            terminated = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_collisions(self):
        reward = 0
        # Number collection
        if self._check_collision(self.ninja_pos, self.PLAYER_SIZE, self.number['pos'], self.NUMBER_SIZE):
            # SFX: Collect sound
            reward += 10
            self.score += 10
            self.numbers_collected += 1
            self._create_particles(self.number['pos'], self.COLOR_NUMBER, 20)
            if self.numbers_collected >= self.WIN_CONDITION_SCORE:
                self.win = True
            else:
                self._spawn_number()
        
        # Enemy collision
        for enemy in self.enemies:
            if self._check_collision(self.ninja_pos, self.PLAYER_SIZE, enemy['pos'], self.ENEMY_SIZE):
                # SFX: Explosion sound
                reward -= 100
                self.game_over = True
                self._create_particles(self.ninja_pos, self.COLOR_ENEMY, 50)
                break
        return reward
    
    def _spawn_number(self):
        while True:
            pos = self.np_random.uniform(
                low=[self.WALL_MARGIN + self.NUMBER_SIZE, self.WALL_MARGIN + self.NUMBER_SIZE],
                high=[self.SCREEN_WIDTH - self.WALL_MARGIN - self.NUMBER_SIZE, self.SCREEN_HEIGHT - self.WALL_MARGIN - self.NUMBER_SIZE],
                size=(2,)
            ).astype(np.float32)
            if np.linalg.norm(pos - self.ninja_pos) > self.PLAYER_SIZE * 3:
                self.number = {'pos': pos}
                break

    def _move_player(self, movement):
        direction = np.array([0, 0], dtype=np.float32)
        if movement == 1: direction[1] = -1  # Up
        elif movement == 2: direction[1] = 1   # Down
        elif movement == 3: direction[0] = -1  # Left
        elif movement == 4: direction[0] = 1   # Right
        
        self.ninja_pos += direction * self.PLAYER_SPEED
        
        self.ninja_pos[0] = np.clip(self.ninja_pos[0], self.WALL_MARGIN + self.PLAYER_SIZE / 2, self.SCREEN_WIDTH - self.WALL_MARGIN - self.PLAYER_SIZE / 2)
        self.ninja_pos[1] = np.clip(self.ninja_pos[1], self.WALL_MARGIN + self.PLAYER_SIZE / 2, self.SCREEN_HEIGHT - self.WALL_MARGIN - self.PLAYER_SIZE / 2)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += self.enemy_current_speed
            enemy['pos'] = enemy['center'] + np.array([math.cos(enemy['angle']), math.sin(enemy['angle'])]) * enemy['radius']

    def _check_collision(self, pos1, size1, pos2, size2):
        distance = np.linalg.norm(pos1 - pos2)
        return distance < (size1 / 2 + size2 / 2)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'lifespan': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_walls()
        self._render_particles()
        if not (self.game_over and not self.win):
             self._render_player()
        self._render_enemies()
        if not self.game_over:
            self._render_number()

    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), self.WALL_MARGIN)

    def _render_player(self):
        x, y = int(self.ninja_pos[0]), int(self.ninja_pos[1])
        size = self.PLAYER_SIZE
        
        glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, size * 2, size * 2))
        self.screen.blit(glow_surf, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x - size / 2, y - size / 2, size, size))

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            size = self.ENEMY_SIZE
            angle = self.steps * 0.1
            
            points = [
                (x + size * math.cos(angle + i * 2 * math.pi / 3), y + size * math.sin(angle + i * 2 * math.pi / 3))
                for i in range(3)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_number(self):
        x, y = int(self.number['pos'][0]), int(self.number['pos'][1])
        radius = self.NUMBER_SIZE
        
        brightness = 0.75 + 0.25 * math.sin(self.steps * 0.2)
        color = tuple(int(c * brightness) for c in self.COLOR_NUMBER)

        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            size = max(1, int(p['lifespan'] / 10))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        self._draw_text(f"SCORE: {self.score}", (10 + self.WALL_MARGIN, 10 + self.WALL_MARGIN), self.COLOR_TEXT, self.font_small, 'topleft')
        self._draw_text(f"NUMBERS: {self.numbers_collected}/{self.WIN_CONDITION_SCORE}", (self.SCREEN_WIDTH / 2, 10 + self.WALL_MARGIN), self.COLOR_TEXT, self.font_small, 'midtop')
        time_str = f"TIME: {max(0, self.time_left // self.FPS):02d}"
        self._draw_text(time_str, (self.SCREEN_WIDTH - 10 - self.WALL_MARGIN, 10 + self.WALL_MARGIN), self.COLOR_TEXT, self.font_small, 'topright')

        if self.game_over:
            msg = "YOU WIN!" if self.win else ("TIME UP!" if self.time_left <= 0 else "GAME OVER")
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), self.COLOR_TEXT, self.font_large, 'center')

    def _draw_text(self, text, pos, color, font, align="topleft"):
        text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect_shadow = text_surf_shadow.get_rect(**{align: (pos[0] + 2, pos[1] + 2)})
        self.screen.blit(text_surf_shadow, text_rect_shadow)
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(**{align: pos})
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "numbers_collected": self.numbers_collected,
            "time_left": self.time_left,
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
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Number Ninja")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 print("Resetting environment.")
                 total_reward = 0
                 obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()