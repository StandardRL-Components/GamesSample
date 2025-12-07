
# Generated: 2025-08-27T12:29:17.061024
# Source Brief: brief_00060.md
# Brief Index: 60

        
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
    """
    An isometric arcade game where the player collects gems while dodging enemies.
    The goal is to collect 20 gems within 60 seconds without being hit 3 times.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character. Collect the sparkling gems "
        "and avoid the purple enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect sparkling gems in an isometric world while dodging cunning enemies "
        "to amass a high score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 20
    TILE_WIDTH, TILE_HEIGHT = 32, 16
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    WIN_CONDITION_GEMS = 20
    MAX_LIVES = 3
    NUM_ENEMIES = 4
    NUM_GEMS = 5

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (0, 220, 220)
    COLOR_PLAYER_SHADOW = (10, 15, 20)
    COLOR_ENEMY = (150, 40, 200)
    COLOR_ENEMY_FLASH = (255, 0, 0)
    COLOR_GEM = [(255, 220, 0), (0, 255, 100), (255, 50, 50), (100, 150, 255)]
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_TIMER_BACK = (60, 60, 80)
    COLOR_UI_TIMER_FRONT = (100, 200, 255)
    COLOR_HEART = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_visual_pos = None
        self.player_target_pos = None
        self.player_lives = 0
        self.player_invincible_timer = 0
        self.gems_collected = 0
        self.gems = []
        self.enemies = []
        self.rng = None

        # Calculate grid offset to center it
        self.grid_offset_x = self.SCREEN_WIDTH / 2
        self.grid_offset_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT) / 2 + 50

        # Initialize state
        self.reset()
        
        # Self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = self.MAX_LIVES
        self.gems_collected = 0
        self.player_invincible_timer = 0

        # Player position
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2], dtype=float)
        self.player_target_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        self.player_visual_pos = np.array(self.player_target_pos, dtype=float)

        # Spawn entities
        self._spawn_gems(self.NUM_GEMS)
        self._spawn_enemies(self.NUM_ENEMIES)

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0.0
        terminated = False

        if not self.game_over:
            self.steps += 1
            reward += 0.01  # Small reward for surviving

            # --- Update Player ---
            self._update_player(movement)
            self.player_target_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1])

            # --- Update Enemies ---
            self._update_enemies()

            # --- Handle Collisions ---
            reward += self._handle_collisions()

            # --- Check Termination Conditions ---
            if self.player_lives <= 0:
                self.game_over = True
                terminated = True
                reward = -100.0
            elif self.gems_collected >= self.WIN_CONDITION_GEMS:
                self.game_over = True
                terminated = True
                reward = 50.0
                self.score += 500 # Win bonus
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
                reward = -10.0 # Time out penalty
        
        # Smooth visual interpolation
        self.player_visual_pos += (np.array(self.player_target_pos) - self.player_visual_pos) * 0.25

        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement):
        dx, dy = 0, 0
        if movement == 1:  # Up
            dx, dy = -1, -1
        elif movement == 2:  # Down
            dx, dy = 1, 1
        elif movement == 3:  # Left
            dx, dy = -1, 1
        elif movement == 4:  # Right
            dx, dy = 1, -1

        new_pos_x = self.player_pos[0] + dx
        new_pos_y = self.player_pos[1] + dy

        # Boundary checks
        if 0 <= new_pos_x < self.GRID_WIDTH and 0 <= new_pos_y < self.GRID_HEIGHT:
            self.player_pos[0] = new_pos_x
            self.player_pos[1] = new_pos_y

    def _update_enemies(self):
        for enemy in self.enemies:
            target_node = enemy['path'][enemy['path_index']]
            direction = np.array(target_node) - enemy['pos']
            
            if np.linalg.norm(direction) < 0.1:
                enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
            else:
                normalized_dir = direction / np.linalg.norm(direction)
                enemy['pos'] += normalized_dir * enemy['speed']
            
            if enemy['flash_timer'] > 0:
                enemy['flash_timer'] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player-Gem collision
        collected_indices = []
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player_pos, gem['pos']):
                collected_indices.append(i)
                self.gems_collected += 1
                self.score += 100
                reward += 10.0 # Significant reward for gem
                # Sfx: gem_collect.wav

        # Remove collected gems and spawn new ones
        if collected_indices:
            self.gems = [gem for i, gem in enumerate(self.gems) if i not in collected_indices]
            self._spawn_gems(len(collected_indices))
        
        # Player-Enemy collision
        if self.player_invincible_timer == 0:
            for enemy in self.enemies:
                if np.linalg.norm(self.player_pos - enemy['pos']) < 0.8: # Collision radius
                    self.player_lives -= 1
                    self.player_invincible_timer = 90 # 3 seconds of invincibility
                    reward -= 20.0 # Significant penalty for hit
                    enemy['flash_timer'] = 15 # Flash red
                    # Sfx: player_hit.wav
                    break
        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_background()
        self._render_game_elements()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "gems": self.gems_collected,
        }

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.grid_offset_x + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.grid_offset_y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_background(self):
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_game_elements(self):
        # Create a list of all renderable objects with their y-coordinate for sorting
        render_queue = []
        
        # Add gems
        for gem in self.gems:
            render_queue.append({'type': 'gem', 'obj': gem, 'y_pos': gem['pos'][1]})

        # Add enemies
        for enemy in self.enemies:
            render_queue.append({'type': 'enemy', 'obj': enemy, 'y_pos': enemy['pos'][1]})

        # Add player
        render_queue.append({'type': 'player', 'obj': None, 'y_pos': self.player_pos[1]})

        # Sort by y-coordinate for correct isometric rendering
        render_queue.sort(key=lambda item: item['y_pos'])

        for item in render_queue:
            if item['type'] == 'gem':
                self._draw_gem(item['obj'])
            elif item['type'] == 'enemy':
                self._draw_enemy(item['obj'])
            elif item['type'] == 'player':
                self._draw_player()

    def _draw_iso_poly(self, surface, color, points):
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_player(self):
        x, y = self.player_visual_pos
        size = self.TILE_WIDTH * 0.4
        
        # Shadow
        shadow_points = [
            (x, y + size * 0.9),
            (x + size * 0.5, y + size * 0.9 + size * 0.25),
            (x, y + size * 0.9 + size * 0.5),
            (x - size * 0.5, y + size * 0.9 + size * 0.25),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in shadow_points], self.COLOR_PLAYER_SHADOW)

        # Body
        color = self.COLOR_PLAYER
        if self.player_invincible_timer > 0 and (self.steps // 3) % 2 == 0:
            color = self.COLOR_BG # Flicker when invincible
        
        body_points = [
            (x, y - size * 0.25),
            (x + size * 0.75, y + size * 0.25),
            (x, y + size * 0.75),
            (x - size * 0.75, y + size * 0.25)
        ]
        self._draw_iso_poly(self.screen, color, [(int(px), int(py)) for px, py in body_points])

    def _draw_gem(self, gem):
        x, y = self._iso_to_screen(gem['pos'][0], gem['pos'][1])
        pulse = math.sin(self.steps * 0.1 + gem['phase']) * 2 + 8
        
        points = [
            (x, y - pulse * 0.7),
            (x + pulse * 0.5, y),
            (x, y + pulse * 0.7),
            (x - pulse * 0.5, y)
        ]
        self._draw_iso_poly(self.screen, gem['color'], points)
        
        # Sparkle effect
        sparkle_x = x + math.sin(self.steps * 0.2 + gem['phase'] * 2) * 5
        sparkle_y = y - 10 + math.cos(self.steps * 0.2 + gem['phase'] * 2) * 5
        pygame.draw.circle(self.screen, (255, 255, 255), (int(sparkle_x), int(sparkle_y)), 1)

    def _draw_enemy(self, enemy):
        x, y = self._iso_to_screen(enemy['pos'][0], enemy['pos'][1])
        size = self.TILE_WIDTH * 0.35
        
        color = self.COLOR_ENEMY
        if enemy['flash_timer'] > 0:
            color = self.COLOR_ENEMY_FLASH

        body_points = [
            (x, y),
            (x + size, y + size * 0.5),
            (x, y + size),
            (x - size, y + size * 0.5)
        ]
        self._draw_iso_poly(self.screen, color, [(int(px), int(py)) for px, py in body_points])


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gems collected
        gem_text = self.font_small.render(f"GEMS: {self.gems_collected}/{self.WIN_CONDITION_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 30))

        # Lives (Hearts)
        heart_size = 20
        for i in range(self.player_lives):
            x = self.SCREEN_WIDTH // 2 - (self.MAX_LIVES * (heart_size + 5)) // 2 + i * (heart_size + 5)
            pygame.draw.circle(self.screen, self.COLOR_HEART, (x + heart_size // 4, 10 + heart_size // 4), heart_size // 4)
            pygame.draw.circle(self.screen, self.COLOR_HEART, (x - heart_size // 4 + heart_size//2, 10 + heart_size // 4), heart_size // 4)
            pygame.draw.polygon(self.screen, self.COLOR_HEART, [(x-heart_size//2+heart_size//2, 10 + heart_size//4), (x+heart_size//2, 10 + heart_size//4), (x, 10+heart_size)])


        # Timer Bar
        timer_width = 150
        timer_height = 15
        time_left_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        pygame.draw.rect(self.screen, self.COLOR_UI_TIMER_BACK, (self.SCREEN_WIDTH - timer_width - 10, 10, timer_width, timer_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TIMER_FRONT, (self.SCREEN_WIDTH - timer_width - 10, 10, timer_width * time_left_ratio, timer_height))

        # Game Over/Win Text
        if self.game_over:
            if self.gems_collected >= self.WIN_CONDITION_GEMS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _spawn_gems(self, count):
        occupied_pos = {tuple(gem['pos']) for gem in self.gems}
        occupied_pos.add(tuple(self.player_pos))
        for _ in range(count):
            while True:
                pos = self.rng.integers(0, [self.GRID_WIDTH, self.GRID_HEIGHT], size=2)
                if tuple(pos) not in occupied_pos:
                    self.gems.append({
                        'pos': pos,
                        'color': random.choice(self.COLOR_GEM),
                        'phase': self.rng.random() * 2 * math.pi
                    })
                    occupied_pos.add(tuple(pos))
                    break

    def _spawn_enemies(self, count):
        self.enemies = []
        for i in range(count):
            path = []
            if i % 4 == 0: # Horizontal patrol
                y = self.rng.integers(3, self.GRID_HEIGHT - 3)
                path = [[2, y], [self.GRID_WIDTH - 3, y]]
            elif i % 4 == 1: # Vertical patrol
                x = self.rng.integers(3, self.GRID_WIDTH - 3)
                path = [[x, 2], [x, self.GRID_HEIGHT - 3]]
            elif i % 4 == 2: # Box patrol
                cx, cy = self.rng.integers(5, self.GRID_WIDTH - 5, size=2)
                s = self.rng.integers(3, 6)
                path = [[cx-s, cy-s], [cx+s, cy-s], [cx+s, cy+s], [cx-s, cy+s]]
            else: # Diagonal patrol
                path = [[2, 2], [self.GRID_WIDTH - 3, self.GRID_HEIGHT - 3]]
            
            self.enemies.append({
                'pos': np.array(path[0], dtype=float),
                'path': path,
                'path_index': 0,
                'speed': self.rng.uniform(0.04, 0.08),
                'flash_timer': 0
            })

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses a different coordinate system, so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()