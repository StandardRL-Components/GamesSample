
# Generated: 2025-08-28T06:26:30.947251
# Source Brief: brief_02937.md
# Brief Index: 2937

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move your character on the isometric grid. "
        "Collect all the gems to win, but watch out for the enemies!"
    )
    game_description = (
        "Collect sparkling gems while dodging cunning enemies in a vibrant isometric arcade world. "
        "Gather 20 gems to win, but lose all your health and the game is over."
    )

    # Frame advance behavior
    auto_advance = False

    # --- Game Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 30
    GRID_HEIGHT = 20
    TILE_WIDTH = 32
    TILE_HEIGHT = 16
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (50, 50, 70)
    COLOR_PLAYER = (255, 120, 0)
    COLOR_PLAYER_GLOW = (255, 120, 0, 50)
    ENEMY_COLORS = {
        'patrol_h': (190, 50, 220),
        'patrol_v': (220, 50, 150),
        'random': (150, 150, 150),
    }
    GEM_COLORS = [
        (255, 80, 80), (80, 255, 80), (80, 150, 255), (255, 255, 80)
    ]
    UI_TEXT_COLOR = (240, 240, 240)
    UI_HEALTH_BAR = (80, 200, 80)
    UI_HEALTH_BAR_BG = (200, 80, 80)
    
    # Game Parameters
    MAX_STEPS = 1000
    NUM_GEMS = 20
    NUM_ENEMIES = 8
    PLAYER_MAX_HEALTH = 3
    ENEMY_MOVE_INTERVAL = 2 # Enemies move every 2 steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.gems = []
        self.enemies = []
        self.gems_collected = 0
        self.np_random = None

        self.validate_implementation()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.player_health = self.PLAYER_MAX_HEALTH

        # Generate valid spawn points
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        # Place player
        self.player_pos = list(all_coords.pop())

        # Place gems
        self.gems = []
        for i in range(self.NUM_GEMS):
            pos = all_coords.pop()
            color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
            self.gems.append({'pos': list(pos), 'color': color})

        # Place enemies
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            pos = all_coords.pop()
            enemy_type_roll = i % 3
            if enemy_type_roll == 0: # Horizontal Patroller
                patrol_range = sorted([pos[0], self.np_random.integers(0, self.GRID_WIDTH)])
                if patrol_range[1] - patrol_range[0] < 2: patrol_range[1] = min(self.GRID_WIDTH - 1, patrol_range[0] + 2)
                self.enemies.append({'pos': list(pos), 'type': 'patrol_h', 'dir': 1, 'range': patrol_range})
            elif enemy_type_roll == 1: # Vertical Patroller
                patrol_range = sorted([pos[1], self.np_random.integers(0, self.GRID_HEIGHT)])
                if patrol_range[1] - patrol_range[0] < 2: patrol_range[1] = min(self.GRID_HEIGHT - 1, patrol_range[0] + 2)
                self.enemies.append({'pos': list(pos), 'type': 'patrol_v', 'dir': 1, 'range': patrol_range})
            else: # Random Mover
                self.enemies.append({'pos': list(pos), 'type': 'random'})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        
        # --- 1. Calculate pre-move state for reward calculation ---
        dist_to_closest_gem_before = self._get_dist_to_closest(self.player_pos, [g['pos'] for g in self.gems])
        dist_to_closest_enemy_before = self._get_dist_to_closest(self.player_pos, [e['pos'] for e in self.enemies])

        # --- 2. Process Player Action ---
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            new_x = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1] + dy))
            self.player_pos = [new_x, new_y]

        # --- 3. Update Enemies ---
        if self.steps % self.ENEMY_MOVE_INTERVAL == 0:
            for enemy in self.enemies:
                self._move_enemy(enemy)
        
        # --- 4. Handle Collisions and Events ---
        # Player-Gem Collision
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem['pos']:
                gem_to_remove = gem
                break
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.gems_collected += 1
            reward += 1.0  # +1 for collecting a gem
            self.score += 10
            # sfx: gem collect sound

        # Player-Enemy Collision
        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.player_health -= 1
                reward -= 5.0 # -5 for colliding with an enemy
                self.score -= 50
                # sfx: player damage sound
                break

        # --- 5. Calculate Proximity Rewards ---
        dist_to_closest_gem_after = self._get_dist_to_closest(self.player_pos, [g['pos'] for g in self.gems])
        dist_to_closest_enemy_after = self._get_dist_to_closest(self.player_pos, [e['pos'] for e in self.enemies])
        
        if dist_to_closest_gem_after < dist_to_closest_gem_before:
            reward += 0.1
        
        if dist_to_closest_enemy_after < dist_to_closest_enemy_before:
            reward -= 0.1

        # --- 6. Check for Termination ---
        self.steps += 1
        if self.gems_collected >= self.NUM_GEMS:
            reward += 100.0 # +100 for winning
            self.score += 1000
            terminated = True
            self.game_over = True
            # sfx: victory fanfare
        elif self.player_health <= 0:
            reward -= 100.0 # -100 for losing
            terminated = True
            self.game_over = True
            # sfx: game over sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_dist_to_closest(self, pos, target_list):
        if not target_list:
            return float('inf')
        return min(math.hypot(pos[0] - t[0], pos[1] - t[1]) for t in target_list)

    def _move_enemy(self, enemy):
        if enemy['type'] == 'patrol_h':
            enemy['pos'][0] += enemy['dir']
            if not enemy['range'][0] <= enemy['pos'][0] <= enemy['range'][1]:
                enemy['dir'] *= -1
                enemy['pos'][0] += 2 * enemy['dir'] # Correct position
        elif enemy['type'] == 'patrol_v':
            enemy['pos'][1] += enemy['dir']
            if not enemy['range'][0] <= enemy['pos'][1] <= enemy['range'][1]:
                enemy['dir'] *= -1
                enemy['pos'][1] += 2 * enemy['dir'] # Correct position
        elif enemy['type'] == 'random':
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            move = self.np_random.choice(len(moves))
            dx, dy = moves[move]
            enemy['pos'][0] = max(0, min(self.GRID_WIDTH - 1, enemy['pos'][0] + dx))
            enemy['pos'][1] = max(0, min(self.GRID_HEIGHT - 1, enemy['pos'][1] + dy))
        
        # Clamp positions just in case
        enemy['pos'][0] = max(0, min(self.GRID_WIDTH - 1, enemy['pos'][0]))
        enemy['pos'][1] = max(0, min(self.GRID_HEIGHT - 1, enemy['pos'][1]))

    def _grid_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(0, y)
            end = self._grid_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(x, 0)
            end = self._grid_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
            
        # Draw entities in depth-sorted order
        entities = []
        for gem in self.gems:
            entities.append({'pos': gem['pos'], 'type': 'gem', 'data': gem})
        for enemy in self.enemies:
            entities.append({'pos': enemy['pos'], 'type': 'enemy', 'data': enemy})
        entities.append({'pos': self.player_pos, 'type': 'player'})

        # Sort by grid y then x for correct isometric rendering
        entities.sort(key=lambda e: (e['pos'][0] + e['pos'][1], e['pos'][1]))

        for entity in entities:
            if entity['type'] == 'gem':
                self._render_gem(entity['data'])
            elif entity['type'] == 'enemy':
                self._render_enemy(entity['data'])
            elif entity['type'] == 'player':
                self._render_player()

    def _render_gem(self, gem):
        x, y = self._grid_to_screen(gem['pos'][0], gem['pos'][1])
        pulse = (math.sin(self.steps * 0.3 + x + y) + 1) / 2  # 0 to 1
        radius = int(4 + pulse * 3)
        pygame.gfxdraw.filled_circle(self.screen, x, y - 8, radius, gem['color'])
        pygame.gfxdraw.aacircle(self.screen, x, y - 8, radius, gem['color'])
        # Sparkle particle effect
        if self.np_random.random() < 0.1:
            spark_x = x + self.np_random.integers(-radius, radius)
            spark_y = y - 8 + self.np_random.integers(-radius, radius)
            pygame.draw.circle(self.screen, (255,255,255), (spark_x, spark_y), 1)

    def _render_enemy(self, enemy):
        x, y = self._grid_to_screen(enemy['pos'][0], enemy['pos'][1])
        color = self.ENEMY_COLORS[enemy['type']]
        points = [
            (x, y - self.TILE_HEIGHT),
            (x + self.TILE_WIDTH / 2, y - self.TILE_HEIGHT / 2),
            (x, y),
            (x - self.TILE_WIDTH / 2, y - self.TILE_HEIGHT / 2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_player(self):
        x, y = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        
        # Glow effect
        glow_radius = int(self.TILE_WIDTH * 0.7)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius - self.TILE_HEIGHT / 2))

        # Player diamond shape
        points = [
            (x, y - self.TILE_HEIGHT),
            (x + self.TILE_WIDTH / 2, y - self.TILE_HEIGHT / 2),
            (x, y),
            (x - self.TILE_WIDTH / 2, y - self.TILE_HEIGHT / 2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Gem count
        gem_text = self.font_large.render(f"GEMS: {self.gems_collected} / {self.NUM_GEMS}", True, self.UI_TEXT_COLOR)
        self.screen.blit(gem_text, (20, 15))

        # Health bar
        health_bar_width = 150
        health_bar_height = 20
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        
        # Background of health bar
        bg_rect = pygame.Rect(self.SCREEN_WIDTH - health_bar_width - 20, 20, health_bar_width, health_bar_height)
        pygame.draw.rect(self.screen, self.UI_HEALTH_BAR_BG, bg_rect, border_radius=5)
        
        # Foreground (current health)
        fg_rect = pygame.Rect(self.SCREEN_WIDTH - health_bar_width - 20, 20, int(health_bar_width * health_ratio), health_bar_height)
        pygame.draw.rect(self.screen, self.UI_HEALTH_BAR, fg_rect, border_radius=5)
        
        # Health text
        health_text = self.font_small.render("HEALTH", True, self.UI_TEXT_COLOR)
        self.screen.blit(health_text, (bg_rect.x + (bg_rect.width - health_text.get_width()) / 2, bg_rect.y + 2))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.gems_collected >= self.NUM_GEMS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surface, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
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
        
        # Test observation space (after a preliminary reset)
        self.reset()
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        # Pygame event handling
        action = [0, 0, 0] # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    terminated = False
                    continue

        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # Since auto_advance is False, we only step if an action is taken or on a timer
            # For manual play, we step on any key press
            if any(keys):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Limit frame rate for manual play
        env.clock.tick(10)

    env.close()