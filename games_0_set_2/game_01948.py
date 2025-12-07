
# Generated: 2025-08-27T18:46:52.826390
# Source Brief: brief_01948.md
# Brief Index: 1948

        
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
        "Controls: Use arrow keys to move your character on the isometric grid. "
        "Space and Shift have no function in this game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect 20 glittering gems before the 60-second timer runs out. "
        "Dodge the patrolling guardians. Gems closer to guardians are worth more points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    
    GRID_WIDTH = 22
    GRID_HEIGHT = 22
    TILE_WIDTH_HALF = 16
    TILE_HEIGHT_HALF = 8

    MAX_TIME_SECONDS = 60
    MAX_GEMS_TO_WIN = 20
    NUM_GEMS_ON_MAP = 8
    NUM_ENEMIES = 3
    
    # Colors
    COLOR_BG = (25, 35, 55)
    COLOR_GRID = (45, 55, 75)
    
    COLOR_PLAYER = (255, 128, 0)
    COLOR_PLAYER_GLOW = (255, 128, 0, 50)
    
    COLOR_ENEMY = (160, 30, 200)
    COLOR_ENEMY_GLOW = (160, 30, 200, 70)
    
    COLOR_GEM_PALETTE = [
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (255, 0, 255),  # Magenta
    ]
    
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_TEXT_SHADOW = (20, 20, 20)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # World origin for centering the isometric grid
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 50

        # State variables are initialized in reset()
        self.player_pos_iso = None
        self.enemies = []
        self.gems = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.timer = 0
        self.game_over = False
        self.last_reward = 0
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.timer = self.MAX_TIME_SECONDS * self.FPS
        self.game_over = False
        self.last_reward = 0
        
        # Player state
        self.player_pos_iso = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Enemy state
        self.enemies = []
        self._spawn_enemies()

        # Gem state
        self.gems = []
        for _ in range(self.NUM_GEMS_ON_MAP):
            self._spawn_gem()
        self._update_gem_values()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        # --- Game Logic Update ---
        self.steps += 1
        self.timer -= 1

        # Store state for reward calculation
        prev_player_pos = list(self.player_pos_iso)
        dist_to_nearest_gem_before = self._get_dist_to_nearest_gem()

        # Update player
        self._update_player(movement)
        
        # Update enemies
        self._update_enemies()
        
        # Update gem values based on new enemy positions
        self._update_gem_values()

        # --- Collision and Event Handling ---
        collected_gem_value = self._check_gem_collection()
        player_hit_enemy = self._check_enemy_collision()
        
        # --- Reward Calculation ---
        dist_to_nearest_gem_after = self._get_dist_to_nearest_gem()
        
        reward = 0
        # Movement reward
        if dist_to_nearest_gem_after < dist_to_nearest_gem_before:
            reward += 1.0  # Moved closer
        elif dist_to_nearest_gem_after > dist_to_nearest_gem_before:
            reward -= 0.1  # Moved away
        
        # Event rewards
        if collected_gem_value > 0:
            reward += 1.0 * collected_gem_value
            # sfx: gem collect sound
        
        # --- Termination Check ---
        terminated = False
        if self.gems_collected >= self.MAX_GEMS_TO_WIN:
            reward += 100  # Victory bonus
            terminated = True
            self.game_over = True
            # sfx: victory fanfare
        elif self.timer <= 0 or player_hit_enemy:
            reward -= 100  # Failure penalty
            terminated = True
            self.game_over = True
            # sfx: failure sound
        
        self.last_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Helper Methods: Game State ---

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_pos_iso[1] -= 1
        elif movement == 2:  # Down
            self.player_pos_iso[1] += 1
        elif movement == 3:  # Left
            self.player_pos_iso[0] -= 1
        elif movement == 4:  # Right
            self.player_pos_iso[0] += 1
        
        # Clamp player position to grid boundaries
        self.player_pos_iso[0] = max(0, min(self.GRID_WIDTH - 1, self.player_pos_iso[0]))
        self.player_pos_iso[1] = max(0, min(self.GRID_HEIGHT - 1, self.player_pos_iso[1]))

    def _update_enemies(self):
        for enemy in self.enemies:
            if self.steps % 4 != 0: # Move enemies slower than player
                continue

            target_pos = enemy['path'][enemy['target_idx']]
            if enemy['pos'] == target_pos:
                enemy['target_idx'] = (enemy['target_idx'] + 1) % len(enemy['path'])
                target_pos = enemy['path'][enemy['target_idx']]
            
            # Move one step towards target
            if enemy['pos'][0] < target_pos[0]: enemy['pos'][0] += 1
            elif enemy['pos'][0] > target_pos[0]: enemy['pos'][0] -= 1
            if enemy['pos'][1] < target_pos[1]: enemy['pos'][1] += 1
            elif enemy['pos'][1] > target_pos[1]: enemy['pos'][1] -= 1
    
    def _update_gem_values(self):
        for gem in self.gems:
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = self._iso_dist(gem['pos'], enemy['pos'])
                if dist < min_dist:
                    min_dist = dist
            # Value is 5 for distance <= 2, 4 for <= 4, etc.
            gem['value'] = max(1, 6 - int(min_dist / 2.0))

    def _spawn_enemies(self):
        # Pre-defined patrol paths
        paths = [
            [[3, 3], [18, 3], [18, 8], [3, 8]],
            [[3, 12], [8, 12], [8, 18], [3, 18]],
            [[12, 12], [18, 12], [18, 18], [12, 18]],
        ]
        for i in range(self.NUM_ENEMIES):
            path = paths[i % len(paths)]
            self.enemies.append({
                'pos': list(path[0]),
                'path': path,
                'target_idx': 1,
            })

    def _spawn_gem(self):
        while True:
            pos = [
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT),
            ]
            # Ensure it doesn't spawn on the player or another gem
            if pos != self.player_pos_iso and not any(g['pos'] == pos for g in self.gems):
                self.gems.append({
                    'pos': pos,
                    'value': 1,
                    'color': self.np_random.choice(self.COLOR_GEM_PALETTE),
                    'anim_offset': self.np_random.integers(0, self.FPS)
                })
                break

    def _check_gem_collection(self):
        for i, gem in enumerate(self.gems):
            if gem['pos'] == self.player_pos_iso:
                self.score += gem['value']
                self.gems_collected += 1
                collected_value = gem['value']
                del self.gems[i]
                self._spawn_gem()
                return collected_value
        return 0

    def _check_enemy_collision(self):
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos_iso:
                return True
        return False
    
    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return float('inf')
        return min(self._iso_dist(self.player_pos_iso, gem['pos']) for gem in self.gems)
    
    def _iso_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # --- Helper Methods: Rendering ---

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
            "gems_collected": self.gems_collected,
            "time_left": max(0, self.timer / self.FPS),
        }
    
    def _iso_to_cart(self, iso_x, iso_y):
        cart_x = (iso_x - iso_y) * self.TILE_WIDTH_HALF + self.origin_x
        cart_y = (iso_x + iso_y) * self.TILE_HEIGHT_HALF + self.origin_y
        return int(cart_x), int(cart_y)
    
    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=(1, 1)):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Render grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_cart(i, 0)
            end = self._iso_to_cart(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_cart(0, i)
            end = self._iso_to_cart(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        
        # Collect all drawable entities and sort by y-coordinate for correct layering
        renderables = []
        
        for enemy in self.enemies:
            renderables.append(('enemy', enemy))
        for gem in self.gems:
            renderables.append(('gem', gem))
        renderables.append(('player', {'pos': self.player_pos_iso}))
        
        renderables.sort(key=lambda item: self._iso_to_cart(*item[1]['pos'])[1])
        
        for type, data in renderables:
            if type == 'player':
                self._render_player(data)
            elif type == 'enemy':
                self._render_enemy(data)
            elif type == 'gem':
                self._render_gem(data)

    def _render_player(self, data):
        x, y = self._iso_to_cart(*data['pos'])
        
        # Glow effect
        glow_radius = self.TILE_WIDTH_HALF * 1.5
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))

        # Player shape (diamond)
        points = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_enemy(self, enemy):
        x, y = self._iso_to_cart(*enemy['pos'])
        
        # Glow effect
        glow_radius = self.TILE_WIDTH_HALF * 1.2
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))

        # Enemy shape (diamond)
        points = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)

    def _render_gem(self, gem):
        x, y = self._iso_to_cart(*gem['pos'])
        
        # Pulsing animation for sparkle
        anim_phase = (self.steps + gem['anim_offset']) % self.FPS
        size_mod = math.sin(anim_phase / self.FPS * 2 * math.pi) * 2
        radius = int(self.TILE_WIDTH_HALF * 0.4 + size_mod)
        radius = max(1, radius)

        # Gem shape (diamond)
        points = [
            (x, y - radius),
            (x + radius, y),
            (x, y + radius),
            (x - radius, y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, gem['color'])
        pygame.gfxdraw.aapolygon(self.screen, points, gem['color'])
        
        # Value text above gem
        self._render_text(
            str(gem['value']), self.font_small, (255,255,255), 
            (x - 4, y - self.TILE_HEIGHT_HALF * 2.5),
            shadow_color=(0,0,0)
        )

    def _render_ui(self):
        # Semi-transparent background bar
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # Score
        score_text = f"SCORE: {self.score}"
        self._render_text(score_text, self.font_large, self.COLOR_UI_TEXT, (15, 7), self.COLOR_UI_TEXT_SHADOW)

        # Gems collected
        gem_text = f"GEMS: {self.gems_collected} / {self.MAX_GEMS_TO_WIN}"
        self._render_text(gem_text, self.font_large, self.COLOR_UI_TEXT, (220, 7), self.COLOR_UI_TEXT_SHADOW)
        
        # Timer
        time_left = max(0, self.timer / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        time_color = (255, 80, 80) if time_left < 10 else self.COLOR_UI_TEXT
        self._render_text(time_text, self.font_large, time_color, (self.SCREEN_WIDTH - 160, 7), self.COLOR_UI_TEXT_SHADOW)

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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a window, so it won't work in a headless environment
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Collector")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(env.FPS)
            
        print(f"Game Over! Final Info: {info}")
        
    except pygame.error as e:
        print("Pygame display could not be initialized. This is expected in a headless environment.")
        print("The environment itself is functional.")

    finally:
        env.close()