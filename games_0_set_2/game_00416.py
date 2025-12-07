
# Generated: 2025-08-27T13:34:48.743765
# Source Brief: brief_00416.md
# Brief Index: 416

        
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
        "Controls: Arrow keys to move your character. "
        "Collect the glowing gems to score points."
    )

    game_description = (
        "Collect shimmering gems while dodging cunning enemies in an isometric arcade world. "
        "Collect 50 gems to win, but lose all 3 lives and you lose the game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 28, 14
    
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_PLAYER = (50, 220, 100)
    COLOR_PLAYER_SHADOW = (20, 25, 30)
    COLOR_ENEMY = (180, 50, 90)
    COLOR_ENEMY_SHADOW = (20, 25, 30)
    GEM_COLORS = [(255, 220, 50), (50, 200, 255), (255, 80, 80)]
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Game parameters
    MAX_STEPS = 1000
    WIN_SCORE = 50
    INITIAL_LIVES = 3
    NUM_ENEMIES = 3
    NUM_GEMS = 5
    INITIAL_ENEMY_SPEED = 0.02
    ENEMY_SPEED_INCREASE = 0.005 # Per 100 steps
    RISKY_GEM_DISTANCE = 3.0 # Grid units for bonus

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        
        # This will be set in reset()
        self.np_random = None
        self.player = {}
        self.enemies = []
        self.gems = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 80

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_speed = self.INITIAL_ENEMY_SPEED

        self.player = {
            "x": self.GRID_WIDTH // 2,
            "y": self.GRID_HEIGHT // 2,
            "lives": self.INITIAL_LIVES,
            "bob": 0.0,
        }

        occupied_cells = {(self.player["x"], self.player["y"])}
        
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            enemy = self._spawn_enemy(occupied_cells)
            self.enemies.append(enemy)
            occupied_cells.add((enemy["x"], enemy["y"]))

        self.gems = []
        for _ in range(self.NUM_GEMS):
            gem = self._spawn_gem(occupied_cells)
            self.gems.append(gem)
            occupied_cells.add((gem["x"], gem["y"]))
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Calculate distance-based reward before moving ---
        reward = self._calculate_distance_reward(movement)
        
        # --- Update game state ---
        self._update_player(movement)
        self._update_enemies()
        
        # --- Handle collisions and collect event-based rewards ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Update animations and difficulty ---
        self.steps += 1
        self.player["bob"] = (self.player["bob"] + 0.2) % (2 * math.pi)
        for gem in self.gems:
            gem["sparkle"] = (gem["sparkle"] + 0.1) % (2 * math.pi)

        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_speed += self.ENEMY_SPEED_INCREASE
        
        # --- Check for termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win reward
            elif self.player["lives"] <= 0:
                reward -= 100  # Lose reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        # --- Sort all dynamic objects by Y-pos for correct isometric rendering ---
        render_queue = []
        for gem in self.gems:
            render_queue.append(("gem", gem))
        for enemy in self.enemies:
            render_queue.append(("enemy", enemy))
        render_queue.append(("player", self.player))
        
        render_queue.sort(key=lambda item: item[1]['y'] * self.GRID_WIDTH + item[1]['x'])
        
        for item_type, obj in render_queue:
            if item_type == "player":
                self._render_player()
            elif item_type == "enemy":
                self._render_enemy(obj)
            elif item_type == "gem":
                self._render_gem(obj)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player["lives"],
        }
        
    # --- Update Logic ---
    def _update_player(self, movement):
        px, py = self.player["x"], self.player["y"]
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        self.player["x"] = np.clip(px, 0, self.GRID_WIDTH - 1)
        self.player["y"] = np.clip(py, 0, self.GRID_HEIGHT - 1)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['pos_f'] += self.enemy_speed
            if enemy['pos_f'] >= 1.0:
                enemy['pos_f'] = 0.0
                if enemy['x'] < enemy['tx']: enemy['x'] += 1
                elif enemy['x'] > enemy['tx']: enemy['x'] -= 1
                elif enemy['y'] < enemy['ty']: enemy['y'] += 1
                elif enemy['y'] > enemy['ty']: enemy['y'] -= 1
                
                if (enemy['x'], enemy['y']) == (enemy['tx'], enemy['ty']):
                    enemy['tx'], enemy['ty'] = enemy['sx'], enemy['sy']
                    enemy['sx'], enemy['sy'] = enemy['x'], enemy['y']

    def _handle_collisions(self):
        reward = 0
        
        # Player-Gem
        collected_indices = []
        for i, gem in enumerate(self.gems):
            if self.player["x"] == gem["x"] and self.player["y"] == gem["y"]:
                # Sound: gem_collect.wav
                self.score += 1
                reward += 1.0

                # Check for risky collection bonus
                min_dist_to_enemy = self._get_min_dist((self.player['x'], self.player['y']), self.enemies)
                if min_dist_to_enemy is not None and min_dist_to_enemy <= self.RISKY_GEM_DISTANCE:
                    reward += 2.0 # Bonus reward
                
                collected_indices.append(i)

        if collected_indices:
            occupied_cells = {(e['x'], e['y']) for e in self.enemies}
            occupied_cells.add((self.player['x'], self.player['y']))
            for g in self.gems: occupied_cells.add((g['x'], g['y']))

            for i in sorted(collected_indices, reverse=True):
                del self.gems[i]
                new_gem = self._spawn_gem(occupied_cells)
                self.gems.append(new_gem)
                occupied_cells.add((new_gem['x'], new_gem['y']))
                
        # Player-Enemy
        for enemy in self.enemies:
            if self.player["x"] == enemy["x"] and self.player["y"] == enemy["y"]:
                # Sound: player_hit.wav
                self.player["lives"] -= 1
                reward -= 10.0
                if self.player["lives"] > 0:
                    # Teleport player to center to avoid instant multi-hits
                    self.player["x"] = self.GRID_WIDTH // 2
                    self.player["y"] = self.GRID_HEIGHT // 2
                break
        return reward

    def _calculate_distance_reward(self, movement):
        reward = 0
        player_pos = (self.player['x'], self.player['y'])
        
        # Reward for moving towards nearest gem
        old_dist_gem = self._get_min_dist(player_pos, self.gems)
        if old_dist_gem is not None:
            new_player_pos = self._get_potential_pos(player_pos, movement)
            new_dist_gem = self._get_min_dist(new_player_pos, self.gems)
            if new_dist_gem < old_dist_gem:
                reward += 0.1
        
        # Penalty for moving towards nearest enemy
        old_dist_enemy = self._get_min_dist(player_pos, self.enemies)
        if old_dist_enemy is not None:
            new_player_pos = self._get_potential_pos(player_pos, movement)
            new_dist_enemy = self._get_min_dist(new_player_pos, self.enemies)
            if new_dist_enemy < old_dist_enemy:
                reward -= 0.1
                
        return reward
        
    def _check_termination(self):
        return (
            self.player["lives"] <= 0
            or self.score >= self.WIN_SCORE
            or self.steps >= self.MAX_STEPS
        )

    # --- Spawning Logic ---
    def _spawn_enemy(self, occupied_cells):
        while True:
            sx, sy = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if (sx, sy) not in occupied_cells:
                break
        
        while True:
            tx, ty = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if (tx, ty) not in occupied_cells and (tx, ty) != (sx, sy):
                break
        
        return {"x": sx, "y": sy, "sx": sx, "sy": sy, "tx": tx, "ty": ty, "pos_f": 0.0}

    def _spawn_gem(self, occupied_cells):
        while True:
            x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in occupied_cells:
                break
        color_index = self.np_random.integers(0, len(self.GEM_COLORS))
        return {"x": x, "y": y, "color": self.GEM_COLORS[color_index], "sparkle": self.np_random.random() * 2 * math.pi}

    # --- Rendering ---
    def _iso_to_cart(self, x, y):
        sx = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        sy = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(sx), int(sy)

    def _draw_iso_cube(self, surface, x, y, color, shadow_color, height=20, bob=0):
        sx, sy = self._iso_to_cart(x, y)
        sy -= bob
        
        shadow_points = [
            (sx, sy + height / 2),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF + height / 2),
            (sx, sy + self.TILE_HEIGHT_HALF * 2 + height / 2),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF + height / 2),
        ]
        pygame.gfxdraw.filled_polygon(surface, shadow_points, shadow_color)
        
        top_points = [
            (sx, sy - height),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF - height),
            (sx, sy + self.TILE_HEIGHT_HALF * 2 - height),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF - height),
        ]
        
        left_color = tuple(max(0, c - 30) for c in color)
        right_color = tuple(max(0, c - 60) for c in color)
        
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)
        
        # Right face
        right_points = [top_points[1], (top_points[1][0], top_points[1][1] + height), (top_points[2][0], top_points[2][1] + height), top_points[2]]
        pygame.gfxdraw.filled_polygon(surface, right_points, right_color)
        pygame.gfxdraw.aapolygon(surface, right_points, right_color)

        # Left face
        left_points = [top_points[3], (top_points[3][0], top_points[3][1] + height), (top_points[2][0], top_points[2][1] + height), top_points[2]]
        pygame.gfxdraw.filled_polygon(surface, left_points, left_color)
        pygame.gfxdraw.aapolygon(surface, left_points, left_color)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_cart(0, y)
            p2 = self._iso_to_cart(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for x in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_cart(x, 0)
            p2 = self._iso_to_cart(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
    
    def _render_player(self):
        bob_amount = math.sin(self.player["bob"]) * 3
        self._draw_iso_cube(self.screen, self.player["x"], self.player["y"], self.COLOR_PLAYER, self.COLOR_PLAYER_SHADOW, height=20, bob=bob_amount)

    def _render_enemy(self, enemy):
        self._draw_iso_cube(self.screen, enemy["x"], enemy["y"], self.COLOR_ENEMY, self.COLOR_ENEMY_SHADOW, height=20)
        
    def _render_gem(self, gem):
        sx, sy = self._iso_to_cart(gem["x"], gem["y"])
        size = 8 + math.sin(gem["sparkle"]) * 3
        
        # Draw shadow
        shadow_rect = pygame.Rect(0, 0, size, size/2)
        shadow_rect.center = (sx, sy + 10)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 50), (0, 0, shadow_rect.width, shadow_rect.height))
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Draw gem
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (sx, sy)
        pygame.draw.ellipse(self.screen, gem["color"], rect)
        
        # Draw sparkle
        highlight_color = tuple(min(255, c + 80) for c in gem["color"])
        pygame.gfxdraw.filled_circle(self.screen, int(sx - size/4), int(sy - size/4), int(size/6), highlight_color)
        
    def _render_ui(self):
        # Score Text
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (10, 10), self.font_large)
        
        # Lives Text
        lives_text = f"LIVES: {self.player['lives']}"
        text_width = self.font_large.size(lives_text)[0]
        self._draw_text(lives_text, (self.SCREEN_WIDTH - text_width - 10, 10), self.font_large)

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_large, center=True)

    def _draw_text(self, text, pos, font, center=False):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    # --- Helper Functions ---
    def _get_min_dist(self, pos, targets):
        if not targets:
            return None
        min_dist = float('inf')
        for target in targets:
            dist = abs(pos[0] - target['x']) + abs(pos[1] - target['y']) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_potential_pos(self, pos, movement):
        px, py = pos
        if movement == 1: py -= 1
        elif movement == 2: py += 1
        elif movement == 3: px -= 1
        elif movement == 4: px += 1
        px = np.clip(px, 0, self.GRID_WIDTH - 1)
        py = np.clip(py, 0, self.GRID_HEIGHT - 1)
        return (px, py)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display and is for testing purposes.
    # It will not run in a headless environment.
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Isometric Gem Collector")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            movement = 0 # No-op
            space_held = 0
            shift_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            env.clock.tick(30) # Match the intended FPS
            
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("This error is expected in a headless environment. The environment itself is functional.")
    finally:
        env.close()