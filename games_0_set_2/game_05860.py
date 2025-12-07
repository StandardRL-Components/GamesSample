
# Generated: 2025-08-28T06:18:49.274482
# Source Brief: brief_05860.md
# Brief Index: 5860

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. Push crates onto the red targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Race against the clock to push all crates onto the target locations."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game parameters
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS # Use time limit as primary termination

    # Grid and Tile
    GRID_WIDTH = 12
    GRID_HEIGHT = 9
    TILE_WIDTH_HALF = 32
    TILE_HEIGHT_HALF = 16
    TILE_Z = 28 # Visual height of tiles/cubes

    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_FLOOR = (60, 68, 80)
    COLOR_FLOOR_SHADOW = (45, 52, 63)
    COLOR_WALL_TOP = (110, 120, 130)
    COLOR_WALL_SIDE = (85, 95, 105)
    COLOR_TARGET = (180, 50, 50)
    COLOR_TARGET_ACTIVE = (80, 190, 80)
    COLOR_CRATE_TOP = (160, 110, 70)
    COLOR_CRATE_SIDE = (120, 80, 50)
    COLOR_PLAYER_TOP = (60, 140, 220)
    COLOR_PLAYER_SIDE = (40, 100, 180)
    COLOR_SHADOW = (0, 0, 0, 64)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 45, 50, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Isometric projection origins
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 100
        
        # Game state variables are initialized in reset()
        self.level_layout = [
            "WWWWWWWWWWWW",
            "W....T.....W",
            "W.P........W",
            "W...C...C..W",
            "W.T.W...W.TW",
            "W...C...C..W",
            "W........T.W",
            "W.T........W",
            "WWWWWWWWWWWW",
        ]

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS

        self.walls = []
        self.crates = []
        self.targets = []
        self.renderables = []
        
        # Parse level layout
        for r, row in enumerate(self.level_layout):
            for c, char in enumerate(row):
                pos = [c, r]
                if char == 'W':
                    self.walls.append(tuple(pos))
                elif char == 'P':
                    self.player = {'pos': pos, 'vis_pos': list(pos), 'type': 'player'}
                elif char == 'C':
                    crate = {'pos': pos, 'vis_pos': list(pos), 'type': 'crate'}
                    self.crates.append(crate)
                elif char == 'T':
                    self.targets.append(tuple(pos))
        
        self.renderables = [self.player] + self.crates

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = 0
        
        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1

        # Store pre-move state for reward calculation
        prev_crates_on_target = self._get_crates_on_target_count()
        prev_crate_dists = [self._get_min_dist_to_target(c['pos']) for c in self.crates]

        # Player Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            player_pos = self.player['pos']
            next_pos = (player_pos[0] + dx, player_pos[1] + dy)

            if next_pos not in self.walls:
                crate_to_push = self._get_crate_at(next_pos)
                if crate_to_push:
                    # Attempting to push a crate
                    push_to_pos = (next_pos[0] + dx, next_pos[1] + dy)
                    if push_to_pos not in self.walls and not self._get_crate_at(push_to_pos):
                        # Valid push
                        crate_to_push['pos'][0] = push_to_pos[0]
                        crate_to_push['pos'][1] = push_to_pos[1]
                        self.player['pos'][0] = next_pos[0]
                        self.player['pos'][1] = next_pos[1]
                        # sfx: Crate push
                else:
                    # Simple movement
                    self.player['pos'][0] = next_pos[0]
                    self.player['pos'][1] = next_pos[1]
                    # sfx: Footstep
        
        # Smooth interpolation for visual positions
        for entity in self.renderables:
            for i in range(2):
                entity['vis_pos'][i] += (entity['pos'][i] - entity['vis_pos'][i]) * 0.5
        
        # --- Calculate Reward ---
        current_crates_on_target = self._get_crates_on_target_count()
        new_crate_dists = [self._get_min_dist_to_target(c['pos']) for c in self.crates]
        
        # Reward for placing a crate on a target
        if current_crates_on_target > prev_crates_on_target:
            reward += 10 * (current_crates_on_target - prev_crates_on_target)
            # sfx: Success chime
        
        # Reward/penalty for changing distance to nearest target
        dist_change_reward = 0
        for i in range(len(self.crates)):
            diff = prev_crate_dists[i] - new_crate_dists[i]
            if abs(diff) > 0.01: # Only if moved
                dist_change_reward += diff * 0.1 # Positive if closer, negative if further
        reward += dist_change_reward

        # --- Check Termination ---
        terminated = False
        if current_crates_on_target == len(self.targets):
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
            # sfx: Level complete fanfare
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            reward -= 100 # Timeout penalty
            # sfx: Failure sound
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    # --- Helper Methods ---

    def _get_crate_at(self, pos):
        for crate in self.crates:
            if tuple(crate['pos']) == pos:
                return crate
        return None

    def _get_crates_on_target_count(self):
        count = 0
        crate_positions = {tuple(c['pos']) for c in self.crates}
        for target_pos in self.targets:
            if target_pos in crate_positions:
                count += 1
        return count

    def _get_min_dist_to_target(self, pos):
        if not self.targets: return 0
        return min(math.hypot(pos[0] - t[0], pos[1] - t[1]) for t in self.targets)

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    # --- Rendering Methods ---

    def _render_iso_cube(self, surface, vis_pos, top_color, side_color):
        x, y = vis_pos
        iso_x, iso_y = self._grid_to_iso(x, y)
        
        # Shadow
        shadow_center_x, shadow_center_y = self._grid_to_iso(x, y)
        shadow_rect = pygame.Rect(0, 0, self.TILE_WIDTH_HALF * 1.8, self.TILE_HEIGHT_HALF * 1.8)
        shadow_rect.center = (shadow_center_x, shadow_center_y + self.TILE_Z - 2)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
        surface.blit(shadow_surf, shadow_rect.topleft)

        # Cube points
        top_points = [
            (iso_x, iso_y - self.TILE_Z),
            (iso_x + self.TILE_WIDTH_HALF, iso_y - self.TILE_Z + self.TILE_HEIGHT_HALF),
            (iso_x, iso_y - self.TILE_Z + self.TILE_HEIGHT_HALF * 2),
            (iso_x - self.TILE_WIDTH_HALF, iso_y - self.TILE_Z + self.TILE_HEIGHT_HALF),
        ]
        side_1_points = [
            top_points[3], top_points[2],
            (top_points[2][0], top_points[2][1] + self.TILE_Z),
            (top_points[3][0], top_points[3][1] + self.TILE_Z),
        ]
        side_2_points = [
            top_points[2], top_points[1],
            (top_points[1][0], top_points[1][1] + self.TILE_Z),
            (top_points[2][0], top_points[2][1] + self.TILE_Z),
        ]
        
        pygame.gfxdraw.filled_polygon(surface, side_1_points, side_color)
        pygame.gfxdraw.aapolygon(surface, side_1_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_2_points, side_color)
        pygame.gfxdraw.aapolygon(surface, side_2_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

    def _render_game(self):
        # Render floor and targets
        crate_positions = {tuple(c['pos']) for c in self.crates}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                iso_x, iso_y = self._grid_to_iso(c, r)
                points = [
                    (iso_x, iso_y),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF * 2),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
                ]
                
                is_wall = (c, r) in self.walls
                is_target = (c, r) in self.targets
                is_target_active = is_target and (c,r) in crate_positions
                
                if not is_wall:
                    color = self.COLOR_FLOOR
                    if is_target:
                        color = self.COLOR_TARGET_ACTIVE if is_target_active else self.COLOR_TARGET
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_FLOOR_SHADOW)

        # Sort dynamic objects for correct render order (back to front)
        sorted_renderables = sorted(self.renderables, key=lambda e: e['vis_pos'][0] + e['vis_pos'][1])

        # Render walls and dynamic objects
        render_queue = sorted(
            [(c,r) for c,r in self.walls] + [tuple(e['vis_pos']) for e in sorted_renderables],
            key=lambda p: p[0] + p[1]
        )

        obj_idx = 0
        for x, y in render_queue:
            if (int(round(x)), int(round(y))) in self.walls:
                 self._render_iso_cube(self.screen, (x, y), self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE)
            else:
                if obj_idx < len(sorted_renderables):
                    obj = sorted_renderables[obj_idx]
                    if obj['type'] == 'player':
                        self._render_iso_cube(self.screen, obj['vis_pos'], self.COLOR_PLAYER_TOP, self.COLOR_PLAYER_SIDE)
                    elif obj['type'] == 'crate':
                        self._render_iso_cube(self.screen, obj['vis_pos'], self.COLOR_CRATE_TOP, self.COLOR_CRATE_SIDE)
                    obj_idx += 1

    def _render_ui(self):
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"{time_left:.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        
        ui_bg_rect = timer_rect.inflate(20, 10)
        ui_bg_surf = pygame.Surface(ui_bg_rect.size, pygame.SRCALPHA)
        ui_bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surf, ui_bg_rect)
        self.screen.blit(timer_surf, timer_rect)

        # Crates on target
        crates_count = self._get_crates_on_target_count()
        total_crates = len(self.targets)
        crates_text = f"Crates: {crates_count} / {total_crates}"
        crates_surf = self.font_small.render(crates_text, True, self.COLOR_TEXT)
        crates_rect = crates_surf.get_rect(topleft=(20, 15))
        
        ui_bg_rect_2 = crates_rect.inflate(20, 10)
        ui_bg_surf_2 = pygame.Surface(ui_bg_rect_2.size, pygame.SRCALPHA)
        ui_bg_surf_2.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surf_2, ui_bg_rect_2)
        self.screen.blit(crates_surf, crates_rect)

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "TIME UP!"
            color = self.COLOR_TARGET_ACTIVE if self.win else self.COLOR_TARGET
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            bg_rect = msg_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill(self.COLOR_UI_BG)
            
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(msg_surf, msg_rect)

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
            "time_left": max(0, self.timer / self.FPS),
            "crates_on_target": self._get_crates_on_target_count(),
        }

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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup a window to display the game
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Sokoban")
    
    terminated = False
    running = True
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Display ---
        # Pygame uses (width, height), but our obs is (height, width, 3). Transpose it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()