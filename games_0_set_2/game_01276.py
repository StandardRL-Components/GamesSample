
# Generated: 2025-08-27T16:36:32.770194
# Source Brief: brief_01276.md
# Brief Index: 1276

        
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
        "Controls: Arrow keys to move cursor. Space to place tile. Shift to cycle tile type."
    )

    game_description = (
        "Build the tallest, most stable tile tower you can in 60 seconds. "
        "Each placement affects the tower's balance. A collapse ends the game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS # 60 seconds
        self.WIN_HEIGHT = 20
        self.GRID_SIZE = 14
        self.TILE_UNIT_W, self.TILE_UNIT_H = 20, 10
        self.TILE_Z_H = 12

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_STABLE = (100, 180, 100)
        self.COLOR_UNSTABLE = (200, 200, 80)
        self.COLOR_CRITICAL = (220, 100, 100)
        self.COLOR_PLACED = (100, 110, 120)
        self.COLOR_UI_BG = (50, 55, 60, 200)

        # --- Action Space ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 36)
        self.font_huge = pygame.font.Font(None, 64)

        # --- Tile Definitions ---
        self.TILE_TYPES = [
            {'size': (4, 4), 'color': (60, 120, 220)}, # Large square
            {'size': (2, 2), 'color': (220, 120, 60)}, # Small square
            {'size': (4, 2), 'color': (60, 220, 120)}, # Rectangle
            {'size': (1, 4), 'color': (180, 60, 220)}, # Thin stick
        ]
        
        # --- Game State ---
        self.iso_origin = (self.WIDTH // 2, 100)
        self.rng = np.random.default_rng()
        self.reset()
        
        # --- Final Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.step_reward = 0.0

        self.last_space_held = False
        self.last_shift_held = False

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile_idx = 0
        
        self.particles = []
        self.tower_wobble = [0, 0]

        # State of all placed tiles: {'type': dict, 'pos': (x, y, z), 'instability': float}
        self.placed_tiles = []
        
        # Place a large, immovable base
        base_size = self.GRID_SIZE
        self.placed_tiles.append({
            'type': {'size': (base_size, base_size), 'color': (50, 55, 60)},
            'pos': (0, 0, -1), # At z=-1 so first tile is at z=0
            'instability': 0.0
        })
        
        self.tower_height = 0
        self.current_place_height = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0
        self.game_over = self.game_over or self.steps >= self.MAX_STEPS

        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held, shift_held)
            self._update_physics()
        
        self._update_particles()
        self.steps += 1
        
        reward = self.step_reward
        self.score += reward
        terminated = self.game_over or self.win
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        tile_w, tile_h = self.TILE_TYPES[self.selected_tile_idx]['size']
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - tile_w)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - tile_h)

        # --- Cycle Tile (on press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tile_idx = (self.selected_tile_idx + 1) % len(self.TILE_TYPES)
            # Recalculate cursor clamp after tile size change
            tile_w, tile_h = self.TILE_TYPES[self.selected_tile_idx]['size']
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - tile_w)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - tile_h)


        # --- Place Tile (on press) ---
        if space_held and not self.last_space_held:
            self._place_tile()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _place_tile(self):
        tile_type = self.TILE_TYPES[self.selected_tile_idx]
        tile_w, tile_d = tile_type['size']
        place_pos = (self.cursor_pos[0], self.cursor_pos[1], self.current_place_height)

        # Find supporting tiles on the layer below
        support_tiles = []
        for tile in self.placed_tiles:
            if tile['pos'][2] == self.current_place_height - 1:
                t_x, t_y, _ = tile['pos']
                t_w, t_d = tile['type']['size']
                # Check for overlap
                if not (place_pos[0] >= t_x + t_w or place_pos[0] + tile_w <= t_x or \
                        place_pos[1] >= t_y + t_d or place_pos[1] + tile_d <= t_y):
                    support_tiles.append(tile)

        if not support_tiles:
            # Cannot place in mid-air
            # sfx: invalid placement
            return

        # Simplified stability calculation
        support_com_x, support_com_y, total_mass = 0, 0, 0
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        for tile in support_tiles:
            t_x, t_y, _ = tile['pos']
            t_w, t_d = tile['type']['size']
            mass = t_w * t_d
            support_com_x += (t_x + t_w / 2) * mass
            support_com_y += (t_y + t_d / 2) * mass
            total_mass += mass
            min_x, min_y = min(min_x, t_x), min(min_y, t_y)
            max_x, max_y = max(max_x, t_x + t_w), max(max_y, t_y + t_d)
        
        if total_mass > 0:
            support_com_x /= total_mass
            support_com_y /= total_mass
        
        placed_com_x = place_pos[0] + tile_w / 2
        placed_com_y = place_pos[1] + tile_d / 2
        
        # Is the new tile's CoM within the support bounding box?
        is_supported = min_x <= placed_com_x <= max_x and min_y <= placed_com_y <= max_y
        
        if not is_supported:
            # Immediate collapse for placing off-balance
            self._trigger_collapse(place_pos)
            return

        # Calculate instability based on distance from support CoM
        dist_x = placed_com_x - support_com_x
        dist_y = placed_com_y - support_com_y
        instability = math.sqrt(dist_x**2 + dist_y**2)

        new_tile = {
            'type': tile_type,
            'pos': place_pos,
            'instability': instability
        }
        self.placed_tiles.append(new_tile)
        
        # sfx: place tile
        
        # Check overall tower stability
        total_instability = sum(t.get('instability', 0) for t in self.placed_tiles)
        collapse_threshold = self.tower_height * 1.5 + 3 # Threshold grows with height
        
        if total_instability > collapse_threshold:
            self._trigger_collapse(place_pos)
            return

        # --- Successful Placement ---
        if instability > 1.5: # Unstable placement
            self.step_reward -= 0.05
        else: # Stable placement
            self.step_reward += 0.1

        if self.current_place_height + 1 > self.tower_height:
            self.tower_height = self.current_place_height + 1
            self.step_reward += 1.0
            if self.tower_height >= self.WIN_HEIGHT:
                self.win = True
                self.step_reward += 100.0
        
        self.current_place_height += 1

    def _trigger_collapse(self, collapse_origin_pos):
        self.game_over = True
        self.step_reward -= 10.0
        # sfx: tower collapse
        
        # Create particles for each tile
        for tile in self.placed_tiles:
            if tile['pos'][2] < 0: continue # Don't explode the base
            
            iso_pos = self._to_iso(tile['pos'])
            for _ in range(20):
                self.particles.append({
                    'pos': [iso_pos[0], iso_pos[1]],
                    'vel': [self.rng.uniform(-3, 3), self.rng.uniform(-5, 1)],
                    'size': self.rng.integers(3, 8),
                    'life': self.rng.integers(20, 40),
                    'color': tile['type']['color']
                })
                
    def _update_physics(self):
        # Update tower wobble based on total instability
        total_instability = sum(t.get('instability', 0) for t in self.placed_tiles)
        wobble_amount = min(total_instability * 0.1, 5.0)
        self.tower_wobble[0] = math.sin(self.steps * 0.5) * wobble_amount
        self.tower_wobble[1] = math.cos(self.steps * 0.35) * wobble_amount * 0.5

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.3 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

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
            "height": self.tower_height,
            "win": self.win,
        }

    def _to_iso(self, pos):
        x, y, z = pos
        iso_x = self.iso_origin[0] + (x - y) * self.TILE_UNIT_W
        iso_y = self.iso_origin[1] + (x + y) * self.TILE_UNIT_H - z * self.TILE_Z_H
        return (iso_x, iso_y)

    def _render_tile(self, surface, tile_type, iso_pos, color, wobble=(0,0)):
        w, d = tile_type['size']
        w_px, d_px = w * self.TILE_UNIT_W, d * self.TILE_UNIT_W
        h_px = d * self.TILE_UNIT_H

        x, y = iso_pos[0] + wobble[0], iso_pos[1] + wobble[1]
        
        # Points for the top face
        p_top = (x, y)
        p_left = (x - w_px, y + h_px)
        p_right = (x + d_px, y + h_px)
        p_bottom = (x - w_px + d_px, y + h_px + h_px)
        
        # Points for the bottom face
        z_offset = self.TILE_Z_H
        p_top_b = (p_top[0], p_top[1] + z_offset)
        p_left_b = (p_left[0], p_left[1] + z_offset)
        p_right_b = (p_right[0], p_right[1] + z_offset)
        p_bottom_b = (p_bottom[0], p_bottom[1] + z_offset)

        # Draw faces
        # Top face
        pygame.gfxdraw.aapolygon(surface, [p_top, p_left, p_bottom, p_right], color)
        pygame.gfxdraw.filled_polygon(surface, [p_top, p_left, p_bottom, p_right], color)
        
        # Left face
        left_color = tuple(max(0, c - 30) for c in color)
        pygame.gfxdraw.aapolygon(surface, [p_left, p_left_b, p_bottom_b, p_bottom], left_color)
        pygame.gfxdraw.filled_polygon(surface, [p_left, p_left_b, p_bottom_b, p_bottom], left_color)

        # Right face
        right_color = tuple(max(0, c - 60) for c in color)
        pygame.gfxdraw.aapolygon(surface, [p_right, p_right_b, p_bottom_b, p_bottom], right_color)
        pygame.gfxdraw.filled_polygon(surface, [p_right, p_right_b, p_bottom_b, p_bottom], right_color)

    def _render_game(self):
        # --- Render Grid ---
        for i in range(self.GRID_SIZE + 1):
            start_iso = self._to_iso((i, 0, -1))
            end_iso = self._to_iso((i, self.GRID_SIZE, -1))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)
            
            start_iso = self._to_iso((0, i, -1))
            end_iso = self._to_iso((self.GRID_SIZE, i, -1))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)

        # --- Sort and Render Placed Tiles ---
        # Sort by screen y-pos for correct draw order, then by z for stability
        sorted_tiles = sorted(self.placed_tiles, key=lambda t: (t['pos'][2], t['pos'][0] + t['pos'][1]))
        
        for tile in sorted_tiles:
            iso_pos = self._to_iso(tile['pos'])
            instability = tile.get('instability', 0)
            
            if tile['pos'][2] < 0: # Base tile
                color = tile['type']['color']
                wobble = (0,0)
            else:
                color = self.COLOR_PLACED
                wobble = self.tower_wobble
            
            self._render_tile(self.screen, tile['type'], iso_pos, color, wobble)

        # --- Render Cursor/Ghost Tile ---
        if not self.game_over and not self.win:
            ghost_type = self.TILE_TYPES[self.selected_tile_idx]
            ghost_pos = (self.cursor_pos[0], self.cursor_pos[1], self.current_place_height)
            ghost_iso_pos = self._to_iso(ghost_pos)
            
            color = list(self.COLOR_CURSOR)
            
            # Change ghost color based on placement validity
            support_tiles_exist = any(
                t['pos'][2] == self.current_place_height - 1 and
                not (ghost_pos[0] >= t['pos'][0] + t['type']['size'][0] or
                     ghost_pos[0] + ghost_type['size'][0] <= t['pos'][0] or
                     ghost_pos[1] >= t['pos'][1] + t['type']['size'][1] or
                     ghost_pos[1] + ghost_type['size'][1] <= t['pos'][1])
                for t in self.placed_tiles
            )
            
            if not support_tiles_exist:
                color = list(self.COLOR_CRITICAL) + [100]
            
            self._render_tile(self.screen, ghost_type, ghost_iso_pos, color)
            
        # --- Render Particles ---
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], p['size'], p['size']))

    def _render_ui(self):
        # --- Timer Bar ---
        time_ratio = self.steps / self.MAX_STEPS
        bar_width = self.WIDTH * (1 - time_ratio)
        pygame.draw.rect(self.screen, self.COLOR_STABLE, (0, 0, bar_width, 8))
        if time_ratio > 0.5:
             pygame.draw.rect(self.screen, self.COLOR_UNSTABLE, (0, 0, bar_width, 8))
        if time_ratio > 0.8:
             pygame.draw.rect(self.screen, self.COLOR_CRITICAL, (0, 0, bar_width, 8))

        # --- Height Display ---
        height_text = self.font_large.render(f"Height: {self.tower_height} / {self.WIN_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (15, 15))

        # --- Score Display ---
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 45))

        # --- Tile Selector UI ---
        ui_box = pygame.Surface((160, 100), pygame.SRCALPHA)
        ui_box.fill(self.COLOR_UI_BG)
        
        title = self.font_small.render("Next Tile (Shift)", True, self.COLOR_TEXT)
        ui_box.blit(title, (10, 5))
        
        # Render a small preview of the selected tile
        preview_type = self.TILE_TYPES[self.selected_tile_idx]
        preview_iso = (ui_box.get_width() / 2, 45)
        self._render_tile(ui_box, preview_type, preview_iso, preview_type['color'])
        self.screen.blit(ui_box, (self.WIDTH - 170, 15))

        # --- Game Over / Win Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_huge.render("TOWER COLLAPSED", True, self.COLOR_CRITICAL)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.win:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_huge.render("TOWER COMPLETE!", True, self.COLOR_STABLE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Builder")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Environment Step ---
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Height: {info['height']}")
    env.close()
    pygame.quit()