import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Tile:
    """Helper class to represent a single tile in the game."""
    def __init__(self, pos, size, color):
        self.pos = np.array(pos, dtype=float)  # 3D position [x, y, z]
        self.size = np.array(size, dtype=float) # 3D size [width, depth, height]
        self.color = color
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.is_static = False
        
        # Shaded colors for rendering
        self.color_light = color
        self.color_side_1 = (
            max(0, color[0] - 50),
            max(0, color[1] - 50),
            max(0, color[2] - 50),
        )
        self.color_side_2 = (
            max(0, color[0] - 80),
            max(0, color[1] - 80),
            max(0, color[2] - 80),
        )
        
    @property
    def x(self): return self.pos[0]
    @property
    def y(self): return self.pos[1]
    @property
    def z(self): return self.pos[2]

    @property
    def w(self): return self.size[0] # width
    @property
    def d(self): return self.size[1] # depth
    @property
    def h(self): return self.size[2] # height

    def get_bounds(self):
        return (self.x, self.y, self.z, self.x + self.w, self.y + self.d, self.z + self.h)

    def collides_with(self, other):
        """Check for 3D AABB collision."""
        ax1, ay1, _, ax2, ay2, _ = self.get_bounds()
        bx1, by1, _, bx2, by2, _ = other.get_bounds()
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the tile. Press space to drop it."
    )

    game_description = (
        "Build the tallest, most stable tower you can by strategically stacking falling tiles in an isometric 2D world."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TARGET_HEIGHT = 25
    MAX_STEPS = 1000
    
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_GRID = (30, 40, 60)
    COLOR_TARGET_LINE = (255, 100, 100, 150)
    TILE_COLORS = [
        (66, 135, 245), (245, 66, 66), (66, 245, 135), 
        (245, 227, 66), (168, 66, 245), (245, 135, 66)
    ]

    # Isometric projection constants
    ISO_SCALE = 18
    ISO_ANGLE = math.pi / 6 # 30 degrees
    ISO_COS = math.cos(ISO_ANGLE)
    ISO_SIN = math.sin(ISO_ANGLE)
    
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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        self.iso_origin = (self.SCREEN_WIDTH // 2, 100)
        
        # This is called in the original code, but we will call it after reset()
        # to ensure all variables are initialized before validation.
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.fall_speed = 0.05
        self.tower_height = 0
        self.stability = 1.0

        # Game entities
        self.tower_tiles = []
        self.falling_pieces = []
        self.particles = []
        
        # Create base
        base_size = (10, 10, 1)
        base_pos = (-base_size[0] / 2, -base_size[1] / 2, -base_size[2])
        self.base_tile = Tile(base_pos, base_size, (50, 60, 80))
        self.base_tile.is_static = True
        self.tower_tiles.append(self.base_tile)
        
        self._spawn_new_tile()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            # --- Action Handling ---
            movement, space_held, _ = action
            self._handle_input(movement, space_held)

            # --- Game Logic ---
            self._update_falling_tile()
            self._update_falling_pieces()
            self._update_particles()
            
            # --- Difficulty Scaling ---
            if self.steps > 0 and self.steps % 50 == 0:
                self.fall_speed = min(0.2, self.fall_speed + 0.01)

            # --- Check for placement ---
            placed_tile, support_info = self._check_placement()
            if placed_tile:
                # --- Reward for placement ---
                reward += 0.1
                # Reward for overhang
                if support_info.get('overhang', 0) > 0.4: reward += 1.0
                
                # Calculate stability impact
                old_stability = self.stability
                self._update_stability()
                if self.stability < old_stability - 0.2: reward -= 2.0
                
                self._check_collapse()
                self._update_tower_height()
                self._spawn_new_tile()
                
        self.steps += 1
        
        # --- Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.tower_height >= self.TARGET_HEIGHT and not self.win:
            self.win = True
            self.game_over = True
            terminated = True
            reward = 100.0 # Goal-oriented reward
        elif self.game_over and not self.win:
            reward = -10.0 # Collapse penalty

        # The final reward needs to be a standard Python float.
        final_reward = float(np.clip(reward, -100, 100))

        return (
            self._get_observation(),
            final_reward,
            terminated,
            False, # Truncated is always False in this environment
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if not self.falling_tile: return
        
        move_speed = 0.2
        if movement == 1: self.falling_tile.pos[1] -= move_speed # Up -> Backward
        elif movement == 2: self.falling_tile.pos[1] += move_speed # Down -> Forward
        elif movement == 3: self.falling_tile.pos[0] -= move_speed # Left
        elif movement == 4: self.falling_tile.pos[0] += move_speed # Right

        # Clamp position
        play_area = 4.0
        self.falling_tile.pos[0] = np.clip(self.falling_tile.pos[0], -play_area, play_area)
        self.falling_tile.pos[1] = np.clip(self.falling_tile.pos[1], -play_area, play_area)

        if space_held:
            self.falling_tile.velocity[2] = -2.0 # Fast drop

    def _update_falling_tile(self):
        if not self.falling_tile: return
        self.falling_tile.velocity[2] = max(self.falling_tile.velocity[2], -self.fall_speed)
        self.falling_tile.pos += self.falling_tile.velocity

    def _update_falling_pieces(self):
        gravity = -0.01
        pieces_to_remove = []
        for piece in self.falling_pieces:
            piece.velocity[2] += gravity
            piece.pos += piece.velocity
            if piece.pos[2] < -5: # Fallen off screen
                pieces_to_remove.append(piece)
                if not self.win:
                    self.game_over = True
        for piece in pieces_to_remove:
            self.falling_pieces.remove(piece)


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'][2] -= 0.01 # Gravity on particles

    def _check_placement(self):
        if not self.falling_tile: return None, {}
        
        ft = self.falling_tile
        highest_support = None
        max_z = -float('inf')

        for tile in self.tower_tiles:
            if ft.collides_with(tile):
                support_z = tile.z + tile.h
                if support_z > max_z:
                    max_z = support_z
                    highest_support = tile
        
        if ft.z <= max_z:
            ft.pos[2] = max_z
            ft.is_static = True
            self.tower_tiles.append(ft)
            self.falling_tile = None
            self._create_particles(ft.pos + [ft.w/2, ft.d/2, 0], ft.color)
            
            # Calculate overhang for reward
            overhang = 0
            if highest_support:
                dx = abs((ft.x + ft.w/2) - (highest_support.x + highest_support.w/2))
                dy = abs((ft.y + ft.d/2) - (highest_support.y + highest_support.d/2))
                overhang = max(dx / ft.w, dy / ft.d)

            return ft, {'overhang': overhang}
            
        return None, {}

    def _spawn_new_tile(self):
        tile_size = (1.5, 1.5, 1)
        spawn_x = self.np_random.uniform(-2, 2)
        spawn_y = self.np_random.uniform(-2, 2)
        color_idx = self.np_random.integers(len(self.TILE_COLORS))
        color = self.TILE_COLORS[color_idx]
        
        self.falling_tile = Tile(
            pos=[spawn_x, spawn_y, self.TARGET_HEIGHT * 0.8 + 5],
            size=tile_size,
            color=color
        )

    def _update_tower_height(self):
        if not self.tower_tiles:
            self.tower_height = 0
            return
        max_z = 0
        for tile in self.tower_tiles:
            if tile.is_static:
                max_z = max(max_z, tile.z + tile.h)
        self.tower_height = max_z

    def _update_stability(self):
        if len(self.tower_tiles) <= 1:
            self.stability = 1.0
            return
        
        total_supported_area = 0
        total_area = 0
        
        # Check support for each tile above the base
        for tile in self.tower_tiles[1:]:
            tile_area = tile.w * tile.d
            total_area += tile_area
            
            supported_area = 0
            for support in self.tower_tiles:
                if support is tile: continue
                # Check if 'support' is directly beneath 'tile'
                if abs((support.z + support.h) - tile.z) < 0.1:
                    # Calculate intersection area
                    x_overlap = max(0, min(tile.x + tile.w, support.x + support.w) - max(tile.x, support.x))
                    y_overlap = max(0, min(tile.y + tile.d, support.y + support.d) - max(tile.y, support.y))
                    supported_area += x_overlap * y_overlap
            
            total_supported_area += min(supported_area, tile_area)
        
        if total_area > 0:
            self.stability = total_supported_area / total_area
        else:
            self.stability = 1.0

    def _check_collapse(self):
        stable_tiles = {self.base_tile}
        
        for _ in range(len(self.tower_tiles)): # Iterate enough times for support to propagate up
            newly_stabilized = set()
            for tile in self.tower_tiles:
                if tile in stable_tiles: continue
                
                is_supported = False
                for support in stable_tiles:
                    if abs((support.z + support.h) - tile.z) < 0.1 and tile.collides_with(support):
                        is_supported = True
                        break
                if is_supported:
                    newly_stabilized.add(tile)
            
            if not newly_stabilized: break
            stable_tiles.update(newly_stabilized)
        
        unstable_tiles = set(self.tower_tiles) - stable_tiles
        if unstable_tiles:
            for tile in unstable_tiles:
                self.tower_tiles.remove(tile)
                tile.is_static = False
                tile.velocity = np.array([self.np_random.uniform(-0.1, 0.1), self.np_random.uniform(-0.1, 0.1), 0.0])
                self.falling_pieces.append(tile)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.1, 0.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed, self.np_random.uniform(0.1, 0.4)]
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array(vel, dtype=float),
                'life': self.np_random.integers(20, 40),
                'color': color,
            })
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tower_height": round(self.tower_height, 2),
            "stability": round(self.stability, 2),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _project_iso(self, x, y, z):
        """Converts 3D world coordinates to 2D screen coordinates."""
        screen_x = self.iso_origin[0] + (x - y) * self.ISO_SCALE * self.ISO_COS
        screen_y = self.iso_origin[1] + (x + y) * self.ISO_SCALE * self.ISO_SIN - z * self.ISO_SCALE
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, tile):
        """Draws a 3D isometric cube."""
        w, d, h = tile.w, tile.d, tile.h
        x, y, z = tile.pos
        
        if not tile.is_static and tile.velocity[2] == 0: 
             wobble = math.sin(self.steps * 0.3) * 0.2
             z += wobble
        
        points = [
            (x, y, z), (x + w, y, z), (x + w, y + d, z), (x, y + d, z),
            (x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h)
        ]
        screen_points = [self._project_iso(*p) for p in points]

        pygame.gfxdraw.filled_polygon(surface, [screen_points[4], screen_points[5], screen_points[6], screen_points[7]], tile.color_light)
        pygame.gfxdraw.filled_polygon(surface, [screen_points[0], screen_points[3], screen_points[7], screen_points[4]], tile.color_side_1)
        pygame.gfxdraw.filled_polygon(surface, [screen_points[0], screen_points[1], screen_points[5], screen_points[4]], tile.color_side_2)
        
        pygame.gfxdraw.aapolygon(surface, [screen_points[4], screen_points[5], screen_points[6], screen_points[7]], tile.color_light)
        pygame.gfxdraw.aapolygon(surface, [screen_points[0], screen_points[3], screen_points[7], screen_points[4]], tile.color_light)
        pygame.gfxdraw.aapolygon(surface, [screen_points[0], screen_points[1], screen_points[5], screen_points[4]], tile.color_light)

    def _render_game(self):
        for i in range(-6, 7):
            p1 = self._project_iso(i, -6, 0)
            p2 = self._project_iso(i, 6, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            p1 = self._project_iso(-6, i, 0)
            p2 = self._project_iso(6, i, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        _, y_target = self._project_iso(0, 0, self.TARGET_HEIGHT)
        target_line_surface = pygame.Surface((self.SCREEN_WIDTH, 2), pygame.SRCALPHA)
        target_line_surface.fill(self.COLOR_TARGET_LINE)
        self.screen.blit(target_line_surface, (0, y_target))

        render_queue = self.tower_tiles + self.falling_pieces
        if self.falling_tile:
            render_queue.append(self.falling_tile)
        
        render_queue.sort(key=lambda t: t.x + t.y + t.z * 2, reverse=False)

        for tile in render_queue:
            self._draw_iso_cube(self.screen, tile)
            
            if tile is self.falling_tile:
                shadow_z = 0
                for t in self.tower_tiles:
                    if tile.collides_with(t):
                        shadow_z = max(shadow_z, t.z + t.h)
                
                sw, sd = tile.w, tile.d
                sx, sy = tile.x, tile.y
                shadow_points = [
                    self._project_iso(sx, sy, shadow_z),
                    self._project_iso(sx+sw, sy, shadow_z),
                    self._project_iso(sx+sw, sy+sd, shadow_z),
                    self._project_iso(sx, sy+sd, shadow_z),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, shadow_points, (0,0,0,80))

        for p in self.particles:
            px, py = self._project_iso(p['pos'][0], p['pos'][1], p['pos'][2])
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            size = int(max(1, 3 * (p['life'] / 40.0)))
            color_with_alpha = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color_with_alpha, (px, py), size)

    def _render_ui(self):
        height_text = self.font_ui.render(f"Height: {self.tower_height:.1f} / {self.TARGET_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        stab_text = self.font_ui.render("Stability:", True, self.COLOR_UI_TEXT)
        self.screen.blit(stab_text, (10, 35))
        bar_x, bar_y, bar_w, bar_h = 110, 38, 120, 15
        pygame.draw.rect(self.screen, (80, 20, 20), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (20, 180, 20), (bar_x, bar_y, int(bar_w * self.stability), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "TOWER COLLAPSED"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()