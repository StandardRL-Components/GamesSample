import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:45:35.625838
# Source Brief: brief_00672.md
# Brief Index: 672
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    HexaPulse: A strategic tile-matching game within a shrinking circle.

    The player controls a cursor on a hexagonal grid, aiming to place colored tiles
    to form matches of three or more. The play area is confined by a pulsating,
    shrinking energy field. Successful matches score points and replenish the player's
    tile supply. The game rewards strategic placement and quick thinking under the
    pressure of the collapsing field.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Cursor Movement (0: None, 1: Up-Right, 2: Down-Left, 3: Up-Left, 4: Down-Right)
                 Note: To create 6-directional hex movement, we use combinations.
                 This is a simplified 4-dir movement on the axial grid.
    - action[1]: Place Tile (0: Released, 1: Held/Pressed)
    - action[2]: Toggle Tile Type (0: Released, 1: Held/Pressed)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place colored hexagonal tiles to form matches of three or more. The play area "
        "shrinks over time, so think fast and build strategically to maximize your score."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to place a tile and shift to "
        "switch between tile colors."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
    HEX_SIZE = 18
    MAX_STEPS = 5000
    WIN_SCORE = 1000

    # Colors
    COLOR_BG_START = (10, 5, 25)
    COLOR_BG_END = (30, 15, 60)
    COLOR_CIRCLE = (255, 20, 60)
    COLOR_TILE_A = (0, 150, 255)
    COLOR_TILE_B = (255, 120, 0)
    COLOR_GHOST_VALID = (255, 255, 255)
    COLOR_GHOST_INVALID = (100, 100, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_m = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_l = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Axial directions for hex grid movement
        self.axial_directions = [
            (0, 0),  # None
            (1, 0),  # Right
            (-1, 0), # Left
            (0, -1), # Up-Right
            (0, 1),  # Down-Left
            # Note: The 5-action space is mapped to a subset of 6-dir hex movement
            # 1: Up, 2: Down, 3: Left, 4: Right in brief -> mapping to axial
            # For simplicity, we map to axial directions. 1->(1, -1), 2->(-1, 1), 3->(-1,0), 4->(1,0)
        ]
        self.movement_map = [
            (0, 0),   # 0: None
            (0, -1),  # 1: Up (Up-Left in axial)
            (0, 1),   # 2: Down (Down-Right in axial)
            (-1, 0),  # 3: Left
            (1, 0),   # 4: Right
        ]

        # The reset method will be called to initialize the state
        # self.reset() # This would be called by the environment wrapper
        # self.validate_implementation(self) # This is a non-standard check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.circle_max_radius = self.HEIGHT * 0.45
        self.circle_radius = self.circle_max_radius
        self.circle_shrink_rate = (self.circle_max_radius - self.HEX_SIZE) / self.MAX_STEPS

        self.oscillation_hz = 0.5
        self.initial_available_tiles = 7
        self.max_available_tiles = 7
        self.available_tiles = self.initial_available_tiles

        self.tiles = {}  # {(q, r): type_id}
        start_type = self.np_random.integers(1, 3)
        self.tiles[(0, 0)] = start_type

        self.cursor_pos = (0, 0)
        self.ghost_tile_type = 1 if start_type == 2 else 2

        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []
        self.match_effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        movement_idx, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        self._handle_input(movement_idx, shift_pressed)
        
        placement_reward, placed = self._attempt_placement(space_pressed)
        if placed:
            reward += placement_reward

        self._update_circle()
        self._update_effects()
        self._update_progression()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            else:
                reward += -100

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement_idx, shift_pressed):
        if movement_idx != 0:
            # sfx: Cursor move
            move_dq, move_dr = self.movement_map[movement_idx]
            self.cursor_pos = (self.cursor_pos[0] + move_dq, self.cursor_pos[1] + move_dr)

        if shift_pressed:
            # sfx: Toggle sound
            self.ghost_tile_type = 3 - self.ghost_tile_type  # Toggles between 1 and 2

    def _attempt_placement(self, space_pressed):
        if not space_pressed:
            return 0, False

        is_valid = self._is_valid_placement(self.cursor_pos)
        if not is_valid:
            # sfx: Error sound
            return -1, False # Penalty for invalid placement attempt

        # sfx: Tile place sound
        self.tiles[self.cursor_pos] = self.ghost_tile_type
        self.available_tiles -= 1

        matched_tiles = self._find_matches(self.cursor_pos)
        
        placement_reward = 1.0
        
        if len(matched_tiles) >= 3:
            # sfx: Match sound (scale with size)
            if len(matched_tiles) == 3:
                match_reward = 10
                score_gain = 10
            elif len(matched_tiles) == 4:
                match_reward = 20
                score_gain = 25
            else:
                match_reward = 30
                score_gain = 50
            
            placement_reward += match_reward
            self.score += score_gain

            for q, r in matched_tiles:
                pos = self._axial_to_pixel(q, r)
                tile_type = self.tiles.get((q,r))
                color = self.COLOR_TILE_A if tile_type == 1 else self.COLOR_TILE_B
                self._create_particles(pos, color, 15)
                del self.tiles[(q, r)]
            
            self.available_tiles = min(self.max_available_tiles, self.available_tiles + len(matched_tiles))

        return placement_reward, True

    def _is_valid_placement(self, pos):
        if pos in self.tiles or self.available_tiles <= 0:
            return False

        px, py = self._axial_to_pixel(pos[0], pos[1])
        dist_sq = (px - self.CENTER_X)**2 + (py - self.CENTER_Y)**2
        if dist_sq > self.circle_radius**2:
            return False

        for neighbor_pos in self._get_neighbors(pos[0], pos[1]):
            if neighbor_pos in self.tiles:
                return True
        return False

    def _find_matches(self, start_pos):
        if start_pos not in self.tiles:
            return []

        start_type = self.tiles[start_pos]
        q = deque([start_pos])
        visited = {start_pos}
        
        while q:
            current_pos = q.popleft()
            for neighbor_pos in self._get_neighbors(current_pos[0], current_pos[1]):
                if neighbor_pos not in visited and self.tiles.get(neighbor_pos) == start_type:
                    visited.add(neighbor_pos)
                    q.append(neighbor_pos)
        
        return list(visited)

    def _get_neighbors(self, q, r):
        return [
            (q + 1, r), (q - 1, r), (q, r + 1), (q, r - 1),
            (q + 1, r - 1), (q - 1, r + 1)
        ]

    def _update_circle(self):
        self.circle_max_radius = max(self.HEX_SIZE, self.circle_max_radius - self.circle_shrink_rate)
        oscillation = math.sin(self.steps * self.oscillation_hz * 2 * math.pi / 30.0) # 30 FPS assumption
        self.circle_radius = self.circle_max_radius * (0.97 + 0.03 * oscillation)

    def _update_progression(self):
        if self.score >= 500 and self.max_available_tiles == 7:
            self.max_available_tiles = 8
            self.available_tiles = min(self.max_available_tiles, self.available_tiles + 1)
        
        if self.score >= 750: self.oscillation_hz = 0.6
        elif self.score >= 500: self.oscillation_hz = 0.55
        elif self.score >= 250: self.oscillation_hz = 0.525


    def _check_termination(self):
        if self.score >= self.WIN_SCORE: return True
        if self.circle_max_radius <= self.HEX_SIZE + 2:
            if self.available_tiles <= 0:
                return True
            has_valid_move = False
            # Check all empty hexes adjacent to existing tiles
            all_neighbors = set()
            for q, r in self.tiles:
                for n_pos in self._get_neighbors(q,r):
                    if n_pos not in self.tiles:
                        all_neighbors.add(n_pos)
            
            for pos in all_neighbors:
                if self._is_valid_placement(pos):
                    has_valid_move = True
                    break
            
            if not has_valid_move:
                return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_shrinking_circle()
        self._render_tiles()
        self._render_ghost_tile()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "available_tiles": self.available_tiles}

    def _axial_to_pixel(self, q, r):
        x = self.CENTER_X + self.HEX_SIZE * (3/2 * q)
        y = self.CENTER_Y + self.HEX_SIZE * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        return x, y

    def _draw_hexagon(self, surface, color, center, size, width=0, alpha=255):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            points.append((center[0] + size * math.cos(angle), center[1] + size * math.sin(angle)))
        
        if alpha < 255:
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            if width == 0:
                pygame.gfxdraw.filled_polygon(temp_surf, [(int(p[0]), int(p[1])) for p in points], (*color, alpha))
            else: # For outlines, aalines is better
                pygame.draw.aalines(temp_surf, (*color, alpha), True, points, True)
            surface.blit(temp_surf, (0,0))
        else:
            if width == 0:
                pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_shrinking_circle(self):
        radius = int(self.circle_radius)
        center = (self.CENTER_X, self.CENTER_Y)
        for i in range(5):
            alpha = 150 - i * 30
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius + i, (*self.COLOR_CIRCLE, alpha))
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_CIRCLE)

    def _render_tiles(self):
        for (q, r), tile_type in self.tiles.items():
            pos = self._axial_to_pixel(q, r)
            color = self.COLOR_TILE_A if tile_type == 1 else self.COLOR_TILE_B
            self._draw_hexagon(self.screen, color, pos, self.HEX_SIZE * 0.95)

    def _render_ghost_tile(self):
        if self.game_over: return
        pos = self._axial_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        is_valid = self._is_valid_placement(self.cursor_pos)
        
        color = self.COLOR_GHOST_VALID if is_valid else self.COLOR_GHOST_INVALID
        alpha = 180 if is_valid else 100
        
        self._draw_hexagon(self.screen, color, pos, self.HEX_SIZE * 0.95, width=2, alpha=alpha)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 31),
                "max_life": 30,
                "color": color,
            })

    def _update_effects(self):
        # Particles
        surviving_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.98
            p["vel"][1] *= 0.98
            p["life"] -= 1
            if p["life"] > 0:
                surviving_particles.append(p)
        self.particles = surviving_particles

    def _render_effects(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(3 * life_ratio)
            if radius > 0:
                alpha = int(255 * life_ratio)
                # Using a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p["color"], alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p["pos"][0]) - radius, int(p["pos"][1]) - radius))


    def _render_ui(self):
        # Score
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Available Tiles
        tiles_text = self.font_m.render(f"TILES: {self.available_tiles}/{self.max_available_tiles}", True, self.COLOR_TEXT)
        self.screen.blit(tiles_text, (self.WIDTH - tiles_text.get_width() - 15, 10))
        
        # Ghost tile indicator
        ghost_color = self.COLOR_TILE_A if self.ghost_tile_type == 1 else self.COLOR_TILE_B
        self._draw_hexagon(self.screen, ghost_color, (self.WIDTH - 40, 60), self.HEX_SIZE * 0.7)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "FIELD COLLAPSED"
            end_text = self.font_l.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.CENTER_X, self.CENTER_Y))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # Manual play example requires a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("HexaPulse")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space = 0
        shift = 0
        
        # Use keydown for single press actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
        
        # Use get_pressed for continuous actions (movement)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()