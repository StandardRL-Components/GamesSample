
# Generated: 2025-08-28T05:46:02.179292
# Source Brief: brief_05684.md
# Brief Index: 5684

        
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
    An isometric puzzle game where the player places crystals to redirect a laser beam
    to a target. The game is turn-based, with limited resources (crystals) and a
    step limit acting as a timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: ↑↓←→ to move cursor. Space to place crystal. Shift to cycle crystal type."
    
    # Short, user-facing description of the game
    game_description = "Redirect a laser to its target by strategically placing limited-use crystals in an isometric 2D cavern."

    # Frames advance only when an action is received
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_GRID = (50, 50, 70)
    COLOR_LASER_GLOW = (233, 69, 96, 100)
    COLOR_LASER_BEAM = (233, 69, 96, 200)
    COLOR_LASER_CORE = (255, 255, 255)
    COLOR_TARGET = (240, 230, 140)
    COLOR_TARGET_GLOW = (240, 230, 140, 80)
    COLOR_CURSOR = (255, 255, 255)
    CRYSTAL_COLORS = {
        1: (0, 168, 255),    # Blue (Reflector \)
        2: (50, 255, 126),   # Green (Refractor /)
        3: (125, 95, 255),   # Purple (Retroreflector)
    }
    COLOR_FONT = (220, 221, 225)
    
    # Grid & Isometric Projection
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 12, 12
    ISO_TILE_W, ISO_TILE_H = 40, 20
    ISO_Z_HEIGHT = 10  # Visual height of crystals
    ORIGIN_X = SCREEN_W // 2
    ORIGIN_Y = 80

    # Game Mechanics
    MAX_STEPS = 1000
    INITIAL_CRYSTALS = {1: 3, 2: 3, 3: 1}
    CRYSTAL_CHARGES = 3
    MAX_LASER_BOUNCES = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Define reflection logic for each crystal type
        self.reflection_maps = {
            1: {(1, 0): (0, -1), (0, 1): (-1, 0), (-1, 0): (0, 1), (0, -1): (1, 0)},  # \ mirror
            2: {(1, 0): (0, 1), (0, -1): (-1, 0), (-1, 0): (0, -1), (0, 1): (1, 0)},  # / mirror
            3: {(1, 0): (-1, 0), (0, 1): (0, -1), (-1, 0): (1, 0), (0, -1): (0, 1)}, # retroreflector
        }

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.target_hit = False

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_crystal_type = 1
        
        self.crystal_inventory = self.INITIAL_CRYSTALS.copy()
        self.placed_crystals = {}  # {(x, y): {'type': int, 'charges': int}}

        self._generate_layout()

        self.laser_path = []
        self.laser_endpoint = self.laser_origin
        self.last_dist_to_target = self._get_dist_to_target(self.laser_endpoint)
        self._calculate_laser_path()

        self.prev_space_held = True # Prevent placing on first frame
        self.prev_shift_held = True # Prevent cycling on first frame

        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_layout(self):
        # Ensure target and laser are not trivially aligned
        self.laser_origin = (self.np_random.integers(1, self.GRID_W - 2), 0)
        self.laser_start_dir = (0, 1)
        
        self.target_pos = (self.np_random.integers(1, self.GRID_W - 2), self.GRID_H - 1)
        if self.target_pos[0] == self.laser_origin[0]:
            self.target_pos = ((self.target_pos[0] + 1) % self.GRID_W, self.target_pos[1])

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.game_over = self._check_termination()
        if self.game_over:
            if self.target_hit:
                reward += 50
            return self._get_observation(), reward, True, False, self._get_info()

        self.steps += 1
        
        # Handle actions
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        moved = self._handle_movement(movement)
        placed = self._handle_placement(space_pressed)
        cycled = self._handle_cycling(shift_pressed)

        if moved or placed or cycled:
             self._calculate_laser_path()
        
        self._update_particles()
        
        reward += self._calculate_reward()
        if placed:
            reward -= 1 # Cost for placing a crystal

        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        terminated = self._check_termination()
        if terminated and self.target_hit:
            reward += 50
            self.score += 50
            # Sound: Victory fanfare

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 0: return False
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)
        return True

    def _handle_placement(self, space_pressed):
        if not space_pressed: return False
        
        pos = tuple(self.cursor_pos)
        if pos not in self.placed_crystals and self.crystal_inventory[self.selected_crystal_type] > 0:
            self.placed_crystals[pos] = {
                'type': self.selected_crystal_type,
                'charges': self.CRYSTAL_CHARGES
            }
            self.crystal_inventory[self.selected_crystal_type] -= 1
            # Sound: Crystal placement
            self._add_particles(self._grid_to_iso(pos[0] + 0.5, pos[1] + 0.5), 15, self.CRYSTAL_COLORS[self.selected_crystal_type])
            return True
        # Sound: Action failed
        return False

    def _handle_cycling(self, shift_pressed):
        if not shift_pressed: return False
        
        self.selected_crystal_type += 1
        if self.selected_crystal_type > 3:
            self.selected_crystal_type = 1
        # Sound: UI selection change
        return True

    def _calculate_laser_path(self):
        path = [self.laser_origin]
        pos = np.array(self.laser_origin, dtype=float)
        direction = np.array(self.laser_start_dir, dtype=float)
        
        for _ in range(self.MAX_LASER_BOUNCES):
            next_pos = tuple(np.round(pos + direction).astype(int))
            
            if next_pos == self.target_pos:
                path.append(self.target_pos)
                self.target_hit = True
                self.laser_endpoint = self.target_pos
                self.laser_path = path
                # Sound: Laser hits target
                self._add_particles(self._grid_to_iso(next_pos[0] + 0.5, next_pos[1] + 0.5), 50, self.COLOR_TARGET)
                return

            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                path.append(tuple(np.round(pos).astype(int)))
                self.laser_endpoint = tuple(np.round(pos).astype(int))
                self.laser_path = path
                return

            if next_pos in self.placed_crystals:
                crystal = self.placed_crystals[next_pos]
                if crystal['charges'] > 0:
                    crystal['charges'] -= 1
                    path.append(next_pos)
                    pos = np.array(next_pos, dtype=float)
                    dir_tuple = tuple(direction.astype(int))
                    new_dir_tuple = self.reflection_maps[crystal['type']].get(dir_tuple)
                    
                    if new_dir_tuple:
                        direction = np.array(new_dir_tuple, dtype=float)
                        # Sound: Laser reflects
                        self._add_particles(self._grid_to_iso(next_pos[0] + 0.5, next_pos[1] + 0.5), 10, self.CRYSTAL_COLORS[crystal['type']])
                        continue
            
            pos += direction

        self.laser_endpoint = tuple(np.round(pos).astype(int))
        path.append(self.laser_endpoint)
        self.laser_path = path

    def _get_dist_to_target(self, pos):
        return np.linalg.norm(np.array(pos) - np.array(self.target_pos))

    def _calculate_reward(self):
        current_dist = self._get_dist_to_target(self.laser_endpoint)
        reward = (self.last_dist_to_target - current_dist) * 0.1
        self.last_dist_to_target = current_dist
        return reward

    def _check_termination(self):
        if self.target_hit:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        
        inventory_empty = sum(self.crystal_inventory.values()) == 0
        if inventory_empty:
            for crystal in self.placed_crystals.values():
                if crystal['charges'] > 0:
                    return False
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_target()
        self._render_crystals()
        self._render_laser()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * (self.ISO_TILE_W / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.ISO_TILE_H / 2)
        return int(iso_x), int(iso_y)

    def _render_grid(self):
        for y in range(self.GRID_H + 1):
            start = self._grid_to_iso(0, y)
            end = self._grid_to_iso(self.GRID_W, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_W + 1):
            start = self._grid_to_iso(x, 0)
            end = self._grid_to_iso(x, self.GRID_H)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_target(self):
        center_iso = self._grid_to_iso(self.target_pos[0] + 0.5, self.target_pos[1] + 0.5)
        radius = int(self.ISO_TILE_W / 3)
        pygame.gfxdraw.filled_circle(self.screen, center_iso[0], center_iso[1], radius, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, center_iso[0], center_iso[1], radius, self.COLOR_TARGET)
        if self.np_random.random() > 0.1:
            glow_radius = int(radius * (1.5 + self.np_random.random() * 0.5))
            pygame.gfxdraw.filled_circle(self.screen, center_iso[0], center_iso[1], glow_radius, self.COLOR_TARGET_GLOW)

    def _render_crystals(self):
        for pos, crystal in self.placed_crystals.items():
            top_point = self._grid_to_iso(pos[0], pos[1])
            right_point = self._grid_to_iso(pos[0] + 1, pos[1])
            bottom_point = self._grid_to_iso(pos[0] + 1, pos[1] + 1)
            left_point = self._grid_to_iso(pos[0], pos[1] + 1)
            
            z = self.ISO_Z_HEIGHT
            color = self.CRYSTAL_COLORS[crystal['type']]
            darker_color = tuple(max(0, c - 50) for c in color)
            
            top_face = [(top_point[0], top_point[1] - z), (right_point[0], right_point[1] - z), (bottom_point[0], bottom_point[1] - z), (left_point[0], left_point[1] - z)]
            side_face_1 = [top_point, (top_point[0], top_point[1] - z), (right_point[0], right_point[1] - z), right_point]
            side_face_2 = [right_point, (right_point[0], right_point[1] - z), (bottom_point[0], bottom_point[1] - z), bottom_point]

            pygame.gfxdraw.filled_polygon(self.screen, side_face_1, darker_color)
            pygame.gfxdraw.filled_polygon(self.screen, side_face_2, darker_color)
            
            if crystal['charges'] > 0:
                pygame.gfxdraw.filled_polygon(self.screen, top_face, color)
                pygame.gfxdraw.aapolygon(self.screen, top_face, color)
                for i in range(crystal['charges']):
                    charge_y = top_face[0][1] + (self.ISO_TILE_H/2)
                    charge_x = top_face[3][0] + (top_face[1][0] - top_face[3][0]) * (i + 1) / (self.CRYSTAL_CHARGES + 1)
                    pygame.gfxdraw.filled_circle(self.screen, int(charge_x), int(charge_y), 2, (255,255,255))
            else:
                pygame.gfxdraw.filled_polygon(self.screen, top_face, self.COLOR_GRID)
                pygame.gfxdraw.aapolygon(self.screen, top_face, self.COLOR_GRID)

    def _render_laser(self):
        if len(self.laser_path) < 2: return
        
        iso_path = [self._grid_to_iso(p[0] + 0.5, p[1] + 0.5) for p in self.laser_path]
        anim_phase = (self.steps % 10) / 10.0
        
        pygame.draw.lines(self.screen, self.COLOR_LASER_GLOW, False, iso_path, width=9 + int(math.sin(anim_phase * math.pi) * 4))
        pygame.draw.lines(self.screen, self.COLOR_LASER_BEAM, False, iso_path, width=5)
        pygame.draw.lines(self.screen, self.COLOR_LASER_CORE, False, iso_path, width=2)
        
        start_iso = self._grid_to_iso(self.laser_origin[0]+0.5, self.laser_origin[1]-0.5)
        pygame.gfxdraw.filled_circle(self.screen, start_iso[0], start_iso[1], 8, self.COLOR_LASER_BEAM)
        pygame.gfxdraw.filled_circle(self.screen, start_iso[0], start_iso[1], 4, self.COLOR_LASER_CORE)

    def _render_cursor(self):
        if self.game_over: return
        pos = self.cursor_pos
        points = [self._grid_to_iso(pos[0], pos[1]), self._grid_to_iso(pos[0] + 1, pos[1]), self._grid_to_iso(pos[0] + 1, pos[1] + 1), self._grid_to_iso(pos[0], pos[1] + 1)]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def _render_ui(self):
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_FONT)
        self.screen.blit(time_text, (10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_FONT)
        self.screen.blit(score_text, (10, 35))

        for i, c_type in enumerate(self.CRYSTAL_COLORS.keys()):
            x_pos = self.SCREEN_W - 120 + (i * 40)
            y_pos = 20
            color = self.CRYSTAL_COLORS[c_type]
            
            if c_type == self.selected_crystal_type:
                pygame.draw.rect(self.screen, (255,255,255), (x_pos - 5, y_pos - 5, 30, 30), 2, 3)

            pygame.draw.rect(self.screen, color, (x_pos, y_pos, 20, 20))
            count_text = self.font_small.render(f"{self.crystal_inventory[c_type]}", True, self.COLOR_FONT)
            self.screen.blit(count_text, (x_pos + 5, y_pos + 25))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "TARGET HIT!" if self.target_hit else "GAME OVER"
            color = self.COLOR_TARGET if self.target_hit else self.COLOR_LASER_BEAM
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(end_text, text_rect)

    def _add_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            lifetime = 20 + self.np_random.integers(0, 20)
            self.particles.append([list(pos), [dx, dy], lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1

    def _render_particles(self):
        for p in self.particles:
            pos, _, lifetime, color = p
            alpha = max(0, min(255, int(255 * (lifetime / 40))))
            radius = int(3 * (lifetime / 40))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*color, alpha))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Laser Cavern")
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    
    action = env.action_space.sample()
    action.fill(0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = np.array([movement, 1 if space_held else 0, 1 if shift_held else 0])
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)

    pygame.quit()