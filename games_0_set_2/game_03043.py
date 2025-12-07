
# Generated: 2025-08-28T06:47:47.534197
# Source Brief: brief_03043.md
# Brief Index: 3043

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through crystal types. Press Space to place a crystal."
    )

    game_description = (
        "Redirect a laser beam through a crystal-filled cavern to hit the target. You have a limited number of moves, so place your crystals wisely!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_W, self.TILE_H = 32, 16
        self.MAX_MOVES = 10
        self.MAX_LASER_LENGTH = 50
        self.MAX_STEPS = 1000

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_crystal_select = pygame.font.SysFont("Consolas", 14, bold=True)

        self.COLOR_BG = (10, 15, 25)
        self.COLOR_GRID = (25, 35, 50)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_TARGET = (255, 215, 0)
        self.COLOR_ORIGIN = (220, 220, 255)
        self.COLOR_LASER = (255, 20, 50)
        self.COLOR_LASER_GLOW = (180, 0, 30)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PLACEMENT_SPOT = (40, 60, 80)
        self.COLOR_TEXT = (230, 230, 230)

        self.crystal_types = [
            {'name': 'Mirror /', 'color': (200, 80, 255), 'shape': 'slash'},
            {'name': 'Mirror \\', 'color': (80, 200, 255), 'shape': 'backslash'},
            {'name': 'Splitter', 'color': (255, 255, 80), 'shape': 'tee'},
            {'name': 'Filter', 'color': (80, 255, 80), 'shape': 'diamond'},
            {'name': 'Blocker', 'color': (180, 180, 180), 'shape': 'square'},
        ]

        self.iso_offset_x = self.WIDTH // 2
        self.iso_offset_y = self.HEIGHT // 2 - 80

        self.grid = None
        self.cursor_pos = None
        self.laser_origin = None
        self.target_pos = None
        self.valid_placements = None
        self.selected_crystal_type = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.laser_path = []
        self.laser_particles = []
        self.last_min_dist_to_target = float('inf')
        self.last_space_state = False
        self.last_shift_state = False
        self.np_random = None

        self.reset()
        # self.validate_implementation() # For testing during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_level()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.selected_crystal_type = 0
        self.cursor_pos = self.valid_placements[0]
        self.last_space_state = False
        self.last_shift_state = False
        self.laser_particles = []

        self._calculate_laser_path()
        self.last_min_dist_to_target = self._get_min_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = self.game_over

        space_pressed = space_held and not self.last_space_state
        shift_pressed = shift_held and not self.last_shift_state
        self.last_space_state = space_held
        self.last_shift_state = shift_held

        if not terminated:
            if shift_pressed:
                self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.crystal_types)
                # sfx: ui_cycle
            
            self._handle_movement(movement)

            if space_pressed and self.cursor_pos in self.valid_placements and self.grid[self.cursor_pos] == 0:
                # This constitutes a "move"
                self.grid[self.cursor_pos] = self.selected_crystal_type + 2  # 0=empty, 1=wall, 2+=crystals
                self.moves_left -= 1
                # sfx: place_crystal

                hit_target, hit_crystal, did_split = self._calculate_laser_path()
                
                # Calculate rewards for this move
                current_min_dist = self._get_min_dist_to_target()
                dist_improvement = self.last_min_dist_to_target - current_min_dist
                reward += dist_improvement * 0.5 # Continuous reward for getting closer
                self.last_min_dist_to_target = current_min_dist
                
                if hit_crystal: reward += 5
                if did_split: reward += 10
                
                self.score += reward

                # Check termination conditions
                if hit_target:
                    terminated = True
                    win_reward = 100
                    self.score += win_reward
                    reward += win_reward
                    # sfx: victory
                elif self.moves_left <= 0:
                    terminated = True
                    loss_penalty = -50
                    self.score += loss_penalty
                    reward += loss_penalty
                    # sfx: failure

        self.game_over = terminated
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Create walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Fixed level design for quality and predictability
        self.laser_origin = (1, self.GRID_HEIGHT // 2)
        self.target_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        
        self.grid[5, 4:8] = 1
        self.grid[10, 2:6] = 1
        self.grid[15, 5:9] = 1

        self.valid_placements = [
            (3, 3), (3, 9),
            (7, 2), (7, 9),
            (12, 4), (12, 7),
            (17, 3), (17, 8),
        ]

    def _handle_movement(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        x = np.clip(x, 0, self.GRID_WIDTH - 1)
        y = np.clip(y, 0, self.GRID_HEIGHT - 1)
        
        if self.grid[x, y] != 1: # Can't move cursor into walls
            self.cursor_pos = (x, y)

    def _calculate_laser_path(self):
        self.laser_path = []
        beams = [(self.laser_origin, (1, 0))]  # Start with one beam moving right
        
        hit_target = False
        hit_crystal = False
        did_split = False
        
        processed_beams = 0
        while beams and processed_beams < 20: # Safety break
            processed_beams += 1
            pos, direction = beams.pop(0)
            
            path_segment = [pos]
            for _ in range(self.MAX_LASER_LENGTH):
                pos = (pos[0] + direction[0], pos[1] + direction[1])
                path_segment.append(pos)
                
                if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                    break # Out of bounds
                
                if pos == self.target_pos:
                    hit_target = True
                    break

                cell_content = self.grid[pos]
                if cell_content == 1: # Wall
                    break
                
                if cell_content >= 2: # Crystal
                    hit_crystal = True
                    crystal_type = self.crystal_types[cell_content - 2]
                    dx, dy = direction
                    
                    if crystal_type['shape'] == 'slash': # /
                        beams.append((pos, (-dy, -dx)))
                    elif crystal_type['shape'] == 'backslash': # \
                        beams.append((pos, (dy, dx)))
                    elif crystal_type['shape'] == 'tee':
                        did_split = True
                        beams.append((pos, (dy, dx)))
                        beams.append((pos, (-dy, -dx)))
                    elif crystal_type['shape'] == 'diamond': # filter
                        beams.append((pos, (dx, dy))) # Pass-through
                    elif crystal_type['shape'] == 'square': # blocker
                        pass # Absorb
                    break # Stop current segment
            
            self.laser_path.append(path_segment)
        return hit_target, hit_crystal, did_split

    def _get_min_dist_to_target(self):
        if not self.laser_path:
            return math.dist(self.laser_origin, self.target_pos)
        
        min_dist = float('inf')
        for segment in self.laser_path:
            for i in range(len(segment) - 1):
                p1 = np.array(segment[i])
                p2 = np.array(segment[i+1])
                pt = np.array(self.target_pos)
                
                l2 = np.sum((p1 - p2)**2)
                if l2 == 0.0:
                    dist = np.linalg.norm(pt - p1)
                else:
                    t = max(0, min(1, np.dot(pt - p1, p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    dist = np.linalg.norm(pt - projection)
                min_dist = min(min_dist, dist)
        return min_dist

    def _iso_transform(self, x, y):
        screen_x = self.iso_offset_x + (x - y) * self.TILE_W
        screen_y = self.iso_offset_y + (x + y) * self.TILE_H
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid, walls, and placement spots
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._iso_transform(x, y)
                tile_type = self.grid[x, y]
                
                tile_points = [
                    self._iso_transform(x, y + 1),
                    self._iso_transform(x + 1, y + 1),
                    self._iso_transform(x + 1, y),
                    (screen_pos[0] + self.TILE_W, screen_pos[1]),
                ]

                if tile_type == 1: # Wall
                    top_points = [
                        (screen_pos[0], screen_pos[1] + self.TILE_H),
                        (screen_pos[0] + self.TILE_W, screen_pos[1]),
                        (screen_pos[0] + 2*self.TILE_W, screen_pos[1] + self.TILE_H),
                        (screen_pos[0] + self.TILE_W, screen_pos[1] + 2*self.TILE_H)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_WALL)
                    pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_WALL)
                elif (x,y) in self.valid_placements:
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, self.COLOR_PLACEMENT_SPOT)
        
        # Render origin and target
        self._render_iso_object(self.laser_origin, self.COLOR_ORIGIN, 'rect')
        self._render_iso_object(self.target_pos, self.COLOR_TARGET, 'circle')
        
        # Render placed crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] >= 2:
                    self._render_crystal((x, y), self.grid[x, y] - 2)

        # Render laser
        self._render_laser()
        
        # Render cursor
        if self.cursor_pos:
            cx, cy = self.cursor_pos
            cursor_points = [
                self._iso_transform(cx, cy + 1),
                self._iso_transform(cx + 1, cy + 1),
                self._iso_transform(cx + 1, cy),
                self._iso_transform(cx, cy),
            ]
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, cursor_points, 2)

    def _render_iso_object(self, pos, color, shape):
        screen_pos = self._iso_transform(pos[0], pos[1])
        center_x = screen_pos[0] + self.TILE_W
        center_y = screen_pos[1] + self.TILE_H
        if shape == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 10, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 10, color)
        elif shape == 'rect':
            rect = pygame.Rect(center_x - 8, center_y - 8, 16, 16)
            pygame.draw.rect(self.screen, color, rect)

    def _render_crystal(self, pos, type_idx):
        crystal = self.crystal_types[type_idx]
        base_pos = self._iso_transform(pos[0], pos[1])
        center_x = base_pos[0] + self.TILE_W
        center_y = base_pos[1] + self.TILE_H
        
        w, h = self.TILE_W, self.TILE_H
        points = []
        if crystal['shape'] == 'slash':
            points = [(center_x - w*0.4, center_y + h*0.4), (center_x + w*0.4, center_y - h*0.4)]
            pygame.draw.line(self.screen, crystal['color'], points[0], points[1], 5)
        elif crystal['shape'] == 'backslash':
            points = [(center_x - w*0.4, center_y - h*0.4), (center_x + w*0.4, center_y + h*0.4)]
            pygame.draw.line(self.screen, crystal['color'], points[0], points[1], 5)
        elif crystal['shape'] == 'tee':
            points = [(center_x, center_y - h*0.5), (center_x, center_y + h*0.5), (center_x, center_y), (center_x - w*0.5, center_y), (center_x + w*0.5, center_y)]
            pygame.draw.line(self.screen, crystal['color'], points[0], points[1], 4)
            pygame.draw.line(self.screen, crystal['color'], points[3], points[4], 4)
        elif crystal['shape'] == 'diamond':
            points = [(center_x, center_y - h*0.6), (center_x + w*0.6, center_y), (center_x, center_y + h*0.6), (center_x - w*0.6, center_y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, crystal['color'])
            pygame.gfxdraw.aapolygon(self.screen, points, crystal['color'])
        elif crystal['shape'] == 'square':
            rect = pygame.Rect(center_x - 8, center_y - 8, 16, 16)
            pygame.draw.rect(self.screen, crystal['color'], rect)

    def _render_laser(self):
        # Pulsing effect for laser
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        glow_alpha = int(100 + pulse * 100)
        
        # Update and draw particles
        self.laser_particles = [p for p in self.laser_particles if p[2] > 0]
        for p in self.laser_particles:
            p[0] += p[3] * 0.1 # move
            p[1] += p[4] * 0.1
            p[2] -= 1 # fade
            size = int(p[2] / 10 * 3)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, self.COLOR_LASER_GLOW)
        
        for segment in self.laser_path:
            if len(segment) < 2: continue
            
            iso_segment = []
            for point in segment:
                p = self._iso_transform(point[0], point[1])
                iso_segment.append((p[0] + self.TILE_W, p[1] + self.TILE_H))

            if len(iso_segment) > 1:
                # Add new particles
                if self.np_random.random() < 0.5:
                    idx = self.np_random.integers(0, len(iso_segment)-1)
                    p1, p2 = iso_segment[idx], iso_segment[idx+1]
                    dir_x, dir_y = p2[0]-p1[0], p2[1]-p1[1]
                    self.laser_particles.append([p1[0], p1[1], 30, dir_x, dir_y]) # x, y, life, dx, dy

                # Draw glow
                pygame.draw.lines(self.screen, self.COLOR_LASER_GLOW, False, iso_segment, 5)
                # Draw core beam
                pygame.draw.lines(self.screen, self.COLOR_LASER, False, iso_segment, 2)


    def _render_ui(self):
        # Moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Crystal selection UI
        ui_panel_rect = pygame.Rect(10, self.HEIGHT - 50, self.WIDTH - 20, 40)
        pygame.draw.rect(self.screen, (20, 30, 45), ui_panel_rect, border_radius=5)
        
        for i, crystal in enumerate(self.crystal_types):
            is_selected = (i == self.selected_crystal_type)
            box_x = 20 + i * 125
            box_rect = pygame.Rect(box_x, self.HEIGHT - 45, 120, 30)
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, box_rect, 2, border_radius=3)
            
            name_text = self.font_crystal_select.render(crystal['name'], True, crystal['color'])
            self.screen.blit(name_text, (box_x + 5, self.HEIGHT - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")