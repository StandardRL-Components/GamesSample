
# Generated: 2025-08-28T04:33:09.796754
# Source Brief: brief_02357.md
# Brief Index: 2357

        
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
        "Controls: Arrow keys to select gears. Space to rotate clockwise, Shift to rotate counter-clockwise."
    )

    game_description = (
        "A steampunk puzzle. Manipulate interconnected gears to achieve the correct alignment and unlock the exit door before you run out of moves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 8, 5
        self.CELL_SIZE = 80
        self.MOVE_LIMIT = 25
        self.FPS = 30
        self.ANIMATION_FRAMES = 20 # Number of frames for a rotation animation

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_GEAR_GOLD = (212, 175, 55)
        self.COLOR_GEAR_DARK_GOLD = (160, 130, 40)
        self.COLOR_DOOR_SILVER = (192, 192, 192)
        self.COLOR_DOOR_DARK_SILVER = (140, 140, 140)
        self.COLOR_EXIT_RED = (200, 50, 50)
        self.COLOR_EXIT_DARK_RED = (150, 30, 30)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SELECT_GLOW = (100, 180, 255)

        # Initialize state variables
        self.gears = []
        self.doors = []
        self.selected_gear_idx = None
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.is_animating = False
        self.pre_animation_door_states = []
        self.last_animation_completed = False

        self.reset()
        # self.validate_implementation() # Optional self-check

    def _create_level(self):
        """Defines the puzzle layout: gears, connections, and doors."""
        self.gears = [
            {'pos': (2, 2), 'radius': 35, 'connections': [1, 2], 'angle': 0, 'target_angle': 0},
            {'pos': (2, 1), 'radius': 35, 'connections': [0], 'angle': 0, 'target_angle': 0},
            {'pos': (3, 2), 'radius': 35, 'connections': [0, 3], 'angle': 0, 'target_angle': 0},
            {'pos': (4, 2), 'radius': 35, 'connections': [2], 'angle': 0, 'target_angle': 0},
        ]
        self.doors = [
            {
                'grid_pos': (1, 1), 'control_gear_idx': 1, 'open_angle': 90, 
                'is_open': False, 'is_exit': False, 'anim_prog': 0.0
            },
            {
                'grid_pos': (5, 2), 'control_gear_idx': 3, 'open_angle': 180, 
                'is_open': False, 'is_exit': True, 'anim_prog': 0.0
            },
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = self.MOVE_LIMIT
        self.game_over = False
        self.win_state = False
        self.is_animating = False
        self.last_animation_completed = False

        self._create_level()
        self.selected_gear_idx = 0 # Start with the first gear selected
        self._update_door_states()
        self.pre_animation_door_states = [d['is_open'] for d in self.doors]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.last_animation_completed = False

        if not self.is_animating:
            self._handle_selection_movement(movement)

            rotation_dir = 0
            if space_held: rotation_dir = 1
            elif shift_held: rotation_dir = -1

            if rotation_dir != 0 and self.selected_gear_idx is not None:
                # A move is made
                self.moves_left -= 1
                reward -= 0.2
                self.is_animating = True
                
                # Store door states before animation for reward calculation
                self.pre_animation_door_states = [d['is_open'] for d in self.doors]
                
                self._propagate_rotation(self.selected_gear_idx, rotation_dir)
                # sfx: mechanical_clunk.wav, gear_turn_start.wav

        self._update_animation()

        if not self.is_animating and self.pre_animation_door_states is not None:
             new_door_states = [d['is_open'] for d in self.doors]
             for i, new_state in enumerate(new_door_states):
                 if new_state and not self.pre_animation_door_states[i]:
                     reward += 1.0 # Reward for opening a door
                     # sfx: door_unlock.wav
                     if self.doors[i]['is_exit']:
                         reward += 5.0 # Extra reward for opening the exit
                         # sfx: final_door_unlock.wav
             self.pre_animation_door_states = None # Consume the reward event
             self.last_animation_completed = True

        terminated = self.moves_left <= 0 or self._check_win_condition()
        if terminated and not self.game_over:
            self.game_over = True
            if self._check_win_condition():
                self.win_state = True
                reward += 100
                # sfx: level_win.wav
            else:
                self.win_state = False
                reward -= 100
                # sfx: level_lose.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_selection_movement(self, movement):
        if movement == 0 or not self.gears:
            return

        if self.selected_gear_idx is None:
            self.selected_gear_idx = 0
            return

        current_gear = self.gears[self.selected_gear_idx]
        current_pos = np.array(current_gear['pos'])
        
        best_candidate = -1
        min_dist = float('inf')

        direction_vectors = {
            1: np.array([0, -1]), # Up
            2: np.array([0, 1]),  # Down
            3: np.array([-1, 0]), # Left
            4: np.array([1, 0]),  # Right
        }
        move_vec = direction_vectors.get(movement)

        for i, gear in enumerate(self.gears):
            if i == self.selected_gear_idx:
                continue
            
            target_pos = np.array(gear['pos'])
            direction_to_target = target_pos - current_pos
            
            # Check if target is generally in the right direction
            if np.dot(direction_to_target, move_vec) > 0:
                dist = np.linalg.norm(direction_to_target)
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = i
        
        if best_candidate != -1:
            self.selected_gear_idx = best_candidate
            # sfx: select_tick.wav

    def _propagate_rotation(self, start_gear_idx, direction):
        q = [(start_gear_idx, direction)]
        visited = {start_gear_idx}
        
        gear = self.gears[start_gear_idx]
        gear['target_angle'] = (gear['target_angle'] + 90 * direction) % 360

        while q:
            curr_idx, curr_dir = q.pop(0)
            for neighbor_idx in self.gears[curr_idx]['connections']:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    neighbor_gear = self.gears[neighbor_idx]
                    neighbor_dir = -curr_dir
                    neighbor_gear['target_angle'] = (neighbor_gear['target_angle'] + 90 * neighbor_dir) % 360
                    q.append((neighbor_idx, neighbor_dir))

    def _update_animation(self):
        animation_is_active = False
        for gear in self.gears:
            # Normalize angles to be within [0, 360)
            gear['angle'] %= 360
            gear['target_angle'] %= 360
            
            diff = (gear['target_angle'] - gear['angle'] + 180) % 360 - 180
            if abs(diff) > 0.1:
                animation_is_active = True
                gear['angle'] += diff * (1.0 / (self.FPS * (self.ANIMATION_FRAMES/self.FPS) * 0.5))
            else:
                gear['angle'] = gear['target_angle']

        for door in self.doors:
            target_prog = 1.0 if door['is_open'] else 0.0
            diff = target_prog - door['anim_prog']
            if abs(diff) > 0.01:
                door['anim_prog'] += diff * 0.1
            else:
                door['anim_prog'] = target_prog

        if not animation_is_active and self.is_animating:
            self.is_animating = False
            self._update_door_states() # Final update after animation
            # sfx: gear_turn_stop.wav

    def _update_door_states(self):
        for door in self.doors:
            control_gear = self.gears[door['control_gear_idx']]
            current_angle = round(control_gear['angle']) % 360
            open_angle = door['open_angle']
            door['is_open'] = current_angle == open_angle

    def _check_win_condition(self):
        exit_doors = [d for d in self.doors if d['is_exit']]
        if not exit_doors: return False
        return all(d['is_open'] for d in exit_doors)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left}

    def _render_game(self):
        self._draw_background_elements()
        self._draw_doors()
        self._draw_gears()
        self._draw_selection_highlight()

    def _draw_background_elements(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw faint background gears for atmosphere
        bg_gear_positions = [(100, 100), (500, 300), (300, 50), (550, 80)]
        for i, pos in enumerate(bg_gear_positions):
            self._draw_gear_shape(pos, 80, (pygame.time.get_ticks() / (100.0 + i*20)) % 360, (35, 40, 45), (30,35,40), 16)

    def _draw_gears(self):
        for gear in self.gears:
            px, py = self._grid_to_pixel(gear['pos'])
            self._draw_gear_shape((px, py), gear['radius'], gear['angle'], self.COLOR_GEAR_GOLD, self.COLOR_GEAR_DARK_GOLD)

    def _draw_gear_shape(self, center, radius, angle, color1, color2, num_teeth=12):
        cx, cy = int(center[0]), int(center[1])
        
        # Spokes
        for i in range(num_teeth // 2):
            rad = math.radians(angle + i * (360 / (num_teeth // 2)))
            end_x = cx + (radius - 5) * math.cos(rad)
            end_y = cy + (radius - 5) * math.sin(rad)
            pygame.draw.line(self.screen, color2, (cx, cy), (int(end_x), int(end_y)), 3)
            
        # Main body
        pygame.gfxdraw.aacircle(self.screen, cx, cy, int(radius * 0.8), color1)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(radius * 0.8), color1)
        
        # Teeth
        for i in range(num_teeth):
            rad1 = math.radians(angle + i * (360 / num_teeth))
            rad2 = math.radians(angle + (i + 0.5) * (360 / num_teeth))
            
            p1 = (cx + radius * 0.7 * math.cos(rad1), cy + radius * 0.7 * math.sin(rad1))
            p2 = (cx + radius * math.cos(rad1), cy + radius * math.sin(rad1))
            p3 = (cx + radius * math.cos(rad2), cy + radius * math.sin(rad2))
            p4 = (cx + radius * 0.7 * math.cos(rad2), cy + radius * 0.7 * math.sin(rad2))
            
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color1)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color1)
        
        # Center pin
        pygame.gfxdraw.aacircle(self.screen, cx, cy, 5, color2)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 5, color2)

    def _draw_doors(self):
        for door in self.doors:
            px, py = self._grid_to_pixel(door['grid_pos'])
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            
            color = self.COLOR_EXIT_RED if door['is_exit'] else self.COLOR_DOOR_SILVER
            dark_color = self.COLOR_EXIT_DARK_RED if door['is_exit'] else self.COLOR_DOOR_DARK_SILVER

            # Animate door sliding open
            slide_offset = int(self.CELL_SIZE * door['anim_prog'])
            
            # Left half
            left_rect = rect.copy()
            left_rect.width = self.CELL_SIZE // 2
            left_rect.x -= slide_offset
            pygame.draw.rect(self.screen, color, left_rect)
            pygame.draw.rect(self.screen, dark_color, left_rect, 3)

            # Right half
            right_rect = rect.copy()
            right_rect.width = self.CELL_SIZE // 2
            right_rect.x = rect.x + self.CELL_SIZE // 2
            right_rect.x += slide_offset
            pygame.draw.rect(self.screen, color, right_rect)
            pygame.draw.rect(self.screen, dark_color, right_rect, 3)

    def _draw_selection_highlight(self):
        if self.selected_gear_idx is not None and not self.is_animating:
            gear = self.gears[self.selected_gear_idx]
            px, py = self._grid_to_pixel(gear['pos'])
            
            # Pulsing glow effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
            radius = int(gear['radius'] + 5 + pulse * 5)
            alpha = int(100 + pulse * 50)
            
            # Create a temporary surface for transparency
            glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(glow_surf, radius, radius, radius - 2, self.COLOR_SELECT_GLOW + (0,))
            pygame.gfxdraw.filled_circle(glow_surf, radius, radius, radius - 2, self.COLOR_SELECT_GLOW + (alpha,))
            self.screen.blit(glow_surf, (px - radius, py - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        self._draw_move_counter()

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "PUZZLE SOLVED" if self.win_state else "OUT OF MOVES"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_move_counter(self):
        cx, cy = self.WIDTH - 60, 60
        radius = 40
        
        # Dial background
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_GRID)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_GRID)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius-2, self.COLOR_BG)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius-2, self.COLOR_BG)

        # Dial Ticks
        for i in range(self.MOVE_LIMIT + 1):
            if i % 5 == 0:
                angle = math.radians(-135 + (i / self.MOVE_LIMIT) * 270)
                start = (cx + (radius-12) * math.cos(angle), cy + (radius-12) * math.sin(angle))
                end = (cx + (radius-5) * math.cos(angle), cy + (radius-5) * math.sin(angle))
                pygame.draw.line(self.screen, self.COLOR_TEXT, start, end, 2)

        # Needle
        progress = max(0, self.moves_left / self.MOVE_LIMIT)
        needle_angle = math.radians(-135 + progress * 270)
        end_x = cx + (radius - 8) * math.cos(needle_angle)
        end_y = cy + (radius - 8) * math.sin(needle_angle)
        pygame.draw.line(self.screen, self.COLOR_EXIT_RED, (cx, cy), (end_x, end_y), 3)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 4, self.COLOR_EXIT_RED)
        
        # Text
        text_surf = self.font_small.render(f"{self.moves_left}", True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(cx, cy + 15))
        self.screen.blit(text_surf, text_rect)
        
        label_surf = self.font_small.render("MOVES", True, self.COLOR_TEXT)
        label_rect = label_surf.get_rect(center=(cx, cy - 15))
        self.screen.blit(label_surf, label_rect)

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
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