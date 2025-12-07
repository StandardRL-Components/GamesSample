
# Generated: 2025-08-27T23:53:54.957839
# Source Brief: brief_03614.md
# Brief Index: 3614

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move cursor. Press space to select a crystal, "
        "then move to an adjacent crystal and press space again to swap. "
        "Match 3 or more to score. You have a limited number of moves."
    )

    game_description = (
        "Swap adjacent crystals to match 3 or more in this isometric puzzle game. "
        "Plan your moves to create chain reactions and clear the board before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_CRYSTAL_TYPES = 6
        self.INITIAL_MOVES = 30
        self.MAX_STEPS = 1000

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Colors ---
        self.COLOR_BG = (25, 20, 35)
        self.CRYSTAL_COLORS = [
            (255, 50, 50), (50, 255, 50), (50, 150, 255),
            (255, 255, 50), (255, 50, 255), (50, 255, 255)
        ]
        self.COLOR_UI = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 165, 0)
        
        # --- Isometric Projection ---
        self.tile_width = 32
        self.tile_height = 16
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80

        # --- State variables ---
        self.board = None
        self.cursor_pos = None
        self.selected_pos = None
        self.game_state = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

        # --- Animation state ---
        self.animations = []
        self.particles = []
        
        # Initialize state
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed Python's random and NumPy's generator for reproducibility
            random.seed(seed)
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.INITIAL_MOVES
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_pos = None
        self.game_state = "SELECT" # States: SELECT, SWAP_ANIM, INVALID_SWAP_ANIM, FALL_ANIM, RESOLVE
        self.animations = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False

        self._create_initial_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        reward = 0
        terminated = False
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # --- Animation Phase ---
        if self.game_state != "SELECT":
            self._update_animations()
            if not self.animations:
                if self.game_state == "SWAP_ANIM":
                    chain_reward, has_cleared = self._resolve_matches()
                    reward += chain_reward
                    if not has_cleared:
                        # Invalid swap, swap back
                        self._create_swap_animation(self.swap_target, self.swap_origin, is_invalid=True)
                        self.game_state = "INVALID_SWAP_ANIM"
                    else:
                        self.game_state = "RESOLVE"
                elif self.game_state == "INVALID_SWAP_ANIM":
                    self.game_state = "SELECT"
                elif self.game_state == "FALL_ANIM":
                    self.game_state = "RESOLVE"
                
                if self.game_state == "RESOLVE":
                    chain_reward, has_cleared = self._resolve_matches()
                    if has_cleared:
                        reward += chain_reward
                    else:
                        self.game_state = "SELECT"
                        if not self._find_all_valid_moves():
                            self._handle_board_shuffle()
                            reward -= 5 # Penalty for forced shuffle

        # --- Input Phase ---
        elif not self.game_over:
            # Handle cursor movement
            if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
            elif movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1
            elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
            elif movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1
            
            # Handle selection/swap
            if space_press:
                r, c = self.cursor_pos
                if self.board[r][c] == 0: # Cannot select empty space
                    pass 
                elif self.selected_pos is None:
                    self.selected_pos = list(self.cursor_pos)
                elif self.selected_pos == self.cursor_pos:
                    self.selected_pos = None # Deselect
                else: # Attempt swap
                    dist = abs(self.selected_pos[0] - r) + abs(self.selected_pos[1] - c)
                    if dist == 1:
                        self.moves_remaining -= 1
                        reward -= 0.1 # Small cost per move attempt
                        self._create_swap_animation(self.selected_pos, self.cursor_pos)
                        self.game_state = "SWAP_ANIM"
                        self.selected_pos = None
                    else: # Not adjacent
                        self.selected_pos = list(self.cursor_pos) # Select the new piece instead
            
            # Handle manual shuffle (if no moves)
            if shift_press:
                if not self._find_all_valid_moves():
                    self._handle_board_shuffle()
                    self.moves_remaining = max(0, self.moves_remaining - 3)
                    reward -= 5
                else:
                    reward -= 1 # Penalty for useless action

        # --- Termination Check ---
        if not self.game_over and self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            reward -= 100  # Loss penalty
        
        if not np.any(self.board): # All crystals cleared
            terminated = True
            self.game_over = True
            reward += 100 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_initial_board(self):
        self.board = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_matches_on_board(self.board) or not self._find_all_valid_moves():
            self.board = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _handle_board_shuffle(self):
        # # sfx: board_shuffle
        valid_moves_exist = True
        while valid_moves_exist:
            flat_board = self.board.flatten()
            self.np_random.shuffle(flat_board)
            self.board = flat_board.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_matches_on_board(self.board):
                valid_moves_exist = False
        
        # Ensure at least one move is possible after shuffle
        if not self._find_all_valid_moves():
            self._create_initial_board() # Failsafe, generate a new valid board

    def _grid_to_screen(self, r, c):
        x = self.origin_x + (c - r) * self.tile_width
        y = self.origin_y + (c + r) * self.tile_height
        return int(x), int(y)

    def _draw_iso_hexagon(self, surface, color, r, c, pos_offset=(0, 0), scale=1.0):
        center_x, center_y = self._grid_to_screen(r, c)
        center_x += pos_offset[0]
        center_y += pos_offset[1]
        
        w = self.tile_width * scale
        h = self.tile_height * scale
        
        points = [
            (center_x, center_y - h * 2),
            (center_x + w, center_y - h),
            (center_x + w, center_y + h),
            (center_x, center_y + h * 2),
            (center_x - w, center_y + h),
            (center_x - w, center_y - h),
        ]
        
        # Use gfxdraw for antialiasing
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

        # Add a subtle 3D effect
        top_color = tuple(min(255, val + 40) for val in color)
        side_color = tuple(max(0, val - 40) for val in color)
        
        top_points = [points[0], points[1], (center_x, center_y), points[5]]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        
        right_points = [points[1], points[2], points[3], (center_x, center_y)]
        pygame.gfxdraw.filled_polygon(surface, right_points, color)

        left_points = [points[3], points[4], points[5], (center_x, center_y)]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color)


    def _update_animations(self):
        if not self.animations:
            return
        
        # An animation is a dict: {type, obj, start_pos, end_pos, duration, progress}
        finished_animations = []
        for anim in self.animations:
            anim['progress'] += 1
            if anim['progress'] >= anim['duration']:
                finished_animations.append(anim)
        
        for anim in finished_animations:
            # Finalize state
            if anim['type'] == 'swap':
                r1, c1 = anim['obj1_rc']
                r2, c2 = anim['obj2_rc']
                self.board[r1][c1], self.board[r2][c2] = self.board[r2][c1], self.board[r1][c1]
            elif anim['type'] == 'fall':
                self._apply_gravity()
                self._refill_board()
            
            self.animations.remove(anim)

    def _find_matches_on_board(self, board):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if board[r][c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and board[r][c] == board[r][c+1] == board[r][c+2]:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                # Vertical
                if r < self.GRID_HEIGHT - 2 and board[r][c] == board[r+1][c] == board[r+2][c]:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return list(matches)

    def _resolve_matches(self):
        matches = self._find_matches_on_board(self.board)
        if not matches:
            return 0, False
        
        # # sfx: match_clear
        reward = 0
        num_cleared = len(matches)
        reward += num_cleared # +1 per crystal
        if num_cleared > 3:
            reward += 5 # Chain reaction bonus
        
        self.score += num_cleared

        for r, c in matches:
            crystal_type = self.board[r][c]
            if crystal_type > 0:
                # Create particles
                for _ in range(10):
                    self.particles.append(self._create_particle(r, c, self.CRYSTAL_COLORS[crystal_type-1]))
            self.board[r][c] = 0
        
        self._create_fall_animation()
        self.game_state = "FALL_ANIM"
        return reward, True

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_r = -1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r][c] == 0 and empty_r == -1:
                    empty_r = r
                elif self.board[r][c] != 0 and empty_r != -1:
                    self.board[empty_r][c] = self.board[r][c]
                    self.board[r][c] = 0
                    empty_r -= 1

    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.board[r][c] == 0:
                    self.board[r][c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)
    
    def _create_swap_animation(self, pos1, pos2, is_invalid=False):
        r1, c1 = pos1
        r2, c2 = pos2
        self.swap_origin, self.swap_target = pos1, pos2
        duration = 10 if not is_invalid else 5
        
        anim1 = {
            'type': 'swap', 'obj1_rc': (r1,c1), 'obj2_rc': (r2,c2),
            'start_pos': self._grid_to_screen(r1, c1),
            'end_pos': self._grid_to_screen(r2, c2),
            'duration': duration, 'progress': 0
        }
        anim2 = {
            'type': 'swap', 'obj1_rc': (r2,c2), 'obj2_rc': (r1,c1),
            'start_pos': self._grid_to_screen(r2, c2),
            'end_pos': self._grid_to_screen(r1, c1),
            'duration': duration, 'progress': 0
        }
        if is_invalid:
            # # sfx: invalid_swap
            anim1['end_pos'], anim1['start_pos'] = anim1['start_pos'], anim1['end_pos']
            anim2['end_pos'], anim2['start_pos'] = anim2['start_pos'], anim2['end_pos']
        else:
            # # sfx: crystal_swap
            pass
        self.animations.extend([anim1, anim2])

    def _create_fall_animation(self):
        fall_data = {}
        for c in range(self.GRID_WIDTH):
            fall_dist = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r][c] == 0:
                    fall_dist += 1
                elif fall_dist > 0:
                    fall_data[(r,c)] = fall_dist
        
        if not fall_data: return
        
        # # sfx: crystals_fall
        anim = {'type': 'fall', 'fall_data': fall_data, 'duration': 8, 'progress': 0}
        self.animations.append(anim)

    def _find_all_valid_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.board[r][c], self.board[r][c+1] = self.board[r][c+1], self.board[r][c]
                    if self._find_matches_on_board(self.board): moves.append(((r,c), (r,c+1)))
                    self.board[r][c], self.board[r][c+1] = self.board[r][c+1], self.board[r][c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.board[r][c], self.board[r+1][c] = self.board[r+1][c], self.board[r][c]
                    if self._find_matches_on_board(self.board): moves.append(((r,c), (r+1,c)))
                    self.board[r][c], self.board[r+1][c] = self.board[r+1][c], self.board[r][c] # Swap back
        return moves
    
    def _create_particle(self, r, c, color):
        x, y = self._grid_to_screen(r, c)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 5)
        return {
            'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
            'color': color, 'lifetime': random.randint(15, 30)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _render_game(self):
        # --- Draw Board ---
        animated_crystals = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                animated_crystals.add(anim['obj1_rc'])
                animated_crystals.add(anim['obj2_rc'])
            elif anim['type'] == 'fall':
                for (r,c) in anim['fall_data'].keys():
                    animated_crystals.add((r,c))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in animated_crystals and self.board[r][c] != 0:
                    color = self.CRYSTAL_COLORS[self.board[r][c] - 1]
                    self._draw_iso_hexagon(self.screen, color, r, c)

        # --- Draw Animations ---
        for anim in self.animations:
            t = anim['progress'] / anim['duration']
            # Ease-out-cubic
            ease_t = 1 - pow(1 - t, 3)

            if anim['type'] == 'swap':
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                curr_x = start_x + (end_x - start_x) * ease_t
                curr_y = start_y + (end_y - start_y) * ease_t
                
                # Determine which object this part of the animation refers to
                obj_rc = anim['obj1_rc'] if anim['end_pos'] == self._grid_to_screen(*anim['obj2_rc']) else anim['obj2_rc']
                
                # Get the crystal type from the *original* position before swap
                orig_r, orig_c = obj_rc
                crystal_type = self.board[orig_r][orig_c]
                
                # If swap is invalid, positions are swapped back, so we need to look up the type from the other object
                if self.game_state == "INVALID_SWAP_ANIM":
                    other_rc = anim['obj2_rc'] if obj_rc == anim['obj1_rc'] else anim['obj1_rc']
                    crystal_type = self.board[other_rc[0]][other_rc[1]]

                if crystal_type > 0:
                    color = self.CRYSTAL_COLORS[crystal_type - 1]
                    base_x, base_y = self._grid_to_screen(obj_rc[0], obj_rc[1])
                    self._draw_iso_hexagon(self.screen, color, 0, 0, pos_offset=(curr_x, curr_y-self.origin_y), scale=1.1)
            
            elif anim['type'] == 'fall':
                for (r,c), dist in anim['fall_data'].items():
                    start_pos = self._grid_to_screen(r, c)
                    end_pos = self._grid_to_screen(r + dist, c)
                    curr_y = start_pos[1] + (end_pos[1] - start_pos[1]) * ease_t
                    crystal_type = self.board[r][c]
                    if crystal_type > 0:
                        color = self.CRYSTAL_COLORS[crystal_type - 1]
                        self._draw_iso_hexagon(self.screen, color, 0, 0, pos_offset=(start_pos[0], curr_y-self.origin_y))

        # --- Draw Particles ---
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, int(3 * (p['lifetime'] / 20.0)))
                pygame.draw.rect(self.screen, p['color'], (p['x'], p['y'], size, size))

        # --- Draw Cursor and Selection ---
        if self.game_state == "SELECT" and not self.game_over:
            # Draw selection highlight
            if self.selected_pos:
                r, c = self.selected_pos
                self._draw_iso_hexagon(self.screen, self.COLOR_SELECTED, r, c, scale=1.2)
            
            # Draw cursor
            r, c = self.cursor_pos
            self._draw_iso_hexagon(self.screen, self.COLOR_CURSOR, r, c, scale=1.1)
            # Fill with BG color to create an outline effect
            if self.board[r][c] > 0:
                color = self.CRYSTAL_COLORS[self.board[r][c] - 1]
                self._draw_iso_hexagon(self.screen, color, r, c, scale=1.0)
            else:
                self._draw_iso_hexagon(self.screen, self.COLOR_BG, r, c, scale=0.9)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Moves
        moves_color = self.COLOR_UI
        if self.moves_remaining <= 5 and not self.game_over:
            # Flashing effect for low moves
            if (self.steps // 5) % 2 == 0:
                moves_color = (255, 80, 80)
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, moves_color)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "BOARD CLEARED!" if not np.any(self.board) else "OUT OF MOVES"
            
            text_surf = self.font_large.render(status_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)
    
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