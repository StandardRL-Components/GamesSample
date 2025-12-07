
# Generated: 2025-08-27T15:23:36.901976
# Source Brief: brief_00972.md
# Brief Index: 972

        
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
        "Controls: Arrows to move cursor. Space to select a tile. "
        "Arrows again to aim swap. Space again to execute swap. Shift to deselect."
    )

    game_description = (
        "A 3x3 match-3 puzzle game. Swap adjacent tiles to create matches of three or more. "
        "Clear the board before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 3
    NUM_COLORS = 5
    MAX_MOVES = 10
    ANIMATION_SPEED = 4  # Higher is faster

    # Colors
    COLOR_BG = (25, 28, 38)
    COLOR_GRID = (50, 55, 70)
    TILE_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 100, 220),  # Blue
        (220, 220, 50),  # Yellow
        (150, 50, 220),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 165, 0)
    COLOR_TEXT = (230, 230, 230)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Grid layout
        self.tile_size = 100
        self.tile_padding = 10
        self.grid_width = self.GRID_SIZE * self.tile_size + (self.GRID_SIZE - 1) * self.tile_padding
        self.grid_height = self.grid_width
        self.grid_x = (self.screen_width - self.grid_width) // 2
        self.grid_y = (self.screen_height - self.grid_height) // 2
        
        # This will be initialized in reset
        self.np_random = None
        self.grid = None
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        # Input and selection state
        self.cursor = (0, 0)
        self.selected_tile = None
        self.last_move_dir = (0, 0)
        
        # Animation state machine
        self.animation_state = 'IDLE'
        self.animation_timer = 0
        self.animation_data = {}
        self.pending_reward = 0
        self.chain_count = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_board()
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        self.cursor = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.selected_tile = None
        self.last_move_dir = (0, 0)
        
        self.animation_state = 'IDLE'
        self.animation_timer = 0
        self.animation_data = {}
        self.pending_reward = 0
        self.chain_count = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        if self.animation_state != 'IDLE':
            self._update_animation()
            reward = self.pending_reward
            self.pending_reward = 0 # Consume reward
        else:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held, shift_held)

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        if shift_held:
            self.selected_tile = None
            # sfx: deselect
            return

        # --- Movement ---
        if movement > 0:
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            self.cursor = (
                (self.cursor[0] + dr) % self.GRID_SIZE,
                (self.cursor[1] + dc) % self.GRID_SIZE
            )
            self.last_move_dir = (dr, dc)
            # sfx: cursor_move
        
        # --- Space Action ---
        if space_held:
            if self.selected_tile is None:
                self.selected_tile = self.cursor
                # sfx: select_tile
            else:
                # Attempt a swap
                r1, c1 = self.selected_tile
                dr, dc = self.last_move_dir
                r2, c2 = r1 + dr, c1 + dc

                if self._is_valid_pos(r2, c2) and self._are_adjacent(r1, c1, r2, c2):
                    self.moves_left -= 1
                    self.animation_data = {'pos1': (r1, c1), 'pos2': (r2, c2)}
                    self._start_animation('SWAP')
                    # sfx: swap_start
                else:
                    self.selected_tile = None # Invalid aim, deselect
                    # sfx: invalid_move

    def _update_animation(self):
        self.animation_timer += self.ANIMATION_SPEED
        
        if self.animation_timer >= 100:
            self.animation_timer = 0
            
            # --- SWAP -> CHECK ---
            if self.animation_state == 'SWAP':
                pos1, pos2 = self.animation_data['pos1'], self.animation_data['pos2']
                self._swap_tiles(pos1, pos2)
                self.selected_tile = None # Deselect after swap
                self.chain_count = 0
                self._check_for_matches()

            # --- CLEAR -> FALL ---
            elif self.animation_state == 'CLEAR':
                matched = self.animation_data['matched']
                for r, c in matched:
                    self.grid[r, c] = -1 # Mark for removal
                self._start_animation('FALL')

            # --- FALL -> CHECK ---
            elif self.animation_state == 'FALL':
                self._apply_gravity_and_refill()
                self.chain_count += 1
                self._check_for_matches()
                
            # --- SWAP_BACK -> IDLE ---
            elif self.animation_state == 'SWAP_BACK':
                pos1, pos2 = self.animation_data['pos1'], self.animation_data['pos2']
                self._swap_tiles(pos1, pos2) # Swap back data
                self._end_turn(final_reward=-0.2)
                
    def _check_for_matches(self):
        matched_tiles = self._find_all_matches()
        if matched_tiles:
            reward = len(matched_tiles)
            if self.chain_count > 0:
                reward += 5 # Chain reaction bonus
            self.pending_reward += reward
            self.animation_data['matched'] = matched_tiles
            self._start_animation('CLEAR')
            # sfx: match_success
        else:
            if self.chain_count == 0: # Initial swap failed
                self._start_animation('SWAP_BACK')
                # sfx: swap_fail
            else: # Chain is over
                self._end_turn(final_reward=0)

    def _end_turn(self, final_reward):
        self.animation_state = 'IDLE'
        self.pending_reward += final_reward
        self.score += self.pending_reward
        
        if np.all(self.grid == -1): # All tiles cleared
            self.pending_reward += 100
            self.score += 100
            self.game_over = True
            # sfx: victory
        elif self.moves_left <= 0:
            self.pending_reward -= 10
            self.score -= 10
            self.game_over = True
            # sfx: game_over
            
    def _start_animation(self, state):
        self.animation_state = state
        self.animation_timer = 0
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = self._get_tile_rect(r, c)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=8)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                if color_idx == -1: continue

                pos_x, pos_y = self._get_tile_center(r, c)
                size = self.tile_size
                
                # Handle animations
                if self.animation_state in ['SWAP', 'SWAP_BACK']:
                    p1, p2 = self.animation_data['pos1'], self.animation_data['pos2']
                    if (r, c) == p1:
                        target_x, target_y = self._get_tile_center(p2[0], p2[1])
                        pos_x = self._lerp(pos_x, target_x, self.animation_timer / 100)
                        pos_y = self._lerp(pos_y, target_y, self.animation_timer / 100)
                    elif (r, c) == p2:
                        target_x, target_y = self._get_tile_center(p1[0], p1[1])
                        pos_x = self._lerp(pos_x, target_x, self.animation_timer / 100)
                        pos_y = self._lerp(pos_y, target_y, self.animation_timer / 100)

                if self.animation_state == 'CLEAR' and (r,c) in self.animation_data['matched']:
                    size = self._lerp(self.tile_size, 0, self.animation_timer / 100)
                
                if self.animation_state == 'FALL' and (r,c) in self.animation_data.get('fall_map', {}):
                    start_r, start_c = self.animation_data['fall_map'][(r,c)]
                    start_x, start_y = self._get_tile_center(start_r, start_c)
                    pos_x = self._lerp(start_x, pos_x, self.animation_timer / 100)
                    pos_y = self._lerp(start_y, pos_y, self.animation_timer / 100)

                self._draw_tile(pos_x, pos_y, size, color_idx)
        
        # Draw cursor and selection
        if self.animation_state == 'IDLE':
            # Cursor
            cursor_rect = self._get_tile_rect(self.cursor[0], self.cursor[1])
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=10)
            
            # Selected
            if self.selected_tile:
                sel_rect = self._get_tile_rect(self.selected_tile[0], self.selected_tile[1])
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, sel_rect, width=6, border_radius=10)
                
                # Aim indicator
                dr, dc = self.last_move_dir
                if dr != 0 or dc != 0:
                    sr, sc = self.selected_tile
                    tr, tc = sr + dr, sc + dc
                    if self._is_valid_pos(tr, tc):
                        start_center = self._get_tile_center(sr, sc)
                        end_center = self._get_tile_center(tr, tc)
                        pygame.draw.line(self.screen, self.COLOR_SELECTED, start_center, end_center, 4)

    def _draw_tile(self, x, y, size, color_idx):
        if size <= 0: return
        rect = pygame.Rect(x - size/2, y - size/2, size, size)
        color = self.TILE_COLORS[color_idx]
        
        # Draw a slightly larger, darker rect for a border effect
        border_rect = rect.inflate(4, 4)
        border_color = tuple(max(0, val - 40) for val in color)
        pygame.draw.rect(self.screen, border_color, border_rect, border_radius=12)
        
        pygame.draw.rect(self.screen, color, rect, border_radius=10)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.grid == -1)
            msg = "YOU WIN!" if win_condition else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_SELECTED if win_condition else self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Helper Functions ---
    def _get_tile_rect(self, r, c):
        x = self.grid_x + c * (self.tile_size + self.tile_padding)
        y = self.grid_y + r * (self.tile_size + self.tile_padding)
        return pygame.Rect(x, y, self.tile_size, self.tile_size)

    def _get_tile_center(self, r, c):
        rect = self._get_tile_rect(r, c)
        return rect.centerx, rect.centery

    def _generate_board(self):
        while True:
            grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches(grid):
                if self._find_possible_matches(grid):
                    return grid

    def _find_possible_matches(self, grid):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r, c + 1] = temp_grid[r, c + 1], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r + 1, c] = temp_grid[r + 1, c], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
        return False

    def _find_all_matches(self, grid=None):
        if grid is None: grid = self.grid
        
        matched_tiles = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] == -1: continue
                
                # Horizontal check
                if c < self.GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matched_tiles.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical check
                if r < self.GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matched_tiles.update([(r, c), (r+1, c), (r+2, c)])
        return matched_tiles

    def _swap_tiles(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _apply_gravity_and_refill(self):
        fall_map = {}
        for c in range(self.GRID_SIZE):
            empty_r = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_r:
                        fall_map[(empty_r, c)] = (r, c) # Map dest to start for animation
                        self.grid[empty_r, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_r -= 1
        
        self.animation_data['fall_map'] = fall_map

        # Refill
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)

    @staticmethod
    def _lerp(start, end, t):
        return start + t * (end - start)

    @staticmethod
    def _are_adjacent(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not terminated:
        # --- Human Input ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
        
        action = [movement, space, shift]
        
        # We only step if an action is taken or an animation is playing
        if any(action) or env.animation_state != 'IDLE':
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            env.screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play
        
    print(f"Game Over! Final Score: {info['score']} (Total Reward: {total_reward})")
    pygame.quit()