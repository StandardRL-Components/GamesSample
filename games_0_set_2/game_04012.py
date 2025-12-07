
# Generated: 2025-08-28T01:08:01.415608
# Source Brief: brief_04012.md
# Brief Index: 4012

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a gem. "
        "Select a second, adjacent gem to swap them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically match gems in a grid-based puzzle to reach a target score "
        "before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 8, 8
    GEM_SIZE = 40
    NUM_GEM_TYPES = 6
    GRID_X = (SCREEN_WIDTH - GRID_COLS * GEM_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_ROWS * GEM_SIZE) // 2
    MAX_MOVES = 20
    TARGET_SCORE = 1000
    ANIMATION_STEPS = 5 # Number of steps for an animation to complete

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_SELECTED = (255, 255, 0, 150)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    GEM_HIGHLIGHTS = [tuple(min(255, c + 80) for c in color) for color in GEM_COLORS]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)
        
        # Etc...        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_pos = None
        self.game_state = "IDLE" # IDLE, SWAPPING, REVERTING, REMOVING, FALLING
        self.animations = []
        self.reward_buffer = 0
        self.prev_space_held = False
        self.game_over_message = ""
        self.last_swap = (None, None)
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use np.random.default_rng for modern numpy
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.game_over_message = ""
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        self.game_state = "IDLE"
        self.animations = []
        self.reward_buffer = 0
        self.prev_space_held = False

        self._initialize_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = 0
        terminated = self.game_over

        if self.game_state == "IDLE":
            self._handle_player_input(action)
        else:
            self._update_animations()

        if self.game_state == "IDLE" and self.reward_buffer != 0:
            reward += self.reward_buffer
            self.reward_buffer = 0
        
        if not terminated and (self.score >= self.TARGET_SCORE or self.moves_left <= 0):
            self.game_over = True
            terminated = True
            if self.score >= self.TARGET_SCORE:
                reward += 100
                self.game_over_message = "YOU WIN!"
            else:
                reward -= 100
                self.game_over_message = "GAME OVER"
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _initialize_grid(self):
        # Fill grid, ensuring no initial matches
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        
        # Ensure at least one move is possible
        if not self._find_possible_moves():
            # This is rare, but if it happens, just re-initialize.
            self._initialize_grid()

    def _handle_player_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- Move cursor ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Handle selection/swap ---
        if space_pressed and not self.game_over and self.game_state == "IDLE":
            r, c = self.cursor_pos[1], self.cursor_pos[0]
            if self.selected_pos is None:
                # First selection
                self.selected_pos = (r, c)
                # SFX: select_gem
            else:
                # Second selection
                r1, c1 = self.selected_pos
                dist = abs(r - r1) + abs(c - c1)

                if dist == 1:
                    # Adjacent selection, attempt swap
                    self.moves_left -= 1
                    self.last_swap = ((r1, c1), (r, c))
                    self._start_swap_animation((r1, c1), (r, c))
                    self.game_state = "SWAPPING"
                
                # Clear selection regardless of outcome
                self.selected_pos = None

    def _start_swap_animation(self, pos1, pos2, is_revert=False):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Swap logical grid
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        # Create animations
        self.animations.append({
            'type': 'move', 'gem_type': self.grid[r2, c2],
            'start_pos': self._grid_to_pixel(r1, c1), 'end_pos': self._grid_to_pixel(r2, c2),
            'progress': 0, 'is_revert': is_revert
        })
        self.animations.append({
            'type': 'move', 'gem_type': self.grid[r1, c1],
            'start_pos': self._grid_to_pixel(r2, c2), 'end_pos': self._grid_to_pixel(r1, c1),
            'progress': 0, 'is_revert': is_revert
        })
        # SFX: swap_start

    def _update_animations(self):
        if not self.animations:
            return

        finished_animations = []
        for anim in self.animations:
            anim['progress'] += 1 / self.ANIMATION_STEPS
            if anim['progress'] >= 1:
                finished_animations.append(anim)
        
        self.animations = [anim for anim in self.animations if anim not in finished_animations]

        if not self.animations:
            # All animations in the current batch are done
            self._on_animation_batch_finish()

    def _on_animation_batch_finish(self):
        if self.game_state == "SWAPPING":
            matches = self._find_matches()
            if matches:
                # SFX: match_success
                self._start_match_sequence(matches)
            else:
                # No match, revert swap
                # SFX: swap_fail
                self.reward_buffer -= 0.1
                r1, c1 = self.last_swap[0]
                r2, c2 = self.last_swap[1]
                self._start_swap_animation((r1, c1), (r2, c2), is_revert=True)
                self.game_state = "REVERTING"
        
        elif self.game_state == "REVERTING":
            self.game_state = "IDLE"
            if not self._find_possible_moves(): self._reshuffle_board()

        elif self.game_state == "REMOVING":
            self._start_fall_sequence()

        elif self.game_state == "FALLING":
            matches = self._find_matches()
            if matches:
                # Chain reaction
                # SFX: match_chain
                self._start_match_sequence(matches)
            else:
                self.game_state = "IDLE"
                if not self._find_possible_moves(): self._reshuffle_board()

    def _start_match_sequence(self, matches):
        # Calculate score and reward
        num_matched = len(matches)
        self.score += num_matched * 10
        self.reward_buffer += num_matched
        if num_matched == 4: self.reward_buffer += 5
        if num_matched >= 5: self.reward_buffer += 10

        # Start removal animation
        for r, c in matches:
            self.animations.append({
                'type': 'remove', 'gem_type': self.grid[r, c],
                'pos': self._grid_to_pixel(r, c), 'progress': 0,
            })
            self.grid[r, c] = -1 # Mark as empty

        self.game_state = "REMOVING"

    def _start_fall_sequence(self):
        # Move gems down in the logical grid and create fall animations
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    gem_type = self.grid[r, c]
                    self.grid[r + empty_count, c] = gem_type
                    self.grid[r, c] = -1
                    self.animations.append({
                        'type': 'fall', 'gem_type': gem_type,
                        'start_y': self._grid_to_pixel(r, c)[1],
                        'end_y': self._grid_to_pixel(r + empty_count, c)[1],
                        'x': self._grid_to_pixel(r, c)[0], 'progress': 0
                    })
        
        # Spawn new gems at the top
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r, c] == -1:
                    gem_type = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    self.grid[r, c] = gem_type
                    self.animations.append({
                        'type': 'fall', 'gem_type': gem_type,
                        'start_y': self._grid_to_pixel(r, c)[1] - self.GRID_ROWS * self.GEM_SIZE,
                        'end_y': self._grid_to_pixel(r, c)[1],
                        'x': self._grid_to_pixel(r, c)[0], 'progress': 0
                    })
        
        self.game_state = "FALLING"
        # SFX: gems_fall

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches(): moves.append(((r,c), (r,c+1)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c] # Swap back
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches(): moves.append(((r,c), (r+1,c)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c] # Swap back
        return moves

    def _reshuffle_board(self):
        # SFX: reshuffle
        self.reward_buffer -= 5 # Penalty for needing a reshuffle
        self.selected_pos = None # Clear selection before reshuffle
        self._initialize_grid()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _grid_to_pixel(self, r, c):
        return (
            self.GRID_X + c * self.GEM_SIZE + self.GEM_SIZE // 2,
            self.GRID_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        )

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (self.GRID_X + c * self.GEM_SIZE, self.GRID_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw static gems
        anim_grid_coords = set()
        for anim in self.animations:
            if anim['type'] == 'move':
                # The grid positions are already swapped, so we find the original spots
                p1_x, p1_y = anim['start_pos']
                p2_x, p2_y = anim['end_pos']
                c1, r1 = (p1_x - self.GRID_X - self.GEM_SIZE//2) // self.GEM_SIZE, (p1_y - self.GRID_Y - self.GEM_SIZE//2) // self.GEM_SIZE
                c2, r2 = (p2_x - self.GRID_X - self.GEM_SIZE//2) // self.GEM_SIZE, (p2_y - self.GRID_Y - self.GEM_SIZE//2) // self.GEM_SIZE
                anim_grid_coords.add((r1,c1))
                anim_grid_coords.add((r2,c2))

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != -1 and (r,c) not in anim_grid_coords:
                    self._draw_gem(self.screen, self.grid[r, c], self._grid_to_pixel(r, c))

        # Draw animated gems
        for anim in self.animations:
            p = anim['progress']
            if anim['type'] == 'move':
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                x = int(start_x + (end_x - start_x) * p)
                y = int(start_y + (end_y - start_y) * p)
                self._draw_gem(self.screen, anim['gem_type'], (x, y))
            elif anim['type'] == 'remove':
                scale = max(0, 1.0 - p)
                self._draw_gem(self.screen, anim['gem_type'], anim['pos'], scale)
            elif anim['type'] == 'fall':
                x = anim['x']
                y = int(anim['start_y'] + (anim['end_y'] - anim['start_y']) * p)
                self._draw_gem(self.screen, anim['gem_type'], (x, y))

        # Draw cursor and selection
        if not self.game_over:
            # Selection highlight (pulsating)
            if self.selected_pos is not None:
                r, c = self.selected_pos
                px, py = self._grid_to_pixel(r, c)
                pulse = (math.sin(self.steps * 0.3) + 1) / 2
                radius = int(self.GEM_SIZE * 0.5 * (1 + pulse * 0.2))
                alpha = int(100 + pulse * 50)
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, radius, radius, radius, (*self.COLOR_SELECTED[:3], alpha))
                pygame.gfxdraw.aacircle(s, radius, radius, radius, (*self.COLOR_SELECTED[:3], alpha))
                self.screen.blit(s, (px - radius, py - radius))

            # Cursor
            c, r = self.cursor_pos
            rect = (self.GRID_X + c * self.GEM_SIZE, self.GRID_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            s = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_CURSOR, (0, 0, self.GEM_SIZE, self.GEM_SIZE), 4, border_radius=4)
            self.screen.blit(s, rect[:2])

    def _draw_gem(self, surface, gem_type, pos, scale=1.0):
        if gem_type < 0: return
        radius = int(self.GEM_SIZE * 0.4 * scale)
        if radius <= 0: return
        
        px, py = int(pos[0]), int(pos[1])
        color = self.GEM_COLORS[gem_type]
        highlight = self.GEM_HIGHLIGHTS[gem_type]

        pygame.gfxdraw.filled_circle(surface, px, py, radius, color)
        pygame.gfxdraw.aacircle(surface, px, py, radius, color)
        
        # Add a small highlight for 3D effect
        highlight_offset_x = int(radius * 0.3)
        highlight_offset_y = int(radius * -0.3)
        highlight_radius = int(radius * 0.4)
        pygame.gfxdraw.filled_circle(surface, px + highlight_offset_x, py + highlight_offset_y, highlight_radius, highlight)
        pygame.gfxdraw.aacircle(surface, px + highlight_offset_x, py + highlight_offset_y, highlight_radius, highlight)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, (255, 255, 255))
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            msg_surf = self.font_msg.render(self.game_over_message, True, (255, 255, 0))
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

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