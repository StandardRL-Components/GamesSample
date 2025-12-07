
# Generated: 2025-08-28T05:24:01.468273
# Source Brief: brief_02611.md
# Brief Index: 2611

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Shift to select a gem, "
        "then move to an adjacent gem and press Space to swap."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Clear the entire board "
        "before you run out of moves! Plan your swaps to create cascading combos for a high score."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.GEM_SIZE = 40
        self.NUM_GEM_TYPES = 6
        self.INITIAL_MOVES = 25
        self.MAX_STEPS = 1000
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECT = (255, 255, 255)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.board = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.animations = []
        self.pending_reward = 0
        self.combo_level = 0
        self.last_action_was_press = {'space': False, 'shift': False}
        self.is_resolving = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.board = self._create_initial_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.animations.clear()
        self.pending_reward = 0
        self.combo_level = 0
        self.last_action_was_press = {'space': False, 'shift': False}
        self.is_resolving = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        # --- 1. Update Animations and Board Resolution ---
        if self.animations or self.is_resolving:
            self._update_animations()
            if not self.animations and self.is_resolving:
                self._resolve_board_state()
            # Player input is ignored while animations/resolutions are in progress
        else:
            # --- 2. Process Player Input ---
            movement, space_val, shift_val = action[0], action[1], action[2]
            space_pressed = space_val == 1 and not self.last_action_was_press['space']
            shift_pressed = shift_val == 1 and not self.last_action_was_press['shift']
            self.last_action_was_press['space'] = space_val == 1
            self.last_action_was_press['shift'] = shift_val == 1

            # Handle cursor movement
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

            # Handle gem selection (Shift)
            if shift_pressed:
                if self.selected_gem_pos == self.cursor_pos:
                    self.selected_gem_pos = None # Deselect
                else:
                    self.selected_gem_pos = list(self.cursor_pos) # Select

            # Handle swap attempt (Space)
            if space_pressed and self.selected_gem_pos:
                if self._is_adjacent(self.cursor_pos, self.selected_gem_pos):
                    self._initiate_swap(self.cursor_pos, self.selected_gem_pos)
                    self.moves_left -= 1
                else: # Invalid swap (not adjacent)
                    reward -= 0.1
                    self.animations.append(ShakeAnimation(self.cursor_pos))
                    self.selected_gem_pos = None

        # --- 3. Finalize Step ---
        # Accrue rewards earned during resolution
        reward += self.pending_reward
        self.score += self.pending_reward
        self.pending_reward = 0
        
        # Check termination conditions
        board_cleared = np.all(self.board == -1)
        if (self.moves_left <= 0 and not self.animations and not self.is_resolving) or board_cleared:
            terminated = True
            self.game_over = True
            if board_cleared:
                reward += 100 # Win bonus
            else:
                reward -= 50 # Loss penalty

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}
    
    def _render_text(self, text, font, color, position, shadow_color=(0,0,0)):
        shadow = font.render(text, True, shadow_color)
        self.screen.blit(shadow, (position[0] + 2, position[1] + 2))
        surface = font.render(text, True, color)
        self.screen.blit(surface, position)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.GEM_SIZE, y), 1)
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.GEM_SIZE), 1)

        # Draw gems
        rendered_gems = set()
        for anim in self.animations:
            if hasattr(anim, 'render'):
                anim.render(self)
                if hasattr(anim, 'pos1'): rendered_gems.add(tuple(anim.pos1))
                if hasattr(anim, 'pos2'): rendered_gems.add(tuple(anim.pos2))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (c, r) in rendered_gems: continue
                gem_type = self.board[r, c]
                if gem_type != -1:
                    self._draw_gem(c, r, gem_type)

        # Draw selection highlight
        if self.selected_gem_pos:
            x, y = self.selected_gem_pos
            rect = pygame.Rect(self.GRID_X_OFFSET + x * self.GEM_SIZE, self.GRID_Y_OFFSET + y * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 100 + 155
            pygame.draw.rect(self.screen, (pulse, pulse, pulse), rect, 3)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_X_OFFSET + cx * self.GEM_SIZE, self.GRID_Y_OFFSET + cy * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _draw_gem(self, c, r, gem_type, pos_override=None, size_override=None):
        size = size_override if size_override is not None else self.GEM_SIZE
        if pos_override:
            px, py = pos_override
        else:
            px = self.GRID_X_OFFSET + c * self.GEM_SIZE + self.GEM_SIZE / 2
            py = self.GRID_Y_OFFSET + r * self.GEM_SIZE + self.GEM_SIZE / 2
        
        color = self.GEM_COLORS[gem_type]
        light_color = tuple(min(255, val + 60) for val in color)
        
        radius = int(size * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, light_color)
        
        # Highlight
        highlight_pos = (int(px - radius * 0.3), int(py - radius * 0.3))
        pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], int(radius*0.3), (255,255,255,150))


    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_large, (255, 255, 255), (20, 10))
        self._render_text(f"Moves: {self.moves_left}", self.font_large, (255, 255, 255), (self.SCREEN_WIDTH - 150, 10))
        if self.game_over:
            win_text = "BOARD CLEARED!" if np.all(self.board == -1) else "OUT OF MOVES"
            self._render_text(win_text, self.font_large, self.COLOR_CURSOR, (self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT/2 - 20))

    def _create_initial_board(self):
        while True:
            board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_matches(board):
                self._remove_matches(board, self._find_matches(board))
                self._apply_gravity_and_refill(board)
            if self._find_all_valid_moves(board):
                return board

    def _find_matches(self, board):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if board[r, c] == -1: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    def _remove_matches(self, board, matches):
        for c, r in matches:
            board[r, c] = -1

    def _apply_gravity_and_refill(self, board):
        falls = []
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if board[r, c] != -1:
                    if r != empty_row:
                        board[empty_row, c] = board[r, c]
                        board[r, c] = -1
                        falls.append({'from': (c, r), 'to': (c, empty_row), 'type': board[empty_row, c]})
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                board[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                falls.append({'from': (c, r - (empty_row + 1)), 'to': (c, r), 'type': board[r, c]})
        return falls

    def _find_all_valid_moves(self, board):
        valid_moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    board[r,c], board[r,c+1] = board[r,c+1], board[r,c]
                    if self._find_matches(board): valid_moves.append(((c,r), (c+1,r)))
                    board[r,c], board[r,c+1] = board[r,c+1], board[r,c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    board[r,c], board[r+1,c] = board[r+1,c], board[r,c]
                    if self._find_matches(board): valid_moves.append(((c,r), (c,r+1)))
                    board[r,c], board[r+1,c] = board[r+1,c], board[r,c] # Swap back
        return valid_moves

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _initiate_swap(self, pos1, pos2):
        self.is_resolving = True
        self.combo_level = 0
        self.animations.append(SwapAnimation(pos1, pos2, on_complete=self._post_swap_check))

    def _post_swap_check(self, pos1, pos2):
        # This is a callback, executed after the swap animation
        c1, r1 = int(pos1[0]), int(pos1[1])
        c2, r2 = int(pos2[0]), int(pos2[1])
        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
        
        matches = self._find_matches(self.board)
        if not matches: # Invalid move, swap back
            self.pending_reward -= 0.1
            self.animations.append(SwapAnimation(pos1, pos2, on_complete=lambda p1, p2: setattr(self, 'is_resolving', False)))
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1] # swap back data
        else:
            self._resolve_board_state()

    def _resolve_board_state(self):
        # This function is called after a swap or after gems have fallen
        matches = self._find_matches(self.board)
        if matches:
            if self.combo_level > 0:
                self.pending_reward += 10 # Cascade bonus
            self.combo_level += 1

            num_gems = len(matches)
            self.pending_reward += num_gems # +1 per gem
            if num_gems > 3:
                self.pending_reward += 5 # Bonus for 4+ match

            # Create fade/particle animations
            for c, r in matches:
                self.animations.append(FadeAnimation((c, r), self.board[r, c]))
            self._remove_matches(self.board, matches)
            
            # The on_complete of the last FadeAnimation will trigger gravity
            self.animations[-1].on_complete = self._post_fade_gravity_check
        else:
            self.is_resolving = False # Board is stable
            if not self._find_all_valid_moves(self.board) and np.any(self.board != -1):
                self._shuffle_board()

    def _post_fade_gravity_check(self):
        falls = self._apply_gravity_and_refill(self.board)
        if falls:
            for fall in falls:
                self.animations.append(FallAnimation(fall['from'], fall['to'], fall['type']))
            # After falling, re-check for matches (cascade)
            self.animations[-1].on_complete = self._resolve_board_state
        else:
            self.is_resolving = False # No falls, board is stable
    
    def _shuffle_board(self):
        # Animation for shuffling could be added here
        gem_list = self.board[self.board != -1].tolist()
        self.np_random.shuffle(gem_list)
        gem_queue = deque(gem_list)
        
        new_board = np.full_like(self.board, -1)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.board[r, c] != -1:
                    new_board[r, c] = gem_queue.popleft()
        self.board = new_board

        # Ensure shuffled board is valid
        while self._find_matches(self.board) or not self._find_all_valid_moves(self.board):
            self.board = self._create_initial_board()
        # Add a visual indicator for shuffle
        self.animations.append(FlashAnimation("SHUFFLE", 60))


    def _update_animations(self):
        finished_anims = []
        for anim in self.animations:
            anim.update()
            if anim.is_done():
                finished_anims.append(anim)
                if anim.on_complete:
                    if anim.cb_args:
                        anim.on_complete(*anim.cb_args)
                    else:
                        anim.on_complete()
        self.animations = [anim for anim in self.animations if anim not in finished_anims]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Animation Classes ---
class Animation:
    def __init__(self, duration, on_complete=None, cb_args=None):
        self.duration = duration
        self.timer = 0
        self.on_complete = on_complete
        self.cb_args = cb_args

    def update(self):
        self.timer += 1

    def is_done(self):
        return self.timer >= self.duration
    
    def progress(self):
        return min(1.0, self.timer / self.duration)

class SwapAnimation(Animation):
    def __init__(self, pos1, pos2, on_complete=None):
        super().__init__(duration=10, on_complete=on_complete, cb_args=(pos1, pos2))
        self.pos1 = pos1
        self.pos2 = pos2

    def render(self, env):
        p = self.progress()
        c1, r1 = self.pos1
        c2, r2 = self.pos2
        
        gem_type1 = env.board[r1, c1]
        gem_type2 = env.board[r2, c2]

        start_x1 = env.GRID_X_OFFSET + c1 * env.GEM_SIZE + env.GEM_SIZE / 2
        start_y1 = env.GRID_Y_OFFSET + r1 * env.GEM_SIZE + env.GEM_SIZE / 2
        end_x1 = env.GRID_X_OFFSET + c2 * env.GEM_SIZE + env.GEM_SIZE / 2
        end_y1 = env.GRID_Y_OFFSET + r2 * env.GEM_SIZE + env.GEM_SIZE / 2

        curr_x1 = start_x1 + (end_x1 - start_x1) * p
        curr_y1 = start_y1 + (end_y1 - start_y1) * p
        curr_x2 = end_x1 + (start_x1 - end_x1) * p
        curr_y2 = end_y1 + (start_y1 - end_y1) * p

        if gem_type1 != -1: env._draw_gem(c1, r1, gem_type1, pos_override=(curr_x1, curr_y1))
        if gem_type2 != -1: env._draw_gem(c2, r2, gem_type2, pos_override=(curr_x2, curr_y2))

class FadeAnimation(Animation):
    def __init__(self, pos, gem_type, on_complete=None):
        super().__init__(duration=15, on_complete=on_complete)
        self.pos = pos
        self.gem_type = gem_type
        self.particles = [Particle(pos) for _ in range(10)]

    def update(self):
        super().update()
        for p in self.particles: p.update()

    def render(self, env):
        p = 1.0 - self.progress()
        size = env.GEM_SIZE * p
        c, r = self.pos
        px = env.GRID_X_OFFSET + c * env.GEM_SIZE + env.GEM_SIZE / 2
        py = env.GRID_Y_OFFSET + r * env.GEM_SIZE + env.GEM_SIZE / 2
        
        env._draw_gem(c, r, self.gem_type, pos_override=(px, py), size_override=size)
        
        # Render particles
        color = env.GEM_COLORS[self.gem_type]
        for particle in self.particles:
            particle.render(env.screen, color, px, py)

class FallAnimation(Animation):
    def __init__(self, from_pos, to_pos, gem_type, on_complete=None):
        duration = int(math.sqrt(to_pos[1] - from_pos[1]) * 8)
        super().__init__(duration=max(5, duration), on_complete=on_complete)
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.gem_type = gem_type

    def render(self, env):
        p = self.progress()**2 # Ease-in effect
        c, r_from = self.from_pos
        _, r_to = self.to_pos
        
        start_y = env.GRID_Y_OFFSET + r_from * env.GEM_SIZE + env.GEM_SIZE / 2
        end_y = env.GRID_Y_OFFSET + r_to * env.GEM_SIZE + env.GEM_SIZE / 2
        curr_y = start_y + (end_y - start_y) * p
        px = env.GRID_X_OFFSET + c * env.GEM_SIZE + env.GEM_SIZE / 2

        env._draw_gem(c, r_to, self.gem_type, pos_override=(px, curr_y))

class ShakeAnimation(Animation):
    def __init__(self, pos):
        super().__init__(duration=10)
        self.pos = pos
    
    def render(self, env):
        c, r = self.pos
        gem_type = env.board[r, c]
        if gem_type == -1: return
        offset = math.sin(self.progress() * math.pi * 4) * 3
        px = env.GRID_X_OFFSET + c * env.GEM_SIZE + env.GEM_SIZE / 2 + offset
        py = env.GRID_Y_OFFSET + r * env.GEM_SIZE + env.GEM_SIZE / 2
        env._draw_gem(c, r, gem_type, pos_override=(px, py))

class FlashAnimation(Animation):
    def __init__(self, text, duration):
        super().__init__(duration=duration)
        self.text = text
    
    def render(self, env):
        p = self.progress()
        alpha = 0
        if p < 0.2: alpha = p / 0.2
        elif p > 0.8: alpha = (1-p) / 0.2
        else: alpha = 1.0
        
        color = (255, 255, 0, alpha * 255)
        text_surf = env.font_large.render(self.text, True, color)
        text_surf.set_alpha(alpha * 255)
        pos = (env.SCREEN_WIDTH/2 - text_surf.get_width()/2, env.SCREEN_HEIGHT/2 - text_surf.get_height()/2)
        env.screen.blit(text_surf, pos)

class Particle:
    def __init__(self, grid_pos):
        self.pos = [0, 0]
        self.vel = [(random.random() - 0.5) * 4, (random.random() - 0.5) * 4]
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.05)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # gravity
        self.life -= self.decay
    
    def render(self, surface, color, center_x, center_y):
        if self.life > 0:
            radius = int(self.life * 4)
            pos = (int(center_x + self.pos[0]), int(center_y + self.pos[1]))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)