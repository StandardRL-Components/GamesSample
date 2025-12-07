
# Generated: 2025-08-28T04:23:11.043180
# Source Brief: brief_05238.md
# Brief Index: 5238

        
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
        "Controls: Arrow keys to move cursor. Space to select a gem, "
        "then space on an adjacent gem to swap. Shift to deselect."
    )

    game_description = (
        "Isometric match-3 puzzle game. Swap adjacent gems to create matches of 3 or more. "
        "Clear the entire board before you run out of moves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_ROWS, self.BOARD_COLS = 8, 8
        self.NUM_GEM_TYPES = 6
        self.STARTING_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Visual Constants ---
        self.TILE_WIDTH_HALF = 28
        self.TILE_HEIGHT_HALF = 14
        self.GEM_SIZE_MOD = 0.8  # Percentage of tile size

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]
        self.GEM_HIGHLIGHT_COLORS = [tuple(min(255, c + 60) for c in color) for color in self.GEM_COLORS]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        self.grid_origin_x = self.WIDTH // 2
        self.grid_origin_y = self.HEIGHT // 2 - (self.BOARD_ROWS * self.TILE_HEIGHT_HALF) // 4 + 30

        # --- State Variables ---
        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.win_state = False
        
        self.cursor_pos = [self.BOARD_ROWS // 2, self.BOARD_COLS // 2]
        self.selected_gem = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        # Generate a valid board
        while True:
            self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.BOARD_ROWS, self.BOARD_COLS))
            # Clear any initial matches
            while self._find_and_clear_matches(reward_multiplier=0) > 0:
                self._apply_gravity_and_refill()
            
            if self._find_possible_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # --- Action Handling ---
        if movement != 0:
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dr, 0, self.BOARD_ROWS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dc, 0, self.BOARD_COLS - 1)

        if shift_press and self.selected_gem is not None:
            self.selected_gem = None
            # sfx: deselect sound

        if space_press:
            cursor_r, cursor_c = self.cursor_pos
            if self.selected_gem is None:
                self.selected_gem = (cursor_r, cursor_c)
                # sfx: select sound
            else:
                sel_r, sel_c = self.selected_gem
                # Check for adjacency
                if abs(sel_r - cursor_r) + abs(sel_c - cursor_c) == 1:
                    self.moves_left -= 1
                    
                    # Perform swap
                    self.board[sel_r, sel_c], self.board[cursor_r, cursor_c] = \
                        self.board[cursor_r, cursor_c], self.board[sel_r, sel_c]
                    # sfx: swap sound

                    total_gems_cleared = 0
                    combo_multiplier = 1
                    
                    # Check for matches and handle cascades
                    gems_cleared_this_turn = self._find_and_clear_matches(reward_multiplier=combo_multiplier)
                    
                    if gems_cleared_this_turn > 0:
                        while gems_cleared_this_turn > 0:
                            total_gems_cleared += gems_cleared_this_turn
                            self._apply_gravity_and_refill()
                            combo_multiplier += 1
                            gems_cleared_this_turn = self._find_and_clear_matches(reward_multiplier=combo_multiplier)
                            # sfx: combo sound
                        reward += total_gems_cleared
                    else:
                        # Invalid move, swap back
                        self.board[sel_r, sel_c], self.board[cursor_r, cursor_c] = \
                            self.board[cursor_r, cursor_c], self.board[sel_r, sel_c]
                        reward = -0.1
                        # sfx: invalid move sound
                    
                    self.selected_gem = None
                else: # Not adjacent
                    self.selected_gem = (cursor_r, cursor_c) # Reselect at new cursor pos
                    # sfx: invalid select sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Termination Check ---
        gems_remaining = np.count_nonzero(self.board != -1)
        if gems_remaining == 0:
            self.win_state = True
            self.game_over = True
            terminated = True
            reward += 50 # Win bonus
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
        elif not self._find_possible_moves() and self.moves_left > 0:
            self.game_over = True # No more moves
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_and_clear_matches(self, reward_multiplier=1):
        to_clear = set()
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if c < self.BOARD_COLS - 2 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2] != -1:
                    to_clear.update([(r, c), (r, c+1), (r, c+2)])
                if r < self.BOARD_ROWS - 2 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c] != -1:
                    to_clear.update([(r, c), (r+1, c), (r+2, c)])
        
        if not to_clear:
            return 0

        for r, c in to_clear:
            gem_type = self.board[r, c]
            if gem_type != -1:
                self._create_particles(r, c, self.GEM_COLORS[gem_type])
                self.board[r, c] = -1 # Mark as empty
        
        # sfx: match sound
        return len(to_clear) * reward_multiplier

    def _apply_gravity_and_refill(self):
        for c in range(self.BOARD_COLS):
            empty_slots = []
            for r in range(self.BOARD_ROWS - 1, -1, -1):
                if self.board[r, c] == -1:
                    empty_slots.append(r)
                elif empty_slots:
                    new_r = empty_slots.pop(0)
                    self.board[new_r, c] = self.board[r, c]
                    self.board[r, c] = -1
                    empty_slots.append(r)
        
        # Refill from top
        for c in range(self.BOARD_COLS):
            for r in range(self.BOARD_ROWS):
                if self.board[r, c] == -1:
                    self.board[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_possible_moves(self):
        moves = []
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                # Try swapping right
                if c < self.BOARD_COLS - 1:
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                    if self._check_for_any_match():
                        moves.append(((r, c), (r, c+1)))
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c] # Swap back
                # Try swapping down
                if r < self.BOARD_ROWS - 1:
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                    if self._check_for_any_match():
                        moves.append(((r, c), (r+1, c)))
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c] # Swap back
        return moves

    def _check_for_any_match(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                gem = self.board[r, c]
                if gem == -1: continue
                # Horizontal match
                if c < self.BOARD_COLS - 2 and self.board[r, c+1] == gem and self.board[r, c+2] == gem:
                    return True
                # Vertical match
                if r < self.BOARD_ROWS - 2 and self.board[r+1, c] == gem and self.board[r+2, c] == gem:
                    return True
        return False

    def _grid_to_iso(self, r, c):
        iso_x = self.grid_origin_x + (c - r) * self.TILE_WIDTH_HALF
        iso_y = self.grid_origin_y + (c + r) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_gem(self, surface, r, c, gem_type, alpha=255):
        if gem_type == -1: return
        x, y = self._grid_to_iso(r, c)
        
        w = self.TILE_WIDTH_HALF * self.GEM_SIZE_MOD
        h = self.TILE_HEIGHT_HALF * self.GEM_SIZE_MOD
        
        color = self.GEM_COLORS[gem_type]
        highlight = self.GEM_HIGHLIGHT_COLORS[gem_type]

        points = [
            (x, y - h), (x + w, y), (x, y + h), (x - w, y)
        ]
        
        highlight_points = [
            (x, y - h), (x + w, y), (x, y), (x-w*0.1, y-h*0.1)
        ]
        
        # Apply alpha
        color_a = (*color, alpha)
        highlight_a = (*highlight, alpha)

        pygame.gfxdraw.filled_polygon(surface, points, color_a)
        pygame.gfxdraw.aapolygon(surface, points, color_a)
        pygame.gfxdraw.filled_polygon(surface, highlight_points, highlight_a)
        pygame.gfxdraw.aapolygon(surface, highlight_points, highlight_a)
    
    def _create_particles(self, r, c, color):
        x, y = self._grid_to_iso(r, c)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'color': color,
                'life': random.randint(20, 40)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['radius'] -= 0.1
            p['life'] -= 1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 30))))
                color_with_alpha = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color_with_alpha)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), color_with_alpha)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, x, y, align="topleft"):
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            setattr(text_rect, align, (x + 2, y + 2))
            self.screen.blit(text_surf, text_rect)
            
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect()
            setattr(text_rect, align, (x, y))
            self.screen.blit(text_surf, text_rect)

        draw_text(f"Score: {int(self.score)}", self.font_medium, self.COLOR_TEXT, 10, 10)
        draw_text(f"Moves: {self.moves_left}", self.font_medium, self.COLOR_TEXT, self.WIDTH - 10, 10, align="topright")

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            draw_text(msg, self.font_large, self.COLOR_CURSOR, self.WIDTH / 2, self.HEIGHT / 2, align="center")

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # Draw grid lines
        for r in range(self.BOARD_ROWS + 1):
            start = self._grid_to_iso(r, 0)
            end = self._grid_to_iso(r, self.BOARD_COLS)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for c in range(self.BOARD_COLS + 1):
            start = self._grid_to_iso(0, c)
            end = self._grid_to_iso(self.BOARD_ROWS, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw gems
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                self._draw_gem(self.screen, r, c, self.board[r,c])
        
        # Draw selected gem highlight
        if self.selected_gem is not None:
            r, c = self.selected_gem
            x, y = self._grid_to_iso(r, c)
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            radius = int(self.TILE_WIDTH_HALF * (0.8 + pulse * 0.2))
            alpha = int(100 + pulse * 50)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (*self.COLOR_CURSOR, alpha))
            pygame.gfxdraw.aacircle(self.screen, x, y, radius-1, (*self.COLOR_CURSOR, alpha))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cx, cy = self._grid_to_iso(cursor_r, cursor_c)
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        cursor_points = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, cursor_points, 2)
        
        self._update_and_draw_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "selected_gem": self.selected_gem,
            "is_win": self.win_state
        }
    
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display to avoid conflicts with headless rendering
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Match-3")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete actions
        # This mapping is slightly different from the env's internal logic
        # because we are polling keys, not using events.
        # This demonstrates how an agent would interact.
        # Arrow keys are mapped to grid directions, not iso directions.
        if keys[pygame.K_UP]: movement = 1 # Up
        elif keys[pygame.K_DOWN]: movement = 2 # Down
        elif keys[pygame.K_LEFT]: movement = 3 # Left
        elif keys[pygame.K_RIGHT]: movement = 4 # Right
        else: movement = 0
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # We only step if an action is taken, or if the game is over to show the final screen
        if any(action) or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")

        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for human play

    env.close()