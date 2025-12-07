import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Hold an arrow key and press Space to swap tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based color-matching puzzle. Swap adjacent tiles to create matches of three or more to clear the board before time runs out. Create combos for extra points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 8
    TILE_SIZE = 40
    GRID_X = (SCREEN_WIDTH - GRID_COLS * TILE_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_ROWS * TILE_SIZE) // 2
    MAX_STEPS = 30 * 60  # 60 seconds at 30fps
    FPS = 30

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_SCORE = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game State
        self.grid = []
        self.cursor_pos = [0, 0]
        self.last_move_dir = [0, 0]
        self.space_was_held = False
        self.animations = []
        self.particles = []
        self.game_state = 'IDLE' # IDLE, ANIMATING
        self.combo_multiplier = 1
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_move_dir = [0, 0]
        self.space_was_held = False
        self.animations = []
        self.particles = []
        self.game_state = 'IDLE'
        self.combo_multiplier = 1

        self._create_board()
        
        # Ensure starting board has some moves
        if not self._find_possible_moves():
             self.reset(seed=seed) # Recurse until a valid board is made

        return self._get_observation(), self._get_info()

    def _create_board(self):
        # Creates a grid with no initial matches.
        self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_ROWS, self.GRID_COLS)).tolist()
        
        while True:
            matches_found = self._find_all_matches()
            if not matches_found:
                break  # Board is valid, no matches on creation

            # Get all unique tile positions from the matches
            tiles_to_remove = set()
            for match in matches_found:
                for pos in match['positions']:
                    tiles_to_remove.add(pos)

            # Use the existing remove function, which just sets grid values to -1
            self._remove_matches(tiles_to_remove)

            # Now, apply gravity and refill without animations, directly manipulating the grid state.
            # This is the non-animated version of _apply_gravity_and_refill.

            # Gravity:
            for c in range(self.GRID_COLS):
                empty_row = self.GRID_ROWS - 1
                for r in range(self.GRID_ROWS - 1, -1, -1):
                    if self.grid[r][c] != -1:
                        if r != empty_row:
                            self.grid[empty_row][c] = self.grid[r][c]
                            self.grid[r][c] = -1
                        empty_row -= 1
            
            # Refill:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r][c] == -1:
                        self.grid[r][c] = self.np_random.integers(0, len(self.TILE_COLORS))


    def step(self, action):
        self.steps += 1
        self.timer -= 1
        
        reward = -0.01 # Cost of living

        # Update animations and particles
        self._update_animations()
        self._update_particles()

        if self.game_state == 'IDLE':
            # Only process input if not animating
            move_reward = self._handle_input(action)
            reward += move_reward
            if move_reward > 0: # A match was made
                self.combo_multiplier = 1

        # Check for game state transitions after animations finish
        if self.game_state == 'IDLE' and not self.animations:
            match_data = self._find_all_matches()
            if match_data:
                reward += self._process_matches(match_data)
                self.game_state = 'ANIMATING'
                self.combo_multiplier += 1
            else:
                self.combo_multiplier = 1 # Reset combo if no new chain reaction
                if not self._find_possible_moves():
                    # No moves left, reshuffle
                    self._create_board() # This is a simplification; a real game might just shuffle
        
        terminated = self.timer <= 0 or self._is_board_clear() or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self._is_board_clear():
                reward += 100 # Win bonus
            elif self.timer <= 0:
                reward -= 50 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        reward = 0
        
        # --- Movement ---
        moved = False
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            self.last_move_dir = [0, -1]
            moved = True
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            self.last_move_dir = [0, 1]
            moved = True
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            self.last_move_dir = [-1, 0]
            moved = True
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
            self.last_move_dir = [1, 0]
            moved = True

        # --- Swap Action ---
        if space_held and not self.space_was_held and moved:
            cx, cy = self.cursor_pos
            lx, ly = self.last_move_dir
            # The tile to swap with is the one we just moved FROM
            ox, oy = cx - lx, cy - ly

            if 0 <= ox < self.GRID_COLS and 0 <= oy < self.GRID_ROWS:
                # Execute swap
                self.grid[cy][cx], self.grid[oy][ox] = self.grid[oy][ox], self.grid[cy][cx]
                
                # Check if this swap creates a match
                matches = self._find_all_matches()
                if matches:
                    self.game_state = 'ANIMATING'
                    # Add swap animation
                    self.animations.append({'type': 'swap', 'pos1': (cx, cy), 'pos2': (ox, oy), 'progress': 0, 'duration': 8})
                    # Process matches will happen after animation
                else:
                    # Invalid move, swap back
                    self.grid[cy][cx], self.grid[oy][ox] = self.grid[oy][ox], self.grid[cy][cx]
                    # Add failed swap animation (swaps and swaps back)
                    self.animations.append({'type': 'swap_fail', 'pos1': (cx, cy), 'pos2': (ox, oy), 'progress': 0, 'duration': 12})
                    self.game_state = 'ANIMATING'
        
        self.space_was_held = space_held
        return reward

    def _process_matches(self, match_data):
        reward = 0
        tiles_to_remove = set()
        for match in match_data:
            for pos in match['positions']:
                tiles_to_remove.add(pos)
            
            match_len = len(match['positions'])
            base_reward = {3: 5, 4: 10}.get(match_len, 20)
            reward += base_reward * self.combo_multiplier
        
        reward += len(tiles_to_remove) * 1.0 # Reward for each tile cleared

        # Create animations and particles
        clear_duration = 10
        for (x, y) in tiles_to_remove:
            if self.grid[y][x] != -1:
                self.animations.append({'type': 'clear', 'pos': (x, y), 'progress': 0, 'duration': clear_duration})
                self._spawn_particles(x, y, self.grid[y][x])
                self.score += 10 * self.combo_multiplier

        # Schedule state changes after animations
        self.animations.append({'type': 'callback', 'func': self._remove_matches, 'args': [tiles_to_remove], 'progress': 0, 'duration': clear_duration})
        self.animations.append({'type': 'callback', 'func': self._apply_gravity_and_refill, 'args': [], 'progress': 0, 'duration': clear_duration})

        return reward

    def _find_all_matches(self):
        matches = []
        matched_tiles = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r][c] != -1 and (c,r) not in matched_tiles and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    match = [(c, r), (c+1, r), (c+2, r)]
                    i = c + 3
                    while i < self.GRID_COLS and self.grid[r][i] == self.grid[r][c]:
                        match.append((i, r))
                        i += 1
                    matches.append({'positions': match})
                    for pos in match: matched_tiles.add(pos)
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r][c] != -1 and (c,r) not in matched_tiles and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    match = [(c, r), (c, r+1), (c, r+2)]
                    i = r + 3
                    while i < self.GRID_ROWS and self.grid[i][c] == self.grid[r][c]:
                        match.append((c, i))
                        i += 1
                    matches.append({'positions': match})
                    for pos in match: matched_tiles.add(pos)
        return matches

    def _remove_matches(self, tiles_to_remove):
        for x, y in tiles_to_remove:
            self.grid[y][x] = -1

    def _apply_gravity_and_refill(self):
        fall_duration = 12
        # Apply gravity
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                        self.animations.append({'type': 'fall', 'pos_start': (c, r), 'pos_end': (c, empty_row), 'progress': 0, 'duration': fall_duration})
                    empty_row -= 1
        
        # Refill board
        refill_delay = 0
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r][c] == -1:
                    self.grid[r][c] = self.np_random.integers(0, len(self.TILE_COLORS))
                    self.animations.append({'type': 'refill', 'pos': (c, r), 'start_y': -1, 'progress': 0, 'duration': fall_duration, 'delay': refill_delay})
            refill_delay += 2

    def _update_animations(self):
        if not self.animations:
            if self.game_state == 'ANIMATING':
                self.game_state = 'IDLE'
            return

        for anim in self.animations[:]:
            anim['progress'] += 1
            if anim['progress'] >= anim.get('duration', 0) + anim.get('delay', 0):
                if anim['type'] == 'callback':
                    anim['func'](*anim['args'])
                self.animations.remove(anim)

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
            if p['life'] <= 0:
                self.particles.remove(p)
                
    def _is_board_clear(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != -1:
                    return False
        return True

    def _find_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Try swapping right
                if c < self.GRID_COLS - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                # Try swapping down
                if r < self.GRID_ROWS - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_COLS * self.TILE_SIZE, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_ROWS * self.TILE_SIZE), 1)
        
        # Draw tiles
        rendered_tiles = set()
        # Draw animated tiles first
        for anim in self.animations:
            p = anim['progress'] / anim['duration']
            if anim['type'] == 'swap' or anim['type'] == 'swap_fail':
                pos1, pos2 = anim['pos1'], anim['pos2']
                if anim['type'] == 'swap_fail':
                    p = 0.5 - abs(p - 0.5) # Animate out and back
                
                x1, y1 = self.GRID_X + pos1[0] * self.TILE_SIZE, self.GRID_Y + pos1[1] * self.TILE_SIZE
                x2, y2 = self.GRID_X + pos2[0] * self.TILE_SIZE, self.GRID_Y + pos2[1] * self.TILE_SIZE
                
                ix1 = x1 + (x2 - x1) * p
                iy1 = y1 + (y2 - y1) * p
                ix2 = x2 + (x1 - x2) * p
                iy2 = y2 + (y1 - y2) * p
                
                self._draw_tile(ix1, iy1, self.grid[pos1[1]][pos1[0]])
                self._draw_tile(ix2, iy2, self.grid[pos2[1]][pos2[0]])
                rendered_tiles.add(pos1)
                rendered_tiles.add(pos2)
            
            elif anim['type'] == 'clear':
                s = self.TILE_SIZE * (1 - p)
                x, y = anim['pos']
                gx, gy = self.GRID_X + x * self.TILE_SIZE, self.GRID_Y + y * self.TILE_SIZE
                self._draw_tile(gx + (self.TILE_SIZE - s)/2, gy + (self.TILE_SIZE - s)/2, self.grid[y][x], size=s)
                rendered_tiles.add((x,y))

            elif anim['type'] == 'fall':
                sx, sy = anim['pos_start']
                ex, ey = anim['pos_end']
                start_x, start_y = self.GRID_X + sx * self.TILE_SIZE, self.GRID_Y + sy * self.TILE_SIZE
                end_x, end_y = self.GRID_X + ex * self.TILE_SIZE, self.GRID_Y + ey * self.TILE_SIZE
                ix = start_x + (end_x - start_x) * p
                iy = start_y + (end_y - start_y) * p
                self._draw_tile(ix, iy, self.grid[ey][ex])
                rendered_tiles.add((sx, sy))
                rendered_tiles.add((ex, ey))

            elif anim['type'] == 'refill' and anim['progress'] > anim.get('delay', 0):
                p_eff = (anim['progress'] - anim.get('delay', 0)) / anim['duration']
                x, y = anim['pos']
                start_y_px = self.GRID_Y + anim['start_y'] * self.TILE_SIZE
                end_y_px = self.GRID_Y + y * self.TILE_SIZE
                ix = self.GRID_X + x * self.TILE_SIZE
                iy = start_y_px + (end_y_px - start_y_px) * p_eff
                self._draw_tile(ix, iy, self.grid[y][x])
                rendered_tiles.add((x,y))

        # Draw static tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (c, r) not in rendered_tiles and self.grid[r][c] != -1:
                    x, y = self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE
                    self._draw_tile(x, y, self.grid[r][c])
        
        # Draw particles
        for p in self.particles:
            color = self.TILE_COLORS[p['color_idx']]
            pygame.draw.rect(self.screen, color, (p['x'], p['y'], max(0, p['size']), max(0, p['size'])))
        
        # Draw cursor
        if self.game_state == 'IDLE':
            cx, cy = self.cursor_pos
            x, y = self.GRID_X + cx * self.TILE_SIZE, self.GRID_Y + cy * self.TILE_SIZE
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 64
            color = (255, 255, 192 + pulse)
            pygame.draw.rect(self.screen, color, (x, y, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=6)

    def _draw_tile(self, x, y, color_idx, size=None):
        if color_idx == -1: return
        size = size if size is not None else self.TILE_SIZE
        color = self.TILE_COLORS[color_idx]
        
        inset = size * 0.1
        rect = pygame.Rect(int(x + inset/2), int(y + inset/2), int(size - inset), int(size - inset))
        
        pygame.gfxdraw.box(self.screen, rect, color)
        pygame.gfxdraw.rectangle(self.screen, rect, tuple(min(255, c+30) for c in color))

    def _spawn_particles(self, grid_x, grid_y, color_idx):
        # sound: match_pop.wav
        center_x = self.GRID_X + grid_x * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.GRID_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(3, 8),
                'color_idx': color_idx
            })

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))
        
        # Combo
        if self.combo_multiplier > 1:
            combo_text = self.font_small.render(f"x{self.combo_multiplier} COMBO!", True, self.TILE_COLORS[self.combo_multiplier % len(self.TILE_COLORS)])
            self.screen.blit(combo_text, (10, 45))

        # Timer bar
        timer_width = 200
        timer_height = 20
        timer_x = self.SCREEN_WIDTH - timer_width - 10
        timer_y = 10
        
        ratio = max(0, self.timer / self.MAX_STEPS)
        
        # Change color based on time left
        if ratio > 0.5:
            bar_color = (80, 200, 80) # Green
        elif ratio > 0.2:
            bar_color = (220, 220, 80) # Yellow
        else:
            bar_color = (200, 80, 80) # Red
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (timer_x, timer_y, timer_width, timer_height))
        pygame.draw.rect(self.screen, bar_color, (timer_x, timer_y, int(timer_width * ratio), timer_height))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self._is_board_clear() else "TIME'S UP!"
            win_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = win_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(win_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "combo": self.combo_multiplier
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you might need to unset the dummy video driver
    # del os.environ['SDL_VIDEODRIVER']
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Match Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()