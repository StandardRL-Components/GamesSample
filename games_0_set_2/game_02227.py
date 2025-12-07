
# Generated: 2025-08-28T04:10:19.974363
# Source Brief: brief_02227.md
# Brief Index: 2227

        
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
        "Controls: Use arrow keys to move the selector. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap. Press Shift to deselect."
    )

    game_description = (
        "Swap adjacent gems to create matches of three or more. "
        "Reach the target score before you run out of moves!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 8
    TILE_SIZE = 40
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * TILE_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_ROWS * TILE_SIZE) // 2
    
    # Animation speeds (in frames)
    SWAP_DURATION = 10
    CLEAR_DURATION = 8
    FALL_DURATION = 12

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 140, 80),  # Orange
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 255, 0)
    
    # Game parameters
    WIN_SCORE = 500
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.np_random = None

        # State variables will be initialized in reset()
        self.grid = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.cursor_pos = [0, 0]
        self.selected_pos = None

        # Game state machine for animations
        self.game_state = 'IDLE' # IDLE, SWAPPING, CLEARING, FALLING
        self.animation_progress = 0
        self.swapping_tiles = []
        self.clearing_tiles = []
        self.falling_tiles = []
        self.particles = []

        # Action handling
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_movement = 0
        self.reward_buffer = 0
        self.last_swap_info = {'valid': False, 'pos1': None, 'pos2': None}

        self.reset()
        
    def _grid_to_screen(self, x, y):
        return self.GRID_MARGIN_X + x * self.TILE_SIZE, self.GRID_MARGIN_Y + y * self.TILE_SIZE

    def _create_grid(self):
        grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                grid[x, y] = self.np_random.integers(len(self.COLORS))
        return grid
    
    def _ensure_no_initial_matches(self):
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for x, y in matches:
                self.grid[x, y] = self.np_random.integers(len(self.COLORS))
    
    def _ensure_possible_moves(self):
        while len(self._find_possible_swaps()) == 0:
            self.grid = self._create_grid()
            self._ensure_no_initial_matches()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.grid = self._create_grid()
        self._ensure_no_initial_matches()
        self._ensure_possible_moves()

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        
        self.game_state = 'IDLE'
        self.animation_progress = 0
        self.swapping_tiles = []
        self.clearing_tiles = []
        self.falling_tiles = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_movement = 0
        self.reward_buffer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = self.reward_buffer
        self.reward_buffer = 0
        
        self._handle_animations()
        
        if self.game_state == 'IDLE' and not self.game_over:
            self._handle_input(action)
        
        terminated = self.game_over
        if not terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.game_over = terminated
        
        self.clock.tick(30)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement (on press, not hold)
        if movement != 0 and movement != self.prev_movement:
            if movement == 1: # Up
                self.cursor_pos[1] -= 1
            elif movement == 2: # Down
                self.cursor_pos[1] += 1
            elif movement == 3: # Left
                self.cursor_pos[0] -= 1
            elif movement == 4: # Right
                self.cursor_pos[0] += 1
            
            # Wrap cursor
            self.cursor_pos[0] %= self.GRID_COLS
            self.cursor_pos[1] %= self.GRID_ROWS

        self.prev_movement = movement if movement != 0 else self.prev_movement
        if movement == 0: self.prev_movement = 0

        # Space press (rising edge)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
                # sfx: select_tile
            else:
                # Attempt swap
                p1 = self.selected_pos
                p2 = self.cursor_pos
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

                if dist == 1: # Adjacent
                    self.steps += 1
                    self._initiate_swap(p1, p2)
                else: # Invalid swap (not adjacent)
                    self.selected_pos = None # Deselect
                    # sfx: invalid_swap

        # Shift press (rising edge)
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.selected_pos is not None:
            self.selected_pos = None
            # sfx: deselect
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _initiate_swap(self, p1, p2):
        self.game_state = 'SWAPPING'
        self.animation_progress = 0
        self.swapping_tiles = [
            {'from': p1, 'to': p2, 'color': self.grid[p1[0], p1[1]]},
            {'from': p2, 'to': p1, 'color': self.grid[p2[0], p2[1]]}
        ]
        
        # Temporarily swap in grid to check for match validity
        self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
        matches = self._find_all_matches()
        self.last_swap_info = {'valid': len(matches) > 0, 'pos1': p1, 'pos2': p2}
        # Swap back, animation will handle visual swap
        self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
        
        self.selected_pos = None

    def _handle_animations(self):
        if self.game_state == 'IDLE':
            return
            
        self.animation_progress += 1

        if self.game_state == 'SWAPPING':
            if self.animation_progress >= self.SWAP_DURATION:
                p1, p2 = self.last_swap_info['pos1'], self.last_swap_info['pos2']
                # Finalize swap in the grid
                self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
                self.swapping_tiles = []
                
                if self.last_swap_info['valid']:
                    self._resolve_board()
                else: # Invalid swap
                    self.reward_buffer += -0.1
                    self.game_state = 'IDLE' # No match, go back to idle
        
        elif self.game_state == 'CLEARING':
            if self.animation_progress >= self.CLEAR_DURATION:
                self.clearing_tiles = []
                self._apply_gravity()
        
        elif self.game_state == 'FALLING':
            if self.animation_progress >= self.FALL_DURATION:
                self.falling_tiles = []
                self._resolve_board()

    def _resolve_board(self):
        matches = self._find_all_matches()
        if matches:
            self._handle_matches(matches)
        else:
            self.game_state = 'IDLE'
            if len(self._find_possible_swaps()) == 0:
                self.game_over = True
                self.reward_buffer += -10

    def _handle_matches(self, matches):
        # sfx: match_found
        match_len = len(matches)
        if match_len == 3: self.reward_buffer += 1
        elif match_len == 4: self.reward_buffer += 2
        else: self.reward_buffer += 3
        
        self.score += len(matches) * 10

        for x, y in matches:
            self.clearing_tiles.append({'pos': (x, y), 'color': self.grid[x, y]})
            self.grid[x, y] = -1 # Mark as empty
            self._create_particles(x, y, self.COLORS[self.clearing_tiles[-1]['color']])
        
        self.game_state = 'CLEARING'
        self.animation_progress = 0

    def _apply_gravity(self):
        self.falling_tiles = []
        for x in range(self.GRID_COLS):
            empty_count = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    # Animate tile falling
                    self.falling_tiles.append({
                        'from_y': y, 'to_y': y + empty_count, 'x': x,
                        'color': self.grid[x, y]
                    })
                    self.grid[x, y + empty_count] = self.grid[x, y]
                    self.grid[x, y] = -1
        
        # Refill top rows
        for x in range(self.GRID_COLS):
            empty_count = sum(1 for y in range(self.GRID_ROWS) if self.grid[x, y] == -1)
            for i in range(empty_count):
                new_color = self.np_random.integers(len(self.COLORS))
                y = empty_count - 1 - i
                self.grid[x, y] = new_color
                self.falling_tiles.append({
                    'from_y': -1 - i, 'to_y': y, 'x': x, 'color': new_color
                })

        if self.falling_tiles:
            self.game_state = 'FALLING'
            self.animation_progress = 0
        else:
            self.game_state = 'IDLE'

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)

    def _find_possible_swaps(self):
        possible_swaps = []
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                # Test swap right
                if x < self.GRID_COLS - 1:
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                    if len(self._find_all_matches()) > 0:
                        possible_swaps.append(((x, y), (x+1, y)))
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                # Test swap down
                if y < self.GRID_ROWS - 1:
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
                    if len(self._find_all_matches()) > 0:
                        possible_swaps.append(((x, y), (x, y+1)))
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
        return possible_swaps

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_MARGIN_X + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_MARGIN_Y), (x, self.GRID_MARGIN_Y + self.GRID_ROWS * self.TILE_SIZE))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_MARGIN_Y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, y), (self.GRID_MARGIN_X + self.GRID_COLS * self.TILE_SIZE, y))

        rendered_tiles = set()

        if self.game_state == 'SWAPPING':
            prog = self.animation_progress / self.SWAP_DURATION
            for tile in self.swapping_tiles:
                sx, sy = self._grid_to_screen(tile['from'][0], tile['from'][1])
                ex, ey = self._grid_to_screen(tile['to'][0], tile['to'][1])
                px = sx + (ex - sx) * prog
                py = sy + (ey - sy) * prog
                self._draw_tile(px, py, tile['color'])
                rendered_tiles.add(tuple(tile['from']))
                rendered_tiles.add(tuple(tile['to']))

        if self.game_state == 'FALLING':
            prog = self.animation_progress / self.FALL_DURATION
            for tile in self.falling_tiles:
                sx, sy = self._grid_to_screen(tile['x'], tile['from_y'])
                ex, ey = self._grid_to_screen(tile['x'], tile['to_y'])
                py = sy + (ey - sy) * prog
                self._draw_tile(sx, py, tile['color'])
                
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if (x,y) in rendered_tiles or self.grid[x,y] == -1:
                    continue
                
                is_clearing = any(c['pos'] == (x, y) for c in self.clearing_tiles)
                if is_clearing:
                    prog = self.animation_progress / self.CLEAR_DURATION
                    scale = 1.0 - prog
                    color = (0,0,0)
                    for c_tile in self.clearing_tiles:
                        if c_tile['pos'] == (x,y):
                            color = self.COLORS[c_tile['color']]
                            break
                    sx, sy = self._grid_to_screen(x, y)
                    self._draw_tile(sx, sy, color, scale, is_color=True)
                else:
                    is_falling_dest = any(f['to_y'] == y and f['x'] == x for f in self.falling_tiles)
                    if not is_falling_dest:
                        sx, sy = self._grid_to_screen(x, y)
                        self._draw_tile(sx, sy, self.grid[x, y])
        
        self._update_and_draw_particles()

        if not self.game_over:
            cx, cy = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
            rect = pygame.Rect(cx, cy, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=8)

            if self.selected_pos is not None:
                sx, sy = self._grid_to_screen(self.selected_pos[0], self.selected_pos[1])
                rect = pygame.Rect(sx, sy, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 4, border_radius=8)
    
    def _draw_tile(self, px, py, color_data, scale=1.0, is_color=False):
        color = color_data if is_color else self.COLORS[color_data]
        
        size = int(self.TILE_SIZE * scale)
        offset = (self.TILE_SIZE - size) // 2
        rect = pygame.Rect(int(px + offset), int(py + offset), size, size)
        
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        highlight_color = tuple(min(255, c + 40) for c in color)
        pygame.gfxdraw.arc(self.screen, rect.centerx, rect.centery, int(size * 0.35), 135, 315, highlight_color)

    def _create_particles(self, grid_x, grid_y, color):
        sx, sy = self._grid_to_screen(grid_x, grid_y)
        center_x, center_y = sx + self.TILE_SIZE / 2, sy + self.TILE_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(15, 30)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            p[2] *= 0.98
            p[3] *= 0.98
            
            radius = int(p[4] / 6)
            if radius > 0:
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), radius)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))
        
        steps_text = self.font_small.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, (255, 255, 255))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            self.screen.blit(overlay, (0,0))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()