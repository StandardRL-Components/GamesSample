
# Generated: 2025-08-28T01:01:20.660383
# Source Brief: brief_03970.md
# Brief Index: 3970

        
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
    """
    A tile-matching puzzle game environment compliant with the Gymnasium API.
    The player swaps adjacent tiles to create matches of three or more, aiming
    to achieve a high score within a limited number of moves. The game features
    smooth animations for swaps, tile clearing, and falling, creating a polished
    visual experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile, then move to an adjacent tile and press Space again to swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of three or more. Plan your moves carefully to maximize your score before you run out of turns!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 8
    GRID_COLS = 8
    TILE_SIZE = 40
    TILE_MARGIN = 4
    GRID_WIDTH = GRID_COLS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_HEIGHT = GRID_ROWS * (TILE_SIZE + TILE_MARGIN) - TILE_MARGIN
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    NUM_TILE_TYPES = 6
    MAX_MOVES = 25
    TARGET_SCORE = 250
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    CURSOR_COLOR = (255, 255, 255)
    SELECT_COLOR = (255, 165, 0) # Orange

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_state = 'INPUT'
        self.animations = []
        self.particles = []
        self.turn_reward = 0.0
        
        # Track button presses to handle discrete events from continuous holds
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state variables
        self.reset()
    
    def _generate_board(self):
        """Generates a valid initial board with no starting matches and at least one possible move."""
        board = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while True:
            matches = self._find_all_matches(board)
            if not matches:
                break
            for r, c in matches:
                board[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
        
        if not self._has_possible_moves(board):
            return self._generate_board() # Recurse until a valid board is made

        return board

    def _has_possible_moves(self, board):
        """Checks if any valid match-creating moves exist on the board."""
        temp_board = np.copy(board)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    temp_board[r, c], temp_board[r, c + 1] = temp_board[r, c + 1], temp_board[r, c]
                    if self._find_all_matches(temp_board):
                        return True
                    temp_board[r, c], temp_board[r, c + 1] = temp_board[r, c + 1], temp_board[r, c] # Swap back
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    temp_board[r, c], temp_board[r + 1, c] = temp_board[r + 1, c], temp_board[r, c]
                    if self._find_all_matches(temp_board):
                        return True
                    temp_board[r, c], temp_board[r + 1, c] = temp_board[r + 1, c], temp_board[r, c] # Swap back
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_board()
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_tile = None
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.game_state = 'INPUT'
        self.animations = []
        self.particles = []
        self.turn_reward = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if self.game_state == 'INPUT' and not self.game_over:
            reward = self._handle_input(movement, space_pressed, shift_pressed)
        else:
            self._update_animations()

        if self.game_state == 'INPUT' and self.turn_reward != 0.0:
            reward += self.turn_reward
            self.turn_reward = 0.0

        terminated = self._check_termination()
        if terminated and not self.game_over:
            # Assign terminal rewards only once
            if self.score >= self.TARGET_SCORE:
                reward += 100.0 # Win
            else:
                reward -= 100.0 # Loss
            self.game_over = True

        # Safety termination
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            self.game_over = True
            reward -= 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        """Processes player actions when the game is in the 'INPUT' state."""
        reward = 0.0
        
        # Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

        if shift_pressed: self.selected_tile = None
        
        if space_pressed:
            if self.selected_tile is None:
                self.selected_tile = list(self.cursor_pos)
            elif self.selected_tile == list(self.cursor_pos):
                self.selected_tile = None
            elif self._is_adjacent(self.selected_tile, self.cursor_pos):
                self.moves_left -= 1
                self._start_swap(self.selected_tile, self.cursor_pos)
                self.selected_tile = None
            else: # Invalid swap (not adjacent)
                self.selected_tile = list(self.cursor_pos)
                reward = -0.1
        
        return reward

    def _update_animations(self):
        """Progresses all active animations and handles state transitions upon completion."""
        if not self.animations:
            # This logic handles state transitions that don't have an animation object
            if self.game_state == 'MATCH': self._handle_matches()
            elif self.game_state == 'FALL': self._handle_fall()
            elif self.game_state == 'REFILL': self._handle_refill()
            return

        for anim in self.animations:
            anim['progress'] += 1.0 / anim['duration']
        
        finished_anims = [anim for anim in self.animations if anim['progress'] >= 1.0]
        self.animations = [anim for anim in self.animations if anim['progress'] < 1.0]

        if finished_anims and not self.animations:
             self._on_animation_finish(finished_anims[0])

    def _on_animation_finish(self, anim):
        """Callback for when an animation or group of animations finishes."""
        if anim['type'] == 'swap':
            r1, c1 = anim['pos1']
            r2, c2 = anim['pos2']
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

            if anim['swap_back']:
                self.game_state = 'INPUT'
                self.turn_reward += -0.1
            else:
                self.game_state = 'MATCH'
                self._handle_matches(original_swap=(anim['pos1'], anim['pos2']))
        
        elif anim['type'] in ['clear', 'fall', 'refill']:
            # These animations can happen in parallel, so we transition state
            # only after all of them are done.
            if anim['type'] == 'clear': self.game_state = 'FALL'
            elif anim['type'] == 'fall': self.game_state = 'REFILL'
            elif anim['type'] == 'refill': self.game_state = 'MATCH'

    def _handle_matches(self, original_swap=None):
        """Finds matches, calculates rewards, and initiates clear animations."""
        matches = self._find_all_matches(self.grid)

        if not matches:
            if original_swap:
                self._start_swap(original_swap[0], original_swap[1], swap_back=True)
            else:
                self.game_state = 'INPUT'
            return
        
        # sfx: match_found.wav
        num_cleared = len(matches)
        self.turn_reward += num_cleared * 1.0
        if num_cleared > 3: self.turn_reward += 5.0
        self.score += int(num_cleared * 1.0 + (5.0 if num_cleared > 3 else 0.0))

        for r, c in matches:
            self.animations.append({'type': 'clear', 'pos': (r, c), 'color': self.TILE_COLORS[self.grid[r, c]], 'progress': 0.0, 'duration': 15})
            for _ in range(5): self.particles.append(self._create_particle(r, c, self.TILE_COLORS[self.grid[r, c]]))
            self.grid[r, c] = -1 # Mark as empty
        self.game_state = 'CLEAR'

    def _handle_fall(self):
        """Creates animations for tiles falling into empty spaces."""
        moved = False
        for c in range(self.GRID_COLS):
            empty_slots = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[r + empty_slots, c] = self.grid[r, c]
                    self.grid[r, c] = -1
                    self.animations.append({'type': 'fall', 'from_pos': (r, c), 'to_pos': (r + empty_slots, c), 'color': self.TILE_COLORS[self.grid[r + empty_slots, c]], 'progress': 0.0, 'duration': 8 + empty_slots * 2})
                    moved = True
        if not moved: self.game_state = 'REFILL'
        else: self.game_state = 'FALL'

    def _handle_refill(self):
        """Creates animations for new tiles falling from the top."""
        moved = False
        for c in range(self.GRID_COLS):
            empty_count = sum(1 for r in range(self.GRID_ROWS) if self.grid[r, c] == -1)
            for i in range(empty_count):
                r = empty_count - 1 - i
                self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
                self.animations.append({'type': 'refill', 'from_pos': (-1 - i, c), 'to_pos': (r, c), 'color': self.TILE_COLORS[self.grid[r, c]], 'progress': 0.0, 'duration': 10 + i * 2})
                moved = True
        if not moved: self.game_state = 'MATCH'
        else: self.game_state = 'REFILL'

    def _find_all_matches(self, board):
        """Iterates over the board and returns a set of all tile coordinates in a match."""
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if board[r, c] != -1 and board[r, c] == board[r, c + 1] == board[r, c + 2]:
                    for i in range(c, self.GRID_COLS):
                        if board[r, c] == board[r, i]: matches.add((r, i))
                        else: break
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if board[r, c] != -1 and board[r, c] == board[r + 1, c] == board[r + 2, c]:
                    for i in range(r, self.GRID_ROWS):
                        if board[r, c] == board[i, c]: matches.add((i, c))
                        else: break
        return matches

    def _check_termination(self):
        """Checks for win/loss conditions."""
        if self.score >= self.TARGET_SCORE: return True
        if self.moves_left <= 0 and self.game_state == 'INPUT': return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        """Renders all primary game elements, including tiles, animations, and particles."""
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X - 10, self.GRID_Y - 10, self.GRID_WIDTH + 20, self.GRID_HEIGHT + 20), border_radius=10)

        visible_tiles = set((r, c) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))
        for anim in self.animations:
            if anim['type'] == 'swap':
                visible_tiles.discard(tuple(anim['pos1'])); visible_tiles.discard(tuple(anim['pos2']))
            elif anim['type'] in ['clear', 'fall']:
                visible_tiles.discard(tuple(anim.get('from_pos', anim.get('pos'))))
            elif anim['type'] == 'refill':
                visible_tiles.discard(tuple(anim['to_pos']))

        for r, c in visible_tiles:
            if self.grid[r, c] != -1:
                self._draw_tile(r, c, self.TILE_COLORS[self.grid[r, c]])

        self._render_animations()
        self._render_particles()
        if not self.game_over: self._render_cursors()

    def _render_animations(self):
        """Draws elements that are currently being animated."""
        for anim in self.animations:
            p = anim['progress']
            if anim['type'] == 'swap':
                r1, c1 = anim['pos1']; r2, c2 = anim['pos2']
                x1, y1 = self._get_tile_screen_pos(r1, c1); x2, y2 = self._get_tile_screen_pos(r2, c2)
                self._draw_tile_at_pixel(int(x1 + (x2 - x1) * p), int(y1 + (y2 - y1) * p), self.TILE_COLORS[self.grid[r2, c2] if not anim['swap_back'] else self.grid[r1,c1]])
                self._draw_tile_at_pixel(int(x2 + (x1 - x2) * p), int(y2 + (y1 - y2) * p), self.TILE_COLORS[self.grid[r1, c1] if not anim['swap_back'] else self.grid[r2,c2]])
            elif anim['type'] == 'clear':
                x, y = self._get_tile_screen_pos(*anim['pos'])
                self._draw_tile_at_pixel(x, y, anim['color'], scale=1.0 - p, alpha=int(255 * (1.0 - p)))
            elif anim['type'] in ['fall', 'refill']:
                x1, y1 = self._get_tile_screen_pos(*anim['from_pos']); x2, y2 = self._get_tile_screen_pos(*anim['to_pos'])
                self._draw_tile_at_pixel(int(x1 + (x2 - x1) * p), int(y1 + (y2 - y1) * p), anim['color'])

    def _render_particles(self):
        """Draws and updates all active particles."""
        for p in self.particles[:]:
            p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += 0.2; p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p); continue
            size = max(0, (self.TILE_SIZE / 8) * (p['life'] / 20))
            pygame.draw.rect(self.screen, p['color'], (p['x'] - size/2, p['y'] - size/2, size, size))

    def _render_cursors(self):
        """Draws the player's cursor and the highlight for a selected tile."""
        cx, cy = self._get_tile_screen_pos(*self.cursor_pos)
        pygame.draw.rect(self.screen, self.CURSOR_COLOR, (cx - self.TILE_SIZE//2, cy - self.TILE_SIZE//2, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=8)
        if self.selected_tile:
            sx, sy = self._get_tile_screen_pos(*self.selected_tile)
            pulse = abs(math.sin(self.steps * 0.2))
            color = [int(c1 + (c2 - c1) * pulse) for c1, c2 in zip(self.CURSOR_COLOR, self.SELECT_COLOR)]
            pygame.draw.rect(self.screen, color, (sx - self.TILE_SIZE//2, sy - self.TILE_SIZE//2, self.TILE_SIZE, self.TILE_SIZE), 4, border_radius=8)

    def _render_ui(self):
        """Renders the user interface, including score, moves, and game over text."""
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20)))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.TARGET_SCORE else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _draw_tile(self, r, c, color):
        x, y = self._get_tile_screen_pos(r, c)
        self._draw_tile_at_pixel(x, y, color)

    def _draw_tile_at_pixel(self, x, y, color, scale=1.0, alpha=255):
        size = int(self.TILE_SIZE * scale)
        if size <= 1: return
        rect = pygame.Rect(x - size//2, y - size//2, size, size)
        
        if alpha < 255:
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*color, alpha), (0, 0, size, size), border_radius=8)
            self.screen.blit(temp_surf, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        highlight_color = tuple(min(255, c+50) for c in color)
        pygame.draw.rect(self.screen, highlight_color, rect.inflate(-size*0.6, -size*0.6), border_radius=5)
    
    # --- Helper methods ---
    def _is_adjacent(self, pos1, pos2): return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    def _get_tile_screen_pos(self, r, c): return (self.GRID_X + c * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE // 2, self.GRID_Y + r * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE // 2)
    def _start_swap(self, pos1, pos2, swap_back=False): self.game_state = 'SWAP'; self.animations.append({'type': 'swap', 'pos1': pos1, 'pos2': pos2, 'progress': 0.0, 'duration': 10, 'swap_back': swap_back})
    def _create_particle(self, r, c, color): x, y = self._get_tile_screen_pos(r, c); angle = self.np_random.uniform(0, 2 * math.pi); speed = self.np_random.uniform(2, 5); return {'x': x, 'y': y, 'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed, 'life': 20, 'color': color}
    def close(self): pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    pygame.display.set_caption("Tile Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    running, terminated = True, False

    while running:
        current_action = [0, 0, 0] # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: current_action[0] = 1
            elif keys[pygame.K_DOWN]: current_action[0] = 2
            elif keys[pygame.K_LEFT]: current_action[0] = 3
            elif keys[pygame.K_RIGHT]: current_action[0] = 4
            if keys[pygame.K_SPACE]: current_action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: current_action[2] = 1

        obs, reward, terminated, truncated, info = env.step(current_action)
        if reward != 0: print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        if terminated and not env.game_over: print("Game Over!")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        env.clock.tick(60)
    env.close()