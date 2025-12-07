
# Generated: 2025-08-28T06:55:46.579090
# Source Brief: brief_03079.md
# Brief Index: 3079

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile. "
        "Move the cursor to an adjacent tile and press Space again to swap. "
        "Press Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Plan your moves to create combos and reach the target score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_TILES = 5
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WIN_SCORE = 1000
    MAX_MOVES = 30
    MAX_STEPS = MAX_MOVES * 3 # A generous upper bound

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG_DARK = (30, 35, 50)
    COLOR_GRID_BG_LIGHT = (35, 40, 55)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # Game state variables
        self.board = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.selected_tile = None
        
        # For visual effects within a single step
        self.animation_flash_tiles = set()

        self.grid_top = 60
        self.grid_left = (self.SCREEN_WIDTH - (self.GRID_WIDTH * 40)) // 2
        self.tile_size = 40
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tile = None
        self.animation_flash_tiles = set()
        
        self._generate_board()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean
        
        reward = 0
        terminated = False
        
        self.animation_flash_tiles = set()

        # 1. Handle deselect
        if shift_pressed:
            self.selected_tile = None

        # 2. Handle movement
        if movement != 0:
            cx, cy = self.cursor_pos
            if movement == 1: cy = max(0, cy - 1) # Up
            elif movement == 2: cy = min(self.GRID_HEIGHT - 1, cy + 1) # Down
            elif movement == 3: cx = max(0, cx - 1) # Left
            elif movement == 4: cx = min(self.GRID_WIDTH - 1, cx + 1) # Right
            self.cursor_pos = (cx, cy)

        # 3. Handle selection and swapping
        if space_pressed:
            if self.selected_tile is None:
                self.selected_tile = self.cursor_pos
            else:
                if self._is_adjacent(self.selected_tile, self.cursor_pos):
                    reward, terminated = self._attempt_swap(self.selected_tile, self.cursor_pos)
                elif self.selected_tile == self.cursor_pos:
                    self.selected_tile = None # Deselect if pressing space on same tile
                else: 
                    self.selected_tile = self.cursor_pos # Change selection

        self.steps += 1
        
        if not terminated: # Check for other termination conditions
            if self.moves_left <= 0:
                reward += -10 if self.score < self.WIN_SCORE else 0
                terminated = True
            elif not self._has_possible_moves():
                reward += -10 if self.score < self.WIN_SCORE else 0
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Perform swap
        self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
        
        matches = self._find_all_matches()
        
        if not matches:
            # Invalid swap, swap back
            self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
            self.selected_tile = None
            return -0.1, False

        # Valid swap, process matches
        # Sound: match_start.wav
        total_reward = 0
        chain = 0
        while matches:
            chain += 1
            
            num_cleared = len(matches)
            total_reward += num_cleared
            
            # Bonus for larger matches
            temp_board = np.full_like(self.board, -1)
            for x,y in matches: temp_board[y,x] = self.board[y,x]
            
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if temp_board[y,x] != -1:
                        h_len = 1
                        while x + h_len < self.GRID_WIDTH and temp_board[y, x+h_len] == temp_board[y,x]: h_len += 1
                        if h_len == 4: total_reward += 5
                        if h_len >= 5: total_reward += 10
                        v_len = 1
                        while y + v_len < self.GRID_HEIGHT and temp_board[y+v_len, x] == temp_board[y,x]: v_len += 1
                        if v_len == 4: total_reward += 5
                        if v_len >= 5: total_reward += 10
            
            # Sound: combo_#{chain}.wav
            self.score += num_cleared * 10 * chain
            self.animation_flash_tiles.update(matches)
            
            for x, y in matches:
                self.board[y, x] = -1

            self._apply_gravity_and_refill()
            matches = self._find_all_matches()

        self.selected_tile = None
        
        if self.score >= self.WIN_SCORE:
            total_reward += 100
            return total_reward, True

        return total_reward, False

    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(0, self.NUM_TILES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_all_matches():
                matches = self._find_all_matches()
                for x, y in matches:
                    self.board[y, x] = self.np_random.integers(0, self.NUM_TILES)
            if self._has_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.board[y, x] == self.board[y, x+1] == self.board[y, x+2] and self.board[y,x] != -1:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.board[y, x] == self.board[y+1, x] == self.board[y+2, x] and self.board[y,x] != -1:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches
        
    def _has_possible_moves(self):
        temp_board = self.board.copy()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if x < self.GRID_WIDTH - 1:
                    temp_board[y,x], temp_board[y,x+1] = temp_board[y,x+1], temp_board[y,x]
                    if self._find_matches_on_board(temp_board): return True
                    temp_board[y,x], temp_board[y,x+1] = temp_board[y,x+1], temp_board[y,x]
                if y < self.GRID_HEIGHT - 1:
                    temp_board[y,x], temp_board[y+1,x] = temp_board[y+1,x], temp_board[y,x]
                    if self._find_matches_on_board(temp_board): return True
                    temp_board[y,x], temp_board[y+1,x] = temp_board[y+1,x], temp_board[y,x]
        return False

    def _find_matches_on_board(self, board):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if board[y,x] == board[y,x+1] == board[y,x+2] and board[y,x]!=-1: return True
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if board[y,x] == board[y+1,x] == board[y+2,x] and board[y,x]!=-1: return True
        return False

    def _apply_gravity_and_refill(self):
        # Sound: fall.wav
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[y, x] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.board[y + empty_slots, x] = self.board[y, x]
                    self.board[y, x] = -1
            for i in range(empty_slots):
                self.board[i, x] = self.np_random.integers(0, self.NUM_TILES)

    def _is_adjacent(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.grid_left + x * self.tile_size, self.grid_top + y * self.tile_size, self.tile_size, self.tile_size)
                color = self.COLOR_GRID_BG_LIGHT if (x + y) % 2 == 0 else self.COLOR_GRID_BG_DARK
                pygame.draw.rect(self.screen, color, rect)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.board[y, x] != -1:
                    self._draw_tile(x, y, self.board[y, x])

        if self.animation_flash_tiles:
            # Sound: flash.wav
            flash_surface = pygame.Surface((self.tile_size - 4, self.tile_size - 4), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, 180))
            for x, y in self.animation_flash_tiles:
                self.screen.blit(flash_surface, (self.grid_left + x * self.tile_size + 2, self.grid_top + y * self.tile_size + 2))
        
        if self.selected_tile:
            sx, sy = self.selected_tile
            rect = pygame.Rect(self.grid_left + sx * self.tile_size, self.grid_top + sy * self.tile_size, self.tile_size, self.tile_size)
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
            color = (255, 255, 255)
            pygame.draw.rect(self.screen, color, rect, int(1 + 2 * pulse))

        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.grid_left + cx * self.tile_size, self.grid_top + cy * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

    def _draw_tile(self, x, y, tile_val):
        base_color = self.TILE_COLORS[tile_val]
        rect = pygame.Rect(self.grid_left + x * self.tile_size + 4, self.grid_top + y * self.tile_size + 4, self.tile_size - 8, self.tile_size - 8)
        
        highlight_color = tuple(min(255, c + 60) for c in base_color)
        shadow_color = tuple(max(0, c - 60) for c in base_color)
        
        pygame.draw.rect(self.screen, shadow_color, rect.move(2, 2))
        pygame.draw.rect(self.screen, base_color, rect)
        
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 1)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            status = "You Win!" if self.score >= self.WIN_SCORE else "Game Over"
            status_text = self.font_large.render(status, True, (255,255,100))
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if terminated:
                    if event.key == pygame.K_r:
                        terminated = False
                        obs, info = env.reset()
                    continue

                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Allow reset mid-game
                    terminated = False
                    obs, info = env.reset()
                    continue
        
        if event_happened and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")
        
        # We need to re-render even without an action to see pulsing animations
        current_obs = env._get_observation()
        frame = np.transpose(current_obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.flip()
        
        clock.tick(30)

    env.close()