# Generated: 2025-08-28T03:07:16.880659
# Source Brief: brief_01919.md
# Brief Index: 1919

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, then move to an adjacent tile and press Shift to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent tiles to create matches of three or more. Clear tiles to score points, but watch your limited moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    TILE_SIZE = 40
    BOARD_WIDTH = GRID_WIDTH * TILE_SIZE
    BOARD_HEIGHT = GRID_HEIGHT * TILE_SIZE
    BOARD_X_OFFSET = (SCREEN_WIDTH - BOARD_WIDTH) // 2
    BOARD_Y_OFFSET = (SCREEN_HEIGHT - BOARD_HEIGHT) // 2
    
    INITIAL_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 200, 0)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...        
        
        # Initialize state variables
        self.board = None
        self.cursor_pos = None
        self.selected_tile_1 = None
        self.particles = None
        self.steps = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        
        # Note: self.reset() is called by the wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tile_1 = None
        self.particles = []

        self._generate_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        
        # --- Handle Input ---
        self._handle_movement(movement)
        
        if space_held:
            # Sound: select_tile.wav
            self.selected_tile_1 = list(self.cursor_pos)
        
        if shift_held and self.selected_tile_1 is not None:
            reward += self._attempt_swap()
            self.selected_tile_1 = None # Reset selection after swap attempt

        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated:
            if self.moves_left <= 0:
                reward -= 50 # Loss penalty
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Helper Methods for Game Logic ---

    def _generate_board(self):
        self.board = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        # Ensure no initial matches
        while True:
            matches = self._find_all_matches()
            if not np.any(matches):
                break
            # Replace matched tiles with new random ones
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if matches[x, y]:
                        self.board[x, y] = self.np_random.integers(0, self.NUM_COLORS)

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _attempt_swap(self):
        pos1 = self.selected_tile_1
        pos2 = self.cursor_pos
        reward = 0

        if not self._is_adjacent(pos1, pos2):
            # Sound: invalid_swap.wav
            return 0 # No-op for non-adjacent swaps

        self.moves_left -= 1
        
        # Perform swap
        self.board[pos1[0], pos1[1]], self.board[pos2[0], pos2[1]] = self.board[pos2[0], pos2[1]], self.board[pos1[0], pos1[1]]

        # Check for matches and handle cascades
        total_cleared_in_turn = 0
        chain_reaction_bonus = 0
        
        while True:
            matches = self._find_all_matches()
            num_cleared = np.sum(matches)

            if num_cleared == 0:
                break
            
            # Sound: match_clear.wav
            total_cleared_in_turn += num_cleared
            reward += self._calculate_reward(num_cleared, chain_reaction_bonus > 0)
            
            if chain_reaction_bonus > 0:
                reward += chain_reaction_bonus # Add combo bonus
            chain_reaction_bonus += 5 # Increase bonus for next chain
            
            self.score += num_cleared

            # Create particles and clear tiles
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if matches[x, y]:
                        self._create_particles(x, y, self.board[x, y])
                        self.board[x, y] = -1 # Mark as empty

            self._apply_gravity()
            self._fill_top_rows()

        if total_cleared_in_turn == 0:
            # No match found, swap back and penalize
            self.board[pos1[0], pos1[1]], self.board[pos2[0], pos2[1]] = self.board[pos2[0], pos2[1]], self.board[pos1[0], pos1[1]]
            reward -= 0.1 # Penalty for non-matching swap
        else:
            # Check for win condition (very unlikely but possible)
            if np.all(self.board == -1):
                reward += 100
                self.game_over = True

        return reward

    def _find_all_matches(self):
        matches = np.zeros_like(self.board, dtype=bool)
        # Check horizontal matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.board[x, y] != -1 and self.board[x, y] == self.board[x+1, y] == self.board[x+2, y]:
                    matches[x, y] = matches[x+1, y] = matches[x+2, y] = True
        # Check vertical matches
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.board[x, y] != -1 and self.board[x, y] == self.board[x, y+1] == self.board[x, y+2]:
                    matches[x, y] = matches[x, y+1] = matches[x, y+2] = True
        return matches

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[x, y] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    # Move tile down to the highest empty slot
                    dest_y = empty_slots.pop(0)
                    self.board[x, dest_y] = self.board[x, y]
                    self.board[x, y] = -1
                    empty_slots.append(y)

    def _fill_top_rows(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.board[x, y] == -1:
                    self.board[x, y] = self.np_random.integers(0, self.NUM_COLORS)

    def _create_particles(self, grid_x, grid_y, color_index):
        center_x = self.BOARD_X_OFFSET + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.BOARD_Y_OFFSET + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_index]
        for _ in range(15): # Number of particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([center_x, center_y, vx, vy, lifetime, color])

    def _calculate_reward(self, num_cleared=0, is_chain=False):
        reward = num_cleared * 1 # +1 per tile
        if num_cleared == 3: reward += 5
        elif num_cleared == 4: reward += 10
        elif num_cleared >= 5: reward += 20
        return reward

    def _check_termination(self):
        # The result of np.all() is a numpy.bool_, which must be cast to a standard Python bool.
        is_terminated = self.moves_left <= 0 or self.steps >= self.MAX_STEPS or np.all(self.board == -1)
        return bool(is_terminated)

    # --- Gymnasium Interface Methods ---

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
            "cursor_pos": list(self.cursor_pos),
            "selected_tile": self.selected_tile_1,
        }
        
    def close(self):
        pygame.quit()

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_board_grid()
        self._render_tiles()
        self._render_cursor()
        self._update_and_render_particles()

    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_main, 20, 20, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_text(f"Moves: {self.moves_left}", self.font_main, self.SCREEN_WIDTH - 150, 20, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((20, 25, 40, 200))
            self.screen.blit(overlay, (0, 0))
            
            self._render_text("Game Over", self.font_large, self.SCREEN_WIDTH / 2 - self.font_large.size("Game Over")[0] / 2, self.SCREEN_HEIGHT / 2 - 40, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
            self._render_text(f"Final Score: {self.score}", self.font_main, self.SCREEN_WIDTH / 2 - self.font_main.size(f"Final Score: {self.score}")[0] / 2, self.SCREEN_HEIGHT / 2 + 20, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _render_text(self, text, font, x, y, color, shadow_color):
        text_surface = font.render(text, True, shadow_color)
        self.screen.blit(text_surface, (x + 2, y + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def _render_board_grid(self):
        grid_rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET, self.BOARD_WIDTH, self.BOARD_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=8)

    def _render_tiles(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                tile_val = self.board[x, y]
                if tile_val == -1:
                    continue
                
                color = self.TILE_COLORS[tile_val]
                darker_color = tuple(max(0, c - 40) for c in color)
                
                rect_x = self.BOARD_X_OFFSET + x * self.TILE_SIZE
                rect_y = self.BOARD_Y_OFFSET + y * self.TILE_SIZE
                
                tile_rect = pygame.Rect(rect_x + 3, rect_y + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6)
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=8)
                
                inner_rect = pygame.Rect(rect_x + 6, rect_y + 6, self.TILE_SIZE - 12, self.TILE_SIZE - 12)
                pygame.draw.rect(self.screen, darker_color, inner_rect, border_radius=6)

    def _render_cursor(self):
        # Selection highlight for first tile
        if self.selected_tile_1 is not None:
            sel_x, sel_y = self.selected_tile_1
            rect_x = self.BOARD_X_OFFSET + sel_x * self.TILE_SIZE
            rect_y = self.BOARD_Y_OFFSET + sel_y * self.TILE_SIZE
            sel_rect = pygame.Rect(rect_x, rect_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, sel_rect, 4, border_radius=10)

        # Main cursor
        cursor_x, cursor_y = self.cursor_pos
        rect_x = self.BOARD_X_OFFSET + cursor_x * self.TILE_SIZE
        rect_y = self.BOARD_Y_OFFSET + cursor_y * self.TILE_SIZE
        cursor_rect = pygame.Rect(rect_x, rect_y, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Oscillates between 0 and 1
        line_width = int(2 + pulse * 3)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=10)

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime -= 1
            if p[4] > 0:
                active_particles.append(p)
                alpha = max(0, min(255, int(255 * (p[4] / 30))))
                # Use gfxdraw for alpha blending
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 2, p[5] + (alpha,))
        self.particles = active_particles

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is imported by RL libs
    
    # To run with a real window, we need a separate pygame screen
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Puzzle")
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    while not done:
        # Action defaults to no-op
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1

        action = [movement, space, shift]
        
        # Only step if an action is taken, since auto_advance is False
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else: # If no action, just get the current observation for rendering
            obs = env._get_observation()

        # Render the observation from the environment to the real screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # If game is over, wait for a key press to reset
        if done:
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        done = True # to exit outer loop
                    if event.type == pygame.KEYDOWN:
                        if event.key != pygame.K_ESCAPE:
                            waiting_for_reset = False
            if not done: # if we didn't quit, reset the game
                obs, info = env.reset()
                done = False

    env.close()