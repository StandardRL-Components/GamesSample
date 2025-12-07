
# Generated: 2025-08-27T14:31:18.737490
# Source Brief: brief_00700.md
# Brief Index: 700

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select/deselect a crystal. "
        "When a crystal is selected, use arrow keys to swap it with an adjacent one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based isometric puzzle game. Swap crystals to create matches of three or more. "
        "Clear the entire board within the move limit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 6
    MAX_MOVES = 10
    NUM_CRYSTAL_TYPES = 3

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_TILE = (45, 50, 61)
    CRYSTAL_COLORS = [
        (255, 80, 100),   # Red
        (80, 255, 150),  # Green
        (80, 150, 255),  # Blue
    ]
    CRYSTAL_HIGHLIGHTS = [
        (255, 150, 170),
        (150, 255, 200),
        (150, 200, 255),
    ]
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (35, 40, 50)
    
    # Isometric rendering parameters
    TILE_WIDTH_ISO = 48
    TILE_HEIGHT_ISO = 24
    TILE_DEPTH_ISO = 16

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...        
        self.iso_offset_x = self.SCREEN_WIDTH // 2
        self.iso_offset_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_ISO // 2) + 20

        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_pos = None
        self.last_match_effects = []
        self.sparkle_particles = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _grid_to_iso(self, r, c):
        x = self.iso_offset_x + (c - r) * self.TILE_WIDTH_ISO // 2
        y = self.iso_offset_y + (c + r) * self.TILE_HEIGHT_ISO // 2
        return int(x), int(y)

    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            
            # Ensure no initial matches
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.board[r, c] = 0
                self._apply_gravity()
                # Refill top rows
                for c in range(self.GRID_WIDTH):
                    for r in range(self.GRID_HEIGHT):
                        if self.board[r, c] == 0:
                            self.board[r, c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)
            
            # Ensure at least one move is possible
            if self._has_possible_move():
                break

    def _has_possible_move(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Test swap right
                if c < self.GRID_WIDTH - 1:
                    self.board[r, c], self.board[r, c + 1] = self.board[r, c + 1], self.board[r, c]
                    if self._find_matches():
                        self.board[r, c], self.board[r, c + 1] = self.board[r, c + 1], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r, c + 1] = self.board[r, c + 1], self.board[r, c]
                # Test swap down
                if r < self.GRID_HEIGHT - 1:
                    self.board[r, c], self.board[r + 1, c] = self.board[r + 1, c], self.board[r, c]
                    if self._find_matches():
                        self.board[r, c], self.board[r + 1, c] = self.board[r + 1, c], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r + 1, c] = self.board[r + 1, c], self.board[r, c]
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_board()
        
        # Initialize all game state, for example:
        self.steps = 0
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_pos = None
        self.last_match_effects = []
        self.sparkle_particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        self.last_match_effects.clear()

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_press = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Handle player action ---
        if self.selected_pos is None:
            # --- CURSOR MOVEMENT AND SELECTION ---
            if movement != 0:
                dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dr, 0, self.GRID_HEIGHT - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dc, 0, self.GRID_WIDTH - 1)
            
            if space_press and self.board[self.cursor_pos[0], self.cursor_pos[1]] != 0:
                self.selected_pos = list(self.cursor_pos)
                # sfx: select crystal
        else:
            # --- CRYSTAL SWAP ---
            if space_press:
                self.selected_pos = None # Deselect
                # sfx: deselect
            elif movement != 0:
                self.moves_left -= 1
                
                dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
                r1, c1 = self.selected_pos
                r2, c2 = r1 + dr, c1 + dc

                if 0 <= r2 < self.GRID_HEIGHT and 0 <= c2 < self.GRID_WIDTH:
                    # Perform swap
                    self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
                    # sfx: swap

                    total_cleared_this_turn = 0
                    while True:
                        matches = self._find_matches()
                        if not matches:
                            break
                        
                        # sfx: match found
                        total_cleared_this_turn += len(matches)
                        for r, c in matches:
                            self.last_match_effects.append(self._grid_to_iso(r, c))
                            self.board[r, c] = 0
                        
                        self._apply_gravity()

                    if total_cleared_this_turn > 0:
                        reward += total_cleared_this_turn
                        self.score += total_cleared_this_turn
                    else:
                        # Invalid move, swap back
                        self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
                        reward = -0.2
                        # sfx: invalid move
                else: # Out of bounds move
                    reward = -0.2
                    # sfx: invalid move

                self.selected_pos = None # Deselect after move attempt

        # --- Check Termination Conditions ---
        if np.sum(self.board) == 0:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
            # sfx: win fanfare
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: lose sound
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                val = self.board[r, c]
                if val != 0 and val == self.board[r, c+1] and val == self.board[r, c+2]:
                    for i in range(3): matches.add((r, c + i))
                    # Check for longer matches
                    for i in range(3, self.GRID_WIDTH - c):
                        if self.board[r, c+i] == val:
                            matches.add((r, c+i))
                        else: break
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                val = self.board[r, c]
                if val != 0 and val == self.board[r+1, c] and val == self.board[r+2, c]:
                    for i in range(3): matches.add((r + i, c))
                    # Check for longer matches
                    for i in range(3, self.GRID_HEIGHT - r):
                        if self.board[r+i, c] == val:
                            matches.add((r+i, c))
                        else: break
        return list(matches)
    
    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_slots = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r, c] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.board[r + empty_slots, c] = self.board[r, c]
                    self.board[r, c] = 0

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

    def _draw_iso_cube(self, surface, x, y, color, highlight_color):
        points_top = [
            (x, y - self.TILE_DEPTH_ISO),
            (x + self.TILE_WIDTH_ISO // 2, y - self.TILE_DEPTH_ISO + self.TILE_HEIGHT_ISO // 2),
            (x, y - self.TILE_DEPTH_ISO + self.TILE_HEIGHT_ISO),
            (x - self.TILE_WIDTH_ISO // 2, y - self.TILE_DEPTH_ISO + self.TILE_HEIGHT_ISO // 2)
        ]
        points_left = [
            (x - self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2 - self.TILE_DEPTH_ISO),
            (x, y + self.TILE_HEIGHT_ISO - self.TILE_DEPTH_ISO),
            (x, y + self.TILE_HEIGHT_ISO),
            (x - self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2)
        ]
        points_right = [
            (x + self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2 - self.TILE_DEPTH_ISO),
            (x, y + self.TILE_HEIGHT_ISO - self.TILE_DEPTH_ISO),
            (x, y + self.TILE_HEIGHT_ISO),
            (x + self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2)
        ]
        
        # Darken side colors
        side_color = tuple(max(0, val - 40) for val in color)
        
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, color)
        pygame.gfxdraw.aapolygon(surface, points_top, highlight_color)

    def _render_game(self):
        # Draw grid tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._grid_to_iso(r, c)
                points = [
                    (x, y),
                    (x + self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2),
                    (x, y + self.TILE_HEIGHT_ISO),
                    (x - self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID_TILE)

        # Draw crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.board[r, c]
                if crystal_type != 0:
                    x, y = self._grid_to_iso(r, c)
                    color = self.CRYSTAL_COLORS[crystal_type - 1]
                    highlight = self.CRYSTAL_HIGHLIGHTS[crystal_type - 1]
                    self._draw_iso_cube(self.screen, x, y, color, highlight)
                    
                    # Add sparkles
                    if self.np_random.random() < 0.001:
                        self.sparkle_particles.append([x, y - self.TILE_DEPTH_ISO // 2, self.np_random.random() * 2 - 1, self.np_random.random() * -1 - 0.5, 5])

        # Update and draw sparkles
        for p in self.sparkle_particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 0.2
            if p[4] <= 0:
                self.sparkle_particles.remove(p)
            else:
                pygame.draw.circle(self.screen, (255, 255, 200), (int(p[0]), int(p[1])), int(p[4]))

        # Draw match effects
        for x, y in self.last_match_effects:
            size = self.TILE_WIDTH_ISO * 0.8
            pygame.draw.circle(self.screen, (255, 255, 100), (x, y), int(size), 4)
            pygame.draw.circle(self.screen, (255, 255, 220), (x, y), int(size * 0.6))

        # Draw cursor and selection highlight
        cursor_r, cursor_c = self.cursor_pos
        x, y = self._grid_to_iso(cursor_r, cursor_c)
        points = [
            (x, y),
            (x + self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2),
            (x, y + self.TILE_HEIGHT_ISO),
            (x - self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 3)
        
        if self.selected_pos:
            sel_r, sel_c = self.selected_pos
            x, y = self._grid_to_iso(sel_r, sel_c)
            points = [
                (x, y),
                (x + self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2),
                (x, y + self.TILE_HEIGHT_ISO),
                (x - self.TILE_WIDTH_ISO // 2, y + self.TILE_HEIGHT_ISO // 2)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_SELECTED, points, 4)
            pygame.draw.line(self.screen, self.COLOR_SELECTED, (x, y), (x, y-10), 2)
            pygame.draw.line(self.screen, self.COLOR_SELECTED, (x-5, y-15), (x+5, y-15), 2)


    def _render_ui(self):
        # UI background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 180, 70), border_radius=5)
        
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 100))
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Map keyboard keys to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_press = 0
        shift_held = 0

        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_press = 1
                    action_taken = True
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    done = False
                
                # Any directional key press is an action
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action_taken = True

        if action_taken:
            action = [movement, space_press, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # Render the observation to the screen
        if 'pygame_screen' not in locals():
            pygame.display.set_caption("Crystal Caverns")
            pygame_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        pygame_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over!")
            # Wait for a moment before allowing a reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            
        env.clock.tick(30) # Limit frame rate for human play

    pygame.quit()