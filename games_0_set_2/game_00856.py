
# Generated: 2025-08-27T15:00:08.431289
# Source Brief: brief_00856.md
# Brief Index: 856

        
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
    An isometric puzzle game where the player pushes colored crystals
    to create matches of 3 or more. The goal is to clear the entire
    board within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to push the selected "
        "crystal in the last direction you moved. Hold Shift for a board shuffle."
    )

    # User-facing description of the game
    game_description = (
        "A strategic puzzle game. Clear the board by pushing crystals to form "
        "lines of 3 or more of the same color. Plan your moves carefully!"
    )

    # The game state is static until a user submits an action.
    auto_advance = False

    # --- Constants ---
    # Board and Crystal Config
    BOARD_SIZE = 8
    NUM_CRYSTAL_TYPES = 3  # 1: Red, 2: Green, 3: Blue
    TURN_LIMIT = 250

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 65)
    COLOR_CURSOR = (255, 220, 0)
    CRYSTAL_COLORS = {
        1: (227, 52, 47),   # Red
        2: (46, 204, 64),   # Green
        3: (0, 116, 217),   # Blue
    }
    CRYSTAL_GLOW_COLORS = {
        1: (113, 26, 23),
        2: (23, 102, 32),
        3: (0, 58, 108),
    }

    # Isometric Rendering Config
    TILE_WIDTH = 64
    TILE_HEIGHT = 32
    CUBE_HEIGHT = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Rendering offsets to center the board
        self.board_offset_x = (640 - self.BOARD_SIZE * self.TILE_WIDTH / 2) / 2 + 120
        self.board_offset_y = (400 - self.BOARD_SIZE * self.TILE_HEIGHT / 2) / 2 - 40

        # Initialize state variables
        self.board = None
        self.cursor_pos = None
        self.last_move_direction = None
        self.score = 0
        self.turns_left = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.rng = None
        self.steps = 0
        
        # Initial call to reset to set up the first game state
        self.reset()
        
        # Self-validation
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self._create_board()
        self.cursor_pos = np.array([self.BOARD_SIZE // 2, self.BOARD_SIZE // 2])
        self.last_move_direction = 1 # Default to UP
        
        self.steps = 0
        self.score = 0
        self.turns_left = self.TURN_LIMIT
        self.game_over = False
        self.win = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        action_taken = True
        
        # --- Action Handling ---
        # Priority: Shuffle > Push > Move > No-op
        if shift_held:
            # Shuffle action (high cost)
            self._shuffle_board()
            reward -= 10
            # sfx: board_shuffle_sound
        elif space_held:
            # Push action
            reward += self._handle_push()
        elif movement > 0:
            # Cursor movement
            self._move_cursor(movement)
            reward -= 0.01 # Small cost for thinking/moving
            action_taken = False # Moving cursor doesn't use a "turn"
        else: # No-op
            action_taken = False

        if action_taken:
            self.turns_left -= 1
            self.steps += 1
            
        # --- Post-Action State Update ---
        total_crystals = np.sum(self.board > 0)
        if total_crystals == 0:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50  # Big win bonus
            # sfx: win_fanfare_sound
        elif self.turns_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 50  # Big loss penalty
            # sfx: loss_buzzer_sound
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_board(self):
        """Generates a new board, ensuring no initial matches exist."""
        self.board = self.rng.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.BOARD_SIZE, self.BOARD_SIZE))
        # Keep resolving matches until the board is stable
        while self._find_and_clear_matches(initial_setup=True) > 0:
            self._apply_gravity()
            # Fill empty spaces created at the top
            for c in range(self.BOARD_SIZE):
                for r in range(self.BOARD_SIZE):
                    if self.board[r, c] == 0:
                        self.board[r, c] = self.rng.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _shuffle_board(self):
        """Randomly rearranges all crystals on the board."""
        flat_board = self.board.flatten()
        self.rng.shuffle(flat_board)
        self.board = flat_board.reshape((self.BOARD_SIZE, self.BOARD_SIZE))
        # Ensure the shuffle doesn't create instant matches
        while self._find_and_clear_matches(initial_setup=True) > 0:
            self._apply_gravity()
            for c in range(self.BOARD_SIZE):
                for r in range(self.BOARD_SIZE):
                    if self.board[r, c] == 0:
                        self.board[r, c] = self.rng.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _move_cursor(self, movement):
        """Updates cursor position and stores the last move direction."""
        dr, dc = 0, 0
        if movement == 1: dr = -1  # Up
        elif movement == 2: dr = 1   # Down
        elif movement == 3: dc = -1  # Left
        elif movement == 4: dc = 1   # Right
        
        self.last_move_direction = movement
        
        new_r = np.clip(self.cursor_pos[0] + dr, 0, self.BOARD_SIZE - 1)
        new_c = np.clip(self.cursor_pos[1] + dc, 0, self.BOARD_SIZE - 1)
        self.cursor_pos = np.array([new_r, new_c])

    def _handle_push(self):
        """Executes the crystal push logic and subsequent matching."""
        if self.last_move_direction is None:
            return -0.1 # Penalty for pushing with no direction
        
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[self.last_move_direction]
        r, c = self.cursor_pos
        
        if self.board[r, c] == 0:
            return -0.1 # Pushing an empty space

        if self._push_crystal_line(r, c, dr, dc):
            # sfx: crystal_push_sound
            cleared_count = self._find_and_clear_matches()
            if cleared_count > 0:
                self._apply_gravity()
                # sfx: match_success_sound
                # Reward for clearing crystals
                reward = cleared_count
                if cleared_count >= 4:
                    reward += 5 # Combo bonus
                self.score += reward
                return reward
            else:
                return -1 # Penalty for a non-productive move
        else:
            # sfx: push_fail_sound
            return -0.5 # Penalty for an illegal move

    def _push_crystal_line(self, r_start, c_start, dr, dc):
        """Pushes a line of crystals, checking for validity."""
        line_to_push = []
        r, c = r_start, c_start
        
        # Find all contiguous crystals in the push direction
        while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r, c] != 0:
            line_to_push.append((r, c))
            r, c = r + dr, c + dc
            
        # Check if the push is valid (ends in an empty space on the board)
        if not (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE):
            return False # Pushing out of bounds

        # Move the crystals starting from the end of the line
        for r_p, c_p in reversed(line_to_push):
            self.board[r_p + dr, c_p + dc] = self.board[r_p, c_p]
        
        # Clear the starting position
        self.board[r_start, c_start] = 0
        return True

    def _find_and_clear_matches(self, initial_setup=False):
        """Finds and clears all matching lines of 3+ crystals."""
        to_clear = set()
        
        # Check rows
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE - 2):
                val = self.board[r, c]
                if val != 0 and val == self.board[r, c+1] == self.board[r, c+2]:
                    to_clear.update([(r, c), (r, c+1), (r, c+2)])

        # Check columns
        for c in range(self.BOARD_SIZE):
            for r in range(self.BOARD_SIZE - 2):
                val = self.board[r, c]
                if val != 0 and val == self.board[r+1, c] == self.board[r+2, c]:
                    to_clear.update([(r, c), (r+1, c), (r+2, c)])
        
        if not to_clear:
            return 0
            
        for r, c in to_clear:
            crystal_type = self.board[r, c]
            if crystal_type > 0 and not initial_setup:
                # sfx: crystal_break_sound
                self._spawn_particles(r, c, self.CRYSTAL_COLORS[crystal_type])
            self.board[r, c] = 0
        
        return len(to_clear)

    def _apply_gravity(self):
        """Makes crystals fall down into empty spaces."""
        for c in range(self.BOARD_SIZE):
            empty_row = self.BOARD_SIZE - 1
            for r in range(self.BOARD_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1

    def _spawn_particles(self, r, c, color):
        """Creates a burst of particles at a specified grid location."""
        screen_pos = self._map_to_iso(r, c)
        for _ in range(15):
            vel = [self.rng.uniform(-2, 2), self.rng.uniform(-2, 2) - 1]
            lifetime = self.rng.integers(15, 30)
            size = self.rng.integers(2, 5)
            self.particles.append([list(screen_pos), vel, color, lifetime, size])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "turns_left": self.turns_left,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "crystals_left": int(np.sum(self.board > 0)),
        }

    def _map_to_iso(self, r, c):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.board_offset_x + (c - r) * (self.TILE_WIDTH / 2)
        screen_y = self.board_offset_y + (c + r) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_iso_cube(self, surface, r, c, color, glow_color):
        """Renders a single isometric crystal with a glow effect."""
        x, y = self._map_to_iso(r, c)
        
        # Points for the cube faces
        top_face = [
            (x, y - self.CUBE_HEIGHT),
            (x + self.TILE_WIDTH // 2, y - self.CUBE_HEIGHT + self.TILE_HEIGHT // 2),
            (x, y - self.CUBE_HEIGHT + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH // 2, y - self.CUBE_HEIGHT + self.TILE_HEIGHT // 2),
        ]
        left_face = [
            (x - self.TILE_WIDTH // 2, y - self.CUBE_HEIGHT + self.TILE_HEIGHT // 2),
            (x, y - self.CUBE_HEIGHT + self.TILE_HEIGHT),
            (x, y + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2),
        ]
        right_face = [
            (x + self.TILE_WIDTH // 2, y - self.CUBE_HEIGHT + self.TILE_HEIGHT // 2),
            (x, y - self.CUBE_HEIGHT + self.TILE_HEIGHT),
            (x, y + self.TILE_HEIGHT),
            (x + self.TILE_WIDTH // 2, y + self.TILE_HEIGHT // 2),
        ]
        
        # Render glow by drawing larger, semi-transparent shapes behind
        glow_surface = pygame.Surface((self.TILE_WIDTH*2, self.TILE_HEIGHT*2 + self.CUBE_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surface, (*glow_color, 100), [(p[0]-x+self.TILE_WIDTH, p[1]-y+self.CUBE_HEIGHT) for p in top_face])
        pygame.draw.polygon(glow_surface, (*glow_color, 100), [(p[0]-x+self.TILE_WIDTH, p[1]-y+self.CUBE_HEIGHT) for p in left_face])
        pygame.draw.polygon(glow_surface, (*glow_color, 100), [(p[0]-x+self.TILE_WIDTH, p[1]-y+self.CUBE_HEIGHT) for p in right_face])
        
        blur_surface = pygame.transform.smoothscale(glow_surface, (glow_surface.get_width()//2, glow_surface.get_height()//2))
        blur_surface = pygame.transform.smoothscale(blur_surface, glow_surface.get_size())
        surface.blit(blur_surface, (x - self.TILE_WIDTH, y - self.CUBE_HEIGHT), special_flags=pygame.BLEND_RGBA_ADD)

        # Render main cube faces
        pygame.gfxdraw.filled_polygon(surface, top_face, color)
        pygame.gfxdraw.aapolygon(surface, top_face, color)
        
        left_color = tuple(max(0, val - 40) for val in color)
        pygame.gfxdraw.filled_polygon(surface, left_face, left_color)
        pygame.gfxdraw.aapolygon(surface, left_face, left_color)
        
        right_color = tuple(max(0, val - 20) for val in color)
        pygame.gfxdraw.filled_polygon(surface, right_face, right_color)
        pygame.gfxdraw.aapolygon(surface, right_face, right_color)

    def _render_game(self):
        """Main rendering loop for all game elements."""
        # Render grid
        for r in range(self.BOARD_SIZE + 1):
            start = self._map_to_iso(r, 0)
            end = self._map_to_iso(r, self.BOARD_SIZE)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.BOARD_SIZE + 1):
            start = self._map_to_iso(0, c)
            end = self._map_to_iso(self.BOARD_SIZE, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Render crystals
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                crystal_type = self.board[r, c]
                if crystal_type != 0:
                    self._render_iso_cube(self.screen, r, c, self.CRYSTAL_COLORS[crystal_type], self.CRYSTAL_GLOW_COLORS[crystal_type])
        
        # Render cursor
        cur_r, cur_c = self.cursor_pos
        x, y = self._map_to_iso(cur_r, cur_c)
        cursor_points = [
            (x, y), (x + self.TILE_WIDTH / 2, y + self.TILE_HEIGHT / 2),
            (x, y + self.TILE_HEIGHT), (x - self.TILE_WIDTH / 2, y + self.TILE_HEIGHT / 2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 3)

        # Render particles
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[3] -= 1
            if p[3] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[3] / 20))))
                pygame.draw.circle(self.screen, (*p[2], alpha), p[0], p[4])

        # Render Game Over/Win text
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 150, 150)
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(320, 200))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        """Renders the score, timer, and other UI text."""
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, (240, 240, 240))
        self.screen.blit(score_text, (20, 20))
        
        # Turns Left
        turns_text = self.font_small.render(f"TURNS LEFT: {self.turns_left}", True, (240, 240, 240))
        turns_rect = turns_text.get_rect(topright=(620, 20))
        self.screen.blit(turns_text, turns_rect)
        
        # Crystals Left
        crystals_left = int(np.sum(self.board > 0))
        crystals_text = self.font_small.render(f"CRYSTALS: {crystals_left}", True, (240, 240, 240))
        self.screen.blit(crystals_text, (20, 45))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        # Since auto_advance is False, we control the "frame rate" of actions
        clock.tick(15) # Limit player input speed

    pygame.quit()