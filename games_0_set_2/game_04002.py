
# Generated: 2025-08-28T01:05:36.361556
# Source Brief: brief_04002.md
# Brief Index: 4002

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile. "
        "Move to an adjacent tile and press Space again to swap. "
        "Hold Shift and press Space to reshuffle the board (costs points)."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the entire board before the timer runs out to win. Cascading combos earn more points!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 6
    TILE_SIZE = 48
    GRID_MARGIN_X = (SCREEN_WIDTH - (GRID_COLS * TILE_SIZE)) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - (GRID_ROWS * TILE_SIZE)) // 2 + 20
    NUM_TILE_TYPES = 5
    MAX_STEPS = 1800  # 60 seconds at 30fps
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTION = (255, 255, 255)
    
    TILE_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (200, 50, 255),  # Purple
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.grid = None
        self.cursor_pos = None
        self.first_selection = None
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.total_tiles = self.GRID_ROWS * self.GRID_COLS
        self.cleared_tiles = 0
        self.particles = []
        self.last_action = np.array([0, 0, 0])

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.cleared_tiles = 0
        
        self.cursor_pos = [0, 0]
        self.first_selection = None
        self.particles = []
        self.last_action = np.array([0, 0, 0])

        self._create_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.timer -= 1

        movement, space_action, shift_action = action[0], action[1], action[2]
        space_pressed = space_action == 1 and self.last_action[1] == 0
        shift_held = shift_action == 1

        # --- Handle Input ---
        self._move_cursor(movement)

        if shift_held and space_pressed:
            # Reshuffle action
            reward -= 25
            self._create_board()
            self.first_selection = None
            # sfx: reshuffle_sound

        elif space_pressed:
            reward += self._handle_selection()

        # --- Game Logic ---
        if self.game_over:
             # Prevent further actions if game is already over
             pass
        else:
            # Check for win/loss conditions
            if self.cleared_tiles >= self.total_tiles:
                self.game_over = True
                reward += 100 # Win bonus
            elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
                self.game_over = True
                reward -= 100 # Loss penalty
            elif not self._find_possible_moves():
                # Auto-reshuffle if no moves are left, with a penalty
                self._create_board()
                reward -= 10
                self.first_selection = None
                # sfx: no_moves_sound

        self.last_action = action
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _create_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
            while self._find_matches():
                matches = self._find_matches()
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
            
            if self._find_possible_moves():
                break

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
    
    def _handle_selection(self):
        r, c = self.cursor_pos
        if self.first_selection is None:
            self.first_selection = (r, c)
            # sfx: select_gem_sound
            return 0
        else:
            r1, c1 = self.first_selection
            r2, c2 = r, c
            
            # Deselect if same tile is chosen twice
            if (r1, c1) == (r2, c2):
                self.first_selection = None
                return 0

            # Check for adjacency
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                swap_reward = self._attempt_swap((r1, c1), (r2, c2))
                self.first_selection = None
                return swap_reward
            else:
                # If not adjacent, make current tile the new first selection
                self.first_selection = (r2, c2)
                # sfx: invalid_selection_sound
                return 0
    
    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches1 = self._find_matches()
        
        if not matches1:
            # Invalid move, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sfx: invalid_swap_sound
            return -1 # Small penalty for invalid move
        else:
            # Valid move, process cascades
            # sfx: match_success_sound
            return self._process_cascades()

    def _process_cascades(self):
        total_reward = 0
        combo_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_cleared = len(matches)
            reward_for_this_cascade = int(num_cleared * combo_multiplier)
            total_reward += reward_for_this_cascade
            self.score += reward_for_this_cascade
            self.cleared_tiles += num_cleared

            for r, c in matches:
                self._spawn_particles(r, c, self.grid[r, c])
                self.grid[r, c] = -1 # Mark as empty
            
            self._apply_gravity()
            self._fill_top_rows()

            combo_multiplier += 0.5 # Increase combo for next cascade
            # sfx: cascade_sound

        return total_reward

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.add((r, c))
                    matches.add((r, c+1))
                    matches.add((r, c+2))
        # Vertical matches
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.add((r, c))
                    matches.add((r+1, c))
                    matches.add((r+2, c))
        return matches
    
    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Test swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches():
                        moves.append(((r, c), (r, c+1)))
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c] # Swap back
                # Test swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches():
                        moves.append(((r, c), (r+1, c)))
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c] # Swap back
        return moves

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _spawn_particles(self, r, c, tile_type):
        px = self.GRID_MARGIN_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_MARGIN_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
        color = self.TILE_COLORS[tile_type]
        for _ in range(10): # Spawn 10 particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # lifetime -= 1
            alpha = max(0, min(255, int(255 * (p[4] / 30.0))))
            color = p[5]
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 2, (*color, alpha))

        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_MARGIN_X, self.GRID_MARGIN_Y, self.GRID_COLS * self.TILE_SIZE, self.GRID_ROWS * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r, c]
                if tile_type != -1:
                    color = self.TILE_COLORS[tile_type]
                    x = self.GRID_MARGIN_X + c * self.TILE_SIZE
                    y = self.GRID_MARGIN_Y + r * self.TILE_SIZE
                    
                    # Draw gem as a circle with a border
                    center_x = x + self.TILE_SIZE // 2
                    center_y = y + self.TILE_SIZE // 2
                    radius = self.TILE_SIZE // 2 - 4
                    
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    
                    # Highlight effect
                    highlight_color = (min(255, color[0]+60), min(255, color[1]+60), min(255, color[2]+60))
                    pygame.gfxdraw.aacircle(self.screen, center_x - 3, center_y - 3, radius // 3, highlight_color)

        # Draw first selection highlight
        if self.first_selection:
            r, c = self.first_selection
            x = self.GRID_MARGIN_X + c * self.TILE_SIZE
            y = self.GRID_MARGIN_Y + r * self.TILE_SIZE
            rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
            
            # Pulsating effect
            pulse = abs(math.sin(self.steps * 0.2))
            alpha = 100 + 155 * pulse
            s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_SELECTION, alpha), s.get_rect(), border_radius=8)
            self.screen.blit(s, (x, y))

        # Draw cursor
        cur_r, cur_c = self.cursor_pos
        cursor_x = self.GRID_MARGIN_X + cur_c * self.TILE_SIZE
        cursor_y = self.GRID_MARGIN_Y + cur_r * self.TILE_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=8)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_percent = max(0, self.timer / self.MAX_STEPS)
        timer_bar_width = 200
        timer_bar_height = 20
        bar_x = self.SCREEN_WIDTH - timer_bar_width - 10
        bar_y = 15
        
        # Bar color changes from green to red
        bar_color = (
            int(255 * (1 - timer_percent)),
            int(255 * timer_percent),
            0
        )
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, timer_bar_width, timer_bar_height))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, timer_bar_width * timer_percent, timer_bar_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, timer_bar_width, timer_bar_height), 1)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.cleared_tiles >= self.total_tiles else "TIME'S UP!"
            end_text = self.font_large.render(msg, True, (255, 255, 0))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cleared_tiles": self.cleared_tiles,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for display
    pygame.display.set_caption("Match-3 Gym Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = np.array([movement, space, shift])
        
        # Only step if an action is taken
        if not np.array_equal(action, env.last_action) or movement != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Step: {info['steps']}, "
                f"Action: {action}, "
                f"Reward: {reward:.2f}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Score: {info['score']}"
            )
            
            if terminated:
                print("Game Over!")
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate for manual play

    env.close()