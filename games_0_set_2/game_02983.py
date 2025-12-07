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
        "Controls: Use arrows to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. "
        "Hold Shift to see a hint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent gems to create lines of three "
        "or more. Plan your moves carefully to create chain reactions and reach the "
        "target score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 8
    GRID_COLS = 8
    NUM_GEM_TYPES = 6
    CELL_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2

    WIN_SCORE = 100
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 50)
    COLOR_GRID = (40, 60, 90)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    COLOR_HINT = (0, 255, 128)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]
    TEXT_COLOR = (240, 240, 240)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.prev_space_held = False
        self.particles = []
        self.hint_pos = None

        # self.validate_implementation() # Removed from here to avoid error before reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_message = ""
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        self.prev_space_held = False
        self.particles = []
        self.hint_pos = None

        self._create_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False

        # --- Handle hint ---
        self.hint_pos = self._find_possible_move() if shift_held else None

        # --- Handle cursor movement ---
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Handle selection and swapping on space PRESS ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and not self.game_over:
            x, y = self.cursor_pos
            if self.selected_pos is None:
                # Select a gem
                self.selected_pos = [x, y]
            else:
                # Attempt to swap
                sx, sy = self.selected_pos
                is_adjacent = abs(x - sx) + abs(y - sy) == 1
                if is_adjacent:
                    swap_reward = self._handle_swap((x, y), (sx, sy))
                    if swap_reward > -0.1: # Successful swap
                        self.moves_left -= 1
                        self.score += swap_reward
                        reward += swap_reward
                    else: # Invalid swap
                        reward += swap_reward

                    self.selected_pos = None
                elif (x, y) == (sx, sy):
                    # Deselect
                    self.selected_pos = None
                else:
                    # Select a new gem
                    self.selected_pos = [x, y]

        self.prev_space_held = space_held

        # --- Update game state and check for termination ---
        self._update_particles()
        self.steps += 1

        if not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100  # Win bonus
                self.win_message = "YOU WIN!"
            elif self.moves_left <= 0:
                self.game_over = True
                terminated = True
                reward -= 10  # Loss penalty
                self.win_message = "GAME OVER"

        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium v26+, truncated means terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_initial_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_COLS, self.GRID_ROWS))
        while True:
            # Remove initial matches
            matches = self._find_all_matches()
            if not matches:
                # Check for possible moves
                if self._find_possible_move():
                    break  # Grid is valid
                else:  # No moves, reshuffle
                    self._shuffle_grid()
                    continue

            # If matches exist, replace them and repeat
            for x, y in matches:
                self.grid[x, y] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _shuffle_grid(self):
        flat_grid = self.grid.flatten()
        self.np_random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_COLS, self.GRID_ROWS))

    def _handle_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        # Perform swap
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]

        total_score_from_swap = 0
        chain_level = 0

        while True:
            matches = self._find_all_matches()
            if not matches:
                break

            # Add score
            total_score_from_swap += len(matches)  # +1 per gem
            if chain_level > 0:
                total_score_from_swap += 5  # Chain reaction bonus

            # Process matches: remove gems, apply gravity, fill top
            self._process_matches(matches)
            chain_level += 1

        if chain_level == 0:  # No match was made from the initial swap
            # Swap back
            self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
            return -0.1  # Penalty for invalid move

        # Anti-softlock: if board has no more moves, reshuffle
        if not self._find_possible_move():
            self._create_initial_grid()

        return total_score_from_swap

    def _find_all_matches(self):
        matches = set()
        # Check horizontal
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - 2):
                if self.grid[x, y] == self.grid[x + 1, y] == self.grid[x + 2, y] and self.grid[x, y] != -1:
                    matches.update([(x, y), (x + 1, y), (x + 2, y)])
        # Check vertical
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - 2):
                if self.grid[x, y] == self.grid[x, y + 1] == self.grid[x, y + 2] and self.grid[x, y] != -1:
                    matches.update([(x, y), (x, y + 1), (x, y + 2)])
        return list(matches)

    def _process_matches(self, matches):
        # Create particles and mark gems for removal
        for x, y in matches:
            if self.grid[x, y] != -1:  # Avoid double-processing
                self._create_particles(x, y, self.GEM_COLORS[self.grid[x, y]])
                self.grid[x, y] = -1  # Mark as empty

        # Apply gravity
        for x in range(self.GRID_COLS):
            empty_slots = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = -1

        # Fill top rows with new gems
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] == -1:
                    self.grid[x, y] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_possible_move(self):
        # Check for horizontal swaps
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS - 1):
                # Swap and check
                self.grid[x, y], self.grid[x + 1, y] = self.grid[x + 1, y], self.grid[x, y]
                if self._check_match_at(x, y) or self._check_match_at(x + 1, y):
                    self.grid[x, y], self.grid[x + 1, y] = self.grid[x + 1, y], self.grid[x, y]  # Swap back
                    return [(x, y), (x + 1, y)]
                self.grid[x, y], self.grid[x + 1, y] = self.grid[x + 1, y], self.grid[x, y]  # Swap back

        # Check for vertical swaps
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS - 1):
                self.grid[x, y], self.grid[x, y + 1] = self.grid[x, y + 1], self.grid[x, y]
                if self._check_match_at(x, y) or self._check_match_at(x, y + 1):
                    self.grid[x, y], self.grid[x, y + 1] = self.grid[x, y + 1], self.grid[x, y]
                    return [(x, y), (x, y + 1)]
                self.grid[x, y], self.grid[x, y + 1] = self.grid[x, y + 1], self.grid[x, y]
        return None

    def _check_match_at(self, x, y):
        gem_type = self.grid[x, y]
        # Check horizontal
        h_count = 1
        for i in range(1, 3):
            if x - i >= 0 and self.grid[x - i, y] == gem_type:
                h_count += 1
            else:
                break
        for i in range(1, 3):
            if x + i < self.GRID_COLS and self.grid[x + i, y] == gem_type:
                h_count += 1
            else:
                break
        if h_count >= 3: return True

        # Check vertical
        v_count = 1
        for i in range(1, 3):
            if y - i >= 0 and self.grid[x, y - i] == gem_type:
                v_count += 1
            else:
                break
        for i in range(1, 3):
            if y + i < self.GRID_ROWS and self.grid[x, y + i] == gem_type:
                v_count += 1
            else:
                break
        if v_count >= 3: return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_COLS * self.CELL_SIZE,
                                self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        # Draw gems
        if self.grid is not None:
            for y in range(self.GRID_ROWS):
                for x in range(self.GRID_COLS):
                    gem_type = self.grid[x, y]
                    if gem_type != -1:
                        self._render_gem(x, y, self.GEM_COLORS[gem_type])

        # Draw hint highlight
        if self.hint_pos:
            for x, y in self.hint_pos:
                rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                                    self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_HINT, rect, 3)

        # Draw selection highlight
        if self.selected_pos is not None:
            sx, sy = self.selected_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + sx * self.CELL_SIZE, self.GRID_OFFSET_Y + sy * self.CELL_SIZE,
                                self.CELL_SIZE, self.CELL_SIZE)

            # Pulsing effect for selection
            pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
            width = 2 + int(pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, width, border_radius=5)

        # Draw cursor
        if self.cursor_pos is not None:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE,
                                       self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_gem(self, x, y, color):
        center_x = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 2 - 5

        # Gem body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Highlight for 3D effect
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 2, 135, 315, highlight_color)

        # Shadow for 3D effect
        shadow_color = tuple(max(0, c - 60) for c in color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 1, 315, 135, shadow_color)

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.TEXT_COLOR)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(self.win_message, True, self.TEXT_COLOR)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'lifespan': lifespan,
                'color': color,
                'radius': radius
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] > 0 and p['radius'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left
        }

    def validate_implementation(self):
        """
        This method is not part of the standard Gymnasium API, but is used to
        check that the environment is implemented correctly.
        It is called automatically when the environment is created.
        """
        # Call reset() first to initialize the state, including the grid
        obs, info = self.reset()

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space (using the observation from reset)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8

        # Test reset output
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    # Set the video driver back to something that works
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    # The validation is not part of the standard API, but we can call it here for testing
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")

    obs, info = env.reset()
    terminated = False
    truncated = False

    # --- Manual control mapping ---
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated and not truncated:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break  # only one movement at a time

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        # In manual play, we only step on an actual input to feel turn-based
        if movement != 0 or space_held != env.prev_space_held:
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            truncated = trunc
            if reward != 0:
                print(f"Step: {info['steps']}, Moves: {info['moves_left']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # For manual play, we need a small delay to register single key presses
        clock.tick(15)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()