import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→↑↓ to move selected block. Space/Shift to cycle selection."

    # Must be a short, user-facing description of the game:
    game_description = "A block-pushing puzzle. Move all colored blocks onto the target gaps before you run out of moves."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        assert self.GRID_WIDTH <= self.WIDTH and self.GRID_HEIGHT <= self.HEIGHT

        self.MAX_STEPS = 1000  # Episode length limit

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_GAP = (60, 70, 90)
        self.COLOR_GAP_OUTLINE = (80, 95, 120)
        self.COLOR_SELECT_HIGHLIGHT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 90, 90),   # Red
            (90, 200, 255),  # Blue
            (120, 255, 120),  # Green
            (255, 220, 100),  # Yellow
            (200, 120, 255),  # Purple
            (255, 150, 80),  # Orange
        ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.moves_left = 0
        self.blocks = []
        self.gaps = []
        self.gap_pos_set = set()
        self.selected_block_idx = 0

        # --- RNG (seeded in reset) ---
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        """Generates a solvable level by working backward from a solved state."""
        self.blocks = []
        self.gaps = []

        num_pairs = self.np_random.integers(3, 7)
        scramble_moves = num_pairs * 2 + self.np_random.integers(1, 4)

        self.moves_left = scramble_moves + self.np_random.integers(2, 6)

        all_positions = [(c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)

        initial_positions = all_positions[:num_pairs]

        colors = list(self.BLOCK_COLORS)
        self.np_random.shuffle(colors)

        for i in range(num_pairs):
            pos = initial_positions[i]
            color = colors[i % len(colors)]
            self.blocks.append({'pos': pos, 'color': color})
            self.gaps.append({'pos': pos})

        for _ in range(scramble_moves):
            if not self.blocks: continue
            block_idx = self.np_random.integers(0, len(self.blocks))
            direction_idx = self.np_random.integers(1, 5)

            original_selection = self.selected_block_idx
            self.selected_block_idx = block_idx
            self._move_block(direction_idx, consume_move=False)
            self.selected_block_idx = original_selection

        self.gap_pos_set = {g['pos'] for g in self.gaps}
        if self._check_win_condition():
            # If the scrambled level is already solved, regenerate.
            # Using a new seed to avoid getting stuck in a loop.
            self.reset(seed=self.np_random.integers(0, 1e9))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.selected_block_idx = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _move_block(self, direction, consume_move=True):
        if not self.blocks:
            return False

        block_to_move = self.blocks[self.selected_block_idx]
        original_pos = block_to_move['pos']

        vec = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(direction, (0, 0))
        if vec == (0, 0):
            return False

        current_pos = list(original_pos)
        other_block_positions = {b['pos'] for i, b in enumerate(self.blocks) if i != self.selected_block_idx}

        while True:
            next_pos = (current_pos[0] + vec[0], current_pos[1] + vec[1])
            if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                break
            if next_pos in other_block_positions:
                break
            current_pos[0], current_pos[1] = next_pos

        new_pos = tuple(current_pos)

        if new_pos != original_pos:
            block_to_move['pos'] = new_pos
            if consume_move:
                self.moves_left = max(0, self.moves_left - 1)
            return True
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self._check_termination(), self.steps >= self.MAX_STEPS, self._get_info()

        movement, space_btn, shift_btn = action
        reward = 0
        truncated = False

        gaps_filled_before = {b['pos'] for b in self.blocks if b['pos'] in self.gap_pos_set}

        moved = False
        selection_changed = False

        if space_btn and not shift_btn:
            if self.blocks:
                self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
                selection_changed = True
        elif shift_btn and not space_btn:
            if self.blocks:
                self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
                selection_changed = True

        if movement > 0:
            moved = self._move_block(movement)

        if moved or selection_changed:
            self.steps += 1

        gaps_filled_after = {b['pos'] for b in self.blocks if b['pos'] in self.gap_pos_set}
        newly_filled_count = len(gaps_filled_after - gaps_filled_before)
        if newly_filled_count > 0:
            reward += newly_filled_count
            self.score += newly_filled_count

        terminated = self._check_termination()
        if terminated and not self.win_message:
            if self._check_win_condition():
                reward += 50
                self.score += 50
                self.win_message = "LEVEL COMPLETE!"
            else:
                reward -= 50
                self.score -= 50
                self.win_message = "OUT OF MOVES!"

        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_win_condition(self):
        if not self.gaps: return True
        block_positions = {b['pos'] for b in self.blocks}
        return self.gap_pos_set.issubset(block_positions)

    def _check_termination(self):
        if self.game_over:
            return True

        if self._check_win_condition():
            self.game_over = True
            return True

        if self.moves_left <= 0:
            self.game_over = True
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface_width = self.GRID_COLS * self.CELL_SIZE
        grid_surface_height = self.GRID_ROWS * self.CELL_SIZE

        for x in range(0, grid_surface_width + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, grid_surface_height))
        for y in range(0, grid_surface_height + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (grid_surface_width, y))

        for gap in self.gaps:
            gx, gy = gap['pos']
            rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_GAP, rect)
            pygame.draw.rect(self.screen, self.COLOR_GAP_OUTLINE, rect, 2)

        for i, block in enumerate(self.blocks):
            bx, by = block['pos']
            rect = pygame.Rect(bx * self.CELL_SIZE, by * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)

            if i == self.selected_block_idx and not self.game_over and self.blocks:
                highlight_rect = rect.inflate(8, 8)
                pygame.draw.rect(self.screen, self.COLOR_SELECT_HIGHLIGHT, highlight_rect, 0, 5)

            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, block['color'], inner_rect, 0, 3)
            pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in block['color']), inner_rect, 2, 3)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_left}"
        text_surface = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, self.HEIGHT - 34))

        score_text = f"Score: {self.score}"
        text_surface_score = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_rect = text_surface_score.get_rect(topright=(self.WIDTH - 10, self.HEIGHT - 34))
        self.screen.blit(text_surface_score, score_rect)

        if self.game_over and self.win_message:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_surface = self.font_large.render(self.win_message, True, self.COLOR_SELECT_HIGHLIGHT)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surface, msg_rect)

    def _get_info(self):
        gaps_filled = 0
        if self.blocks:
            gaps_filled = len({b['pos'] for b in self.blocks if b['pos'] in self.gap_pos_set})
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gaps_total": len(self.gaps),
            "gaps_filled": gaps_filled
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")