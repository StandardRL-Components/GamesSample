
# Generated: 2025-08-27T13:00:37.219604
# Source Brief: brief_00230.md
# Brief Index: 230

        
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
    """
    A pixel-art puzzle game where the player must recreate a target image.

    The player controls a cursor on a 10x10 grid and can select from a
    palette of 5 colors. Each paint action consumes one of the 25 available
    moves. The goal is to perfectly match the target image before running
    out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Shift to cycle colors. "
        "Press Space to paint the selected square."
    )

    game_description = (
        "A strategic puzzle game. Recreate the target pixel art image by painting "
        "a grid, but you only have a limited number of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 25

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

        # --- Visuals ---
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID_LINE = (50, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_EMPTY = (40, 40, 50)
        self.PALETTE = [
            (231, 76, 60),   # Red
            (241, 196, 15),  # Yellow
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (155, 89, 182),  # Purple
        ]
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Game State ---
        self.target_grid = None
        self.player_grid = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.prev_shift_held = False
        self.completed_rows = None
        self.completed_cols = None
        self.game_over = False
        self.last_paint_info = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.target_grid = self.np_random.integers(
            0, len(self.PALETTE), size=(self.GRID_SIZE, self.GRID_SIZE)
        )
        self.player_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)

        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0
        self.prev_shift_held = False
        self.game_over = False
        self.completed_rows = np.zeros(self.GRID_SIZE, dtype=bool)
        self.completed_cols = np.zeros(self.GRID_SIZE, dtype=bool)
        self.last_paint_info = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        self._handle_actions(movement, space_held, shift_held)
        
        # Calculate reward only if a paint action was successful
        if self.last_paint_info and self.last_paint_info['just_painted']:
            x, y = self.last_paint_info['pos']
            reward += self._calculate_paint_reward(x, y)
            self.last_paint_info['just_painted'] = False

        self.score += reward
        terminated = self._check_termination()
        
        final_reward = reward
        if terminated:
            if np.array_equal(self.player_grid, self.target_grid):
                final_reward += 100
                self.score += 100
                # play_sound('win')
            elif self.moves_left <= 0:
                final_reward -= 50
                self.score -= 50
                # play_sound('lose')
        
        self._update_animations()

        return (
            self._get_observation(),
            final_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_actions(self, movement, space_held, shift_held):
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Handle color selection (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PALETTE)
            # play_sound('color_cycle')
        self.prev_shift_held = shift_held

        # 3. Handle painting
        if space_held and self.moves_left > 0:
            x, y = self.cursor_pos
            if self.player_grid[y, x] != self.selected_color_idx:
                self.player_grid[y, x] = self.selected_color_idx
                self.moves_left -= 1
                # play_sound('paint')
                self.last_paint_info = {'pos': (x, y), 'timer': 5, 'just_painted': True}

    def _calculate_paint_reward(self, x, y):
        reward = 0
        if self.player_grid[y, x] == self.target_grid[y, x]:
            reward += 1

        if not self.completed_rows[y] and np.array_equal(self.player_grid[y, :], self.target_grid[y, :]):
            reward += 10
            self.completed_rows[y] = True
            # play_sound('row_complete')

        if not self.completed_cols[x] and np.array_equal(self.player_grid[:, x], self.target_grid[:, x]):
            reward += 10
            self.completed_cols[x] = True
            # play_sound('col_complete')
        
        return reward
        
    def _check_termination(self):
        perfect_match = np.array_equal(self.player_grid, self.target_grid)
        out_of_moves = self.moves_left <= 0
        if perfect_match or out_of_moves:
            self.game_over = True
            return True
        return False

    def _update_animations(self):
        if self.last_paint_info and self.last_paint_info['timer'] > 0:
            self.last_paint_info['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        TARGET_GRID_POS = (30, 80)
        TARGET_CELL_SIZE = 12
        PLAYER_GRID_POS = (170, 50)
        PLAYER_CELL_SIZE = 30
        PALETTE_POS = (510, 50)
        PALETTE_SWATCH_SIZE = 40

        self._draw_grid(self.target_grid, TARGET_GRID_POS, TARGET_CELL_SIZE, "Target")
        self._draw_grid(self.player_grid, PLAYER_GRID_POS, PLAYER_CELL_SIZE, "Canvas")
        self._render_cursor(PLAYER_GRID_POS, PLAYER_CELL_SIZE)
        self._render_paint_animation(PLAYER_GRID_POS, PLAYER_CELL_SIZE)
        self._render_palette(PALETTE_POS, PALETTE_SWATCH_SIZE)

    def _draw_grid(self, grid_data, pos, cell_size, label):
        label_surf = self.font_s.render(label, True, self.COLOR_TEXT)
        self.screen.blit(label_surf, (pos[0], pos[1] - 25))
        
        grid_width = self.GRID_SIZE * cell_size
        grid_height = self.GRID_SIZE * cell_size
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = grid_data[y, x]
                color = self.PALETTE[color_idx] if color_idx != -1 else self.COLOR_EMPTY
                rect = (pos[0] + x * cell_size, pos[1] + y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)
        
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (pos[0] + i * cell_size, pos[1]), (pos[0] + i * cell_size, pos[1] + grid_height))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (pos[0], pos[1] + i * cell_size), (pos[0] + grid_width, pos[1] + i * cell_size))

    def _render_cursor(self, grid_pos, cell_size):
        cursor_x = grid_pos[0] + self.cursor_pos[0] * cell_size
        cursor_y = grid_pos[1] + self.cursor_pos[1] * cell_size
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, cell_size, cell_size), 3)

    def _render_paint_animation(self, grid_pos, cell_size):
        if self.last_paint_info and self.last_paint_info['timer'] > 0:
            anim_x, anim_y = self.last_paint_info['pos']
            timer = self.last_paint_info['timer']
            offset = (5 - timer)
            size_mod = offset * 2
            px = grid_pos[0] + anim_x * cell_size + offset
            py = grid_pos[1] + anim_y * cell_size + offset
            ps = cell_size - size_mod
            color_idx = self.player_grid[anim_y, anim_x]
            if color_idx != -1:
                color = self.PALETTE[color_idx]
                pygame.draw.rect(self.screen, color, (px, py, max(0, ps), max(0, ps)))

    def _render_palette(self, pos, swatch_size):
        label_surf = self.font_s.render("Palette", True, self.COLOR_TEXT)
        self.screen.blit(label_surf, (pos[0], pos[1] - 25))
        for i, color in enumerate(self.PALETTE):
            swatch_y = pos[1] + i * (swatch_size + 10)
            pygame.draw.rect(self.screen, color, (pos[0], swatch_y, swatch_size, swatch_size))
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (pos[0] - 3, swatch_y - 3, swatch_size + 6, swatch_size + 6), 3)

    def _render_ui(self):
        score_text = self.font_m.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_m.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("Perfect Match!", (100, 255, 100)) if np.array_equal(self.player_grid, self.target_grid) else ("Out of Moves!", (255, 100, 100))
            
            end_text = self.font_l.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "correct_pixels": int(np.sum(self.player_grid == self.target_grid)),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0]  # Default no-op action
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # Keyboard input to action mapping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display_surf.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()