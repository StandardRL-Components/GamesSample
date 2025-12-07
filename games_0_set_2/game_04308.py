
# Generated: 2025-08-28T02:00:49.047897
# Source Brief: brief_04308.md
# Brief Index: 4308

        
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
        "Controls: Use arrow keys to swap the selected block. Use Space/Shift to change which block is selected."
    )

    game_description = (
        "Recreate the target image by swapping pixel blocks. Each swap costs one move. Plan your moves carefully to solve the puzzle before you run out!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_DIM_W, self.GRID_DIM_H = 4, 4
        self.NUM_BLOCKS = self.GRID_DIM_W * self.GRID_DIM_H
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (30, 35, 50)
        self.COLOR_GRID_LINE = (50, 55, 70)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TEXT_DIM = (150, 150, 160)
        self.COLOR_CORRECT = (60, 220, 120)
        self.COLOR_INCORRECT = (220, 80, 80)
        self.COLOR_SELECT = (80, 150, 255)
        self.COLOR_OVERLAY = (10, 10, 20, 200)

        self.PALETTE = [
            (230, 60, 60), (60, 230, 60), (60, 60, 230), (230, 230, 60),
            (230, 60, 230), (60, 230, 230), (230, 150, 60), (60, 230, 150),
            (150, 60, 230), (200, 200, 200)
        ]

        # Layout
        self.GRID_BLOCK_SIZE = 64
        self.GRID_MARGIN = 4
        self.GRID_BORDER = 2
        self.TOTAL_BLOCK_SIZE = self.GRID_BLOCK_SIZE + self.GRID_MARGIN
        
        grid_pixel_w = self.GRID_DIM_W * self.TOTAL_BLOCK_SIZE
        grid_pixel_h = self.GRID_DIM_H * self.TOTAL_BLOCK_SIZE
        self.main_grid_rect = pygame.Rect(
            (self.SCREEN_WIDTH - grid_pixel_w) // 2 + 50,
            (self.SCREEN_HEIGHT - grid_pixel_h) // 2,
            grid_pixel_w,
            grid_pixel_h
        )

        target_pixel_w = self.GRID_DIM_W * (self.GRID_BLOCK_SIZE // 2 + 2)
        target_pixel_h = self.GRID_DIM_H * (self.GRID_BLOCK_SIZE // 2 + 2)
        self.target_grid_rect = pygame.Rect(
            40, 100, target_pixel_w, target_pixel_h
        )
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_sml = pygame.font.Font(None, 24)
        self.font_med = pygame.font.Font(None, 36)
        self.font_lrg = pygame.font.Font(None, 48)
        
        # --- Game State ---
        self.score = 0  # Persistent across episodes for difficulty scaling
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.moves_remaining = 0
        self.target_config = None
        self.current_config = None
        self.selected_idx = 0
        self.correct_rows = set()
        self.correct_cols = set()
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        level = 1 + self.score // 5
        num_colors = min(len(self.PALETTE), 3 + level)
        
        colors_to_use = self.np_random.choice(np.arange(len(self.PALETTE)), size=num_colors, replace=False)
        
        self.target_config = self.np_random.choice(colors_to_use, size=self.NUM_BLOCKS)
        self.current_config = self.target_config.copy()
        
        # Ensure the starting state is not the winning state
        while np.array_equal(self.current_config, self.target_config):
            self.np_random.shuffle(self.current_config)

        self.moves_remaining = 20 + level * 5
        self.selected_idx = 0
        
        self.correct_rows.clear()
        self.correct_cols.clear()
        self._update_correct_sets(initial=True)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = True
        
        # --- Action Handling ---
        if space_held:
            # Sound: select_tink.wav
            self.selected_idx = (self.selected_idx + 1) % self.NUM_BLOCKS
        elif shift_held:
            # Sound: select_tink.wav
            self.selected_idx = (self.selected_idx - 1 + self.NUM_BLOCKS) % self.NUM_BLOCKS
        elif movement != 0:
            # Sound: block_swap.wav
            reward += self._handle_move(movement)
        else: # No-op
            action_taken = False
        
        if action_taken:
            self.moves_remaining = max(0, self.moves_remaining - 1)
        
        self.steps += 1

        # --- Termination Check ---
        terminated = False
        win = np.array_equal(self.current_config, self.target_config)
        
        if win:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "LEVEL COMPLETE!"
            self.score += 1
            # Sound: win_jingle.wav
        elif self.moves_remaining <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            # Sound: lose_buzzer.wav
        elif self.steps >= 1000:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_move(self, movement):
        y, x = self.selected_idx // self.GRID_DIM_W, self.selected_idx % self.GRID_DIM_W
        ny, nx = y, x

        if movement == 1: ny = (y - 1 + self.GRID_DIM_H) % self.GRID_DIM_H  # Up
        elif movement == 2: ny = (y + 1) % self.GRID_DIM_H  # Down
        elif movement == 3: nx = (x - 1 + self.GRID_DIM_W) % self.GRID_DIM_W  # Left
        elif movement == 4: nx = (x + 1) % self.GRID_DIM_W  # Right
        
        neighbor_idx = ny * self.GRID_DIM_W + nx

        # --- Continuous Reward Calculation ---
        reward = 0
        was_selected_correct = self.current_config[self.selected_idx] == self.target_config[self.selected_idx]
        was_neighbor_correct = self.current_config[neighbor_idx] == self.target_config[neighbor_idx]

        # Perform swap
        self.current_config[self.selected_idx], self.current_config[neighbor_idx] = \
            self.current_config[neighbor_idx], self.current_config[self.selected_idx]

        is_selected_correct = self.current_config[self.selected_idx] == self.target_config[self.selected_idx]
        is_neighbor_correct = self.current_config[neighbor_idx] == self.target_config[neighbor_idx]

        if is_selected_correct and not was_selected_correct: reward += 1.0
        if not is_selected_correct and was_selected_correct: reward -= 0.1
        if is_neighbor_correct and not was_neighbor_correct: reward += 1.0
        if not is_neighbor_correct and was_neighbor_correct: reward -= 0.1
        
        # --- Event-based Reward Calculation ---
        reward += self._update_correct_sets()

        return reward
        
    def _is_row_correct(self, row_idx):
        start = row_idx * self.GRID_DIM_W
        end = start + self.GRID_DIM_W
        return np.array_equal(self.current_config[start:end], self.target_config[start:end])

    def _is_col_correct(self, col_idx):
        return np.array_equal(self.current_config[col_idx::self.GRID_DIM_W], self.target_config[col_idx::self.GRID_DIM_W])

    def _update_correct_sets(self, initial=False):
        reward = 0
        for i in range(self.GRID_DIM_H):
            is_correct = self._is_row_correct(i)
            if is_correct and i not in self.correct_rows:
                if not initial: reward += 5
                self.correct_rows.add(i)
            elif not is_correct and i in self.correct_rows:
                self.correct_rows.remove(i)
        
        for i in range(self.GRID_DIM_W):
            is_correct = self._is_col_correct(i)
            if is_correct and i not in self.correct_cols:
                if not initial: reward += 5
                self.correct_cols.add(i)
            elif not is_correct and i in self.correct_cols:
                self.correct_cols.remove(i)
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Target Grid
        self._render_grid(self.target_grid_rect, self.target_config, is_target=True)
        
        # Render Main Grid
        self._render_grid(self.main_grid_rect, self.current_config, is_target=False)

    def _render_grid(self, grid_rect, config, is_target):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        if is_target:
            block_size = self.GRID_BLOCK_SIZE // 2
            margin = 2
            border = 1
        else:
            block_size = self.GRID_BLOCK_SIZE
            margin = self.GRID_MARGIN
            border = self.GRID_BORDER

        total_block_size = block_size + margin

        for i in range(self.NUM_BLOCKS):
            y, x = i // self.GRID_DIM_W, i % self.GRID_DIM_W
            
            block_rect = pygame.Rect(
                grid_rect.left + x * total_block_size + margin // 2,
                grid_rect.top + y * total_block_size + margin // 2,
                block_size,
                block_size
            )
            
            color_idx = config[i]
            block_color = self.PALETTE[color_idx]
            
            pygame.gfxdraw.box(self.screen, block_rect, block_color)

            if not is_target:
                is_correct = self.current_config[i] == self.target_config[i]
                border_color = self.COLOR_CORRECT if is_correct else self.COLOR_INCORRECT
                
                for b in range(border):
                    pygame.gfxdraw.rectangle(self.screen, block_rect.inflate(b*2, b*2), border_color)

                # Pulsating selection highlight
                if i == self.selected_idx:
                    pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
                    alpha = 100 + int(pulse * 155)
                    size_pulse = int(pulse * 4)
                    
                    highlight_rect = block_rect.inflate(border*2 + size_pulse, border*2 + size_pulse)
                    
                    # Create a temporary surface for alpha blending
                    s = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
                    color_with_alpha = self.COLOR_SELECT[:3] + (alpha,)
                    pygame.gfxdraw.rectangle(s, s.get_rect(), color_with_alpha)
                    self.screen.blit(s, highlight_rect.topleft)

    def _render_ui(self):
        # --- Left Side (Target) ---
        target_text = self.font_med.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.target_grid_rect.centerx - target_text.get_width() // 2, self.target_grid_rect.top - 40))

        # --- Right Side (Stats) ---
        stats_x = self.main_grid_rect.right + 40
        
        score_label = self.font_med.render("LEVEL", True, self.COLOR_TEXT_DIM)
        score_value = self.font_lrg.render(f"{1 + self.score // 5}", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (stats_x, 100))
        self.screen.blit(score_value, (stats_x, 130))

        moves_label = self.font_med.render("MOVES", True, self.COLOR_TEXT_DIM)
        moves_value = self.font_lrg.render(f"{self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_label, (stats_x, 200))
        self.screen.blit(moves_value, (stats_x, 230))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_lrg.render(self.win_message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "level": 1 + self.score // 5,
        }

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    running = True
    while running:
        # --- Action mapping for human play ---
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        
        # --- Step the environment ---
        # For human play, we only step if an action is taken or on a timer
        # Since auto_advance is False, we must provide an action to see updates
        # We'll use a simple cooldown to prevent ultra-fast inputs
        if 'last_action_time' not in locals():
            last_action_time = 0
            
        current_time = pygame.time.get_ticks()
        if any(action) and current_time - last_action_time > 150: # 150ms cooldown
            obs, reward, terminated, truncated, info = env.step(action)
            last_action_time = current_time
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over! Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()
        else: # If no action, we still need to get the observation to draw
             obs = env._get_observation()

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    pygame.quit()