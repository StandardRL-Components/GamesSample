
# Generated: 2025-08-28T05:08:52.312025
# Source Brief: brief_05478.md
# Brief Index: 5478

        
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
        "Controls: Use arrow keys to push all colored blocks. "
        "Goal: Move each block to its matching target zone."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Push colored blocks into their matching target zones "
        "before the timer runs out. Each push moves all blocks at once."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 40
        self.GRID_COLS = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_ROWS = self.SCREEN_HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 200
        self.NUM_PIXELS = 4

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.PIXEL_COLORS = {
            "cyan": {"main": (0, 255, 255), "highlight": (180, 255, 255), "target": (0, 100, 100)},
            "magenta": {"main": (255, 0, 255), "highlight": (255, 180, 255), "target": (100, 0, 100)},
            "yellow": {"main": (255, 255, 0), "highlight": (255, 255, 180), "target": (100, 100, 0)},
            "green": {"main": (0, 255, 0), "highlight": (180, 255, 180), "target": (0, 100, 0)},
        }
        self.COLOR_KEYS = list(self.PIXEL_COLORS.keys())

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.pixels = []
        self.win_message = ""
        
        # --- Validate Implementation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.pixels = []

        # Generate a new level layout
        all_cells = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_cells)
        
        if len(all_cells) < self.NUM_PIXELS * 2:
            raise ValueError("Grid is too small for the number of pixels and targets.")

        chosen_cells = all_cells[:self.NUM_PIXELS * 2]
        target_positions = chosen_cells[:self.NUM_PIXELS]
        pixel_start_positions = chosen_cells[self.NUM_PIXELS:]

        for i in range(self.NUM_PIXELS):
            color_key = self.COLOR_KEYS[i % len(self.COLOR_KEYS)]
            self.pixels.append({
                "pos": list(pixel_start_positions[i]),
                "target": list(target_positions[i]),
                "color_key": color_key,
                "on_target": False
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 (unused)
        # shift_held = action[2] == 1 (unused)
        reward = 0

        old_distances = {i: self._manhattan_distance(p['pos'], p['target']) for i, p in enumerate(self.pixels)}
        old_on_target_state = {i: p['on_target'] for i, p in enumerate(self.pixels)}

        if movement in [1, 2, 3, 4]: # 1=up, 2=down, 3=left, 4=right
            self._handle_push(movement)
            # sfx: push

        self.steps += 1
        
        all_on_target = True
        for i, p in enumerate(self.pixels):
            new_dist = self._manhattan_distance(p['pos'], p['target'])
            
            if new_dist < old_distances[i]:
                reward += 0.1
            elif new_dist > old_distances[i]:
                reward -= 0.1

            p['on_target'] = (new_dist == 0)
            if p['on_target'] and not old_on_target_state[i]:
                reward += 1.0 # sfx: success
            
            if not p['on_target']:
                all_on_target = False

        terminated = False
        if all_on_target:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_message = "COMPLETE!"
            # sfx: win_fanfare
        elif self.steps >= self.MAX_STEPS:
            reward -= 10.0
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP!"
            # sfx: timeout_buzzer

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement_action):
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        direction = direction_map[movement_action]
        
        sort_key = 1 if direction[1] != 0 else 0
        reverse = direction[sort_key] > 0
        sorted_pixels = sorted(self.pixels, key=lambda p: p['pos'][sort_key], reverse=reverse)

        current_positions = {tuple(p['pos']) for p in self.pixels}

        for pixel in sorted_pixels:
            old_pos = tuple(pixel['pos'])
            next_pos = (pixel['pos'][0] + direction[0], pixel['pos'][1] + direction[1])

            if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                continue

            if next_pos in current_positions:
                continue
            
            pixel['pos'] = list(next_pos)
            current_positions.remove(old_pos)
            current_positions.add(next_pos)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_targets()
        self._render_pixels()
        self._render_ui()
        if self.game_over:
            self._render_end_message()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_targets(self):
        for pixel_data in self.pixels:
            color_info = self.PIXEL_COLORS[pixel_data['color_key']]
            rect = pygame.Rect(
                pixel_data['target'][0] * self.CELL_SIZE,
                pixel_data['target'][1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, color_info['target'], rect)
            pygame.draw.rect(self.screen, color_info['main'], rect, 2)

    def _render_pixels(self):
        for pixel_data in self.pixels:
            color_info = self.PIXEL_COLORS[pixel_data['color_key']]
            x, y = pixel_data['pos']
            
            base_rect = pygame.Rect(
                x * self.CELL_SIZE,
                y * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, color_info['main'], base_rect)

            inset = self.CELL_SIZE // 4
            highlight_rect = pygame.Rect(
                x * self.CELL_SIZE + inset,
                y * self.CELL_SIZE + inset,
                self.CELL_SIZE - 2 * inset,
                self.CELL_SIZE - 2 * inset
            )
            pygame.draw.rect(self.screen, color_info['highlight'], highlight_rect)
            
            if pixel_data['on_target']:
                center_x = base_rect.centerx
                center_y = base_rect.centery
                pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), 5)

    def _render_ui(self):
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        time_left = self.MAX_STEPS - self.steps
        time_text = f"MOVES LEFT: {time_left}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

    def _render_end_message(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
        msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "pixels_on_target": sum(1 for p in self.pixels if p['on_target'])
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        _, _ = self.reset()
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Pusher")
    screen_human = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0

    print(env.user_guide)

    while running:
        move_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                    continue
                
                if event.key == pygame.K_UP: move_action = 1
                elif event.key == pygame.K_DOWN: move_action = 2
                elif event.key == pygame.K_LEFT: move_action = 3
                elif event.key == pygame.K_RIGHT: move_action = 4
                
                if move_action != 0:
                    full_action = [move_action, 0, 0]
                    obs, reward, terminated, truncated, info = env.step(full_action)
                    total_reward += reward
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

                    if terminated:
                        print(f"Game Over! Final Score: {info['score']:.2f}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()