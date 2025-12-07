
# Generated: 2025-08-28T05:16:13.410654
# Source Brief: brief_02571.md
# Brief Index: 2571

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys (↑↓←→) to push all blocks in a row or column. "
        "The goal is to move all colored blocks onto their matching target crosses."
    )

    game_description = (
        "A fast-paced puzzle game. Push entire rows and columns of blocks to slide them "
        "into their target positions before the timer runs out. Plan your moves carefully!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.INITIAL_TIME = 30.0

        # Grid and block dimensions
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 6
        self.CELL_SIZE = 50
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.BLOCK_PADDING = 4
        
        # Visuals
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_TARGET = (255, 255, 255)
        self.BLOCK_COLORS = [
            ((255, 80, 80), (220, 50, 50)),   # Red
            ((80, 255, 80), (50, 220, 50)),   # Green
            ((80, 120, 255), (50, 90, 220)),  # Blue
            ((255, 255, 80), (220, 220, 50)), # Yellow
        ]
        self.ANIMATION_DURATION_FRAMES = 6 # 0.2s at 30fps

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_timer = pygame.font.Font(None, 48)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        
        self.blocks = []
        self.targets = []
        
        self.animation_timer = 0
        self.total_distance_before_move = 0
        self.blocks_on_target_before_move = 0
        
        self.reset()
        self.validate_implementation(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.INITIAL_TIME
        self.animation_timer = 0

        self._generate_level()
        
        self.total_distance_before_move = self._get_total_manhattan_distance()
        self.blocks_on_target_before_move = self._count_blocks_on_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1.0 / self.FPS
        reward = 0.0
        
        # Handle animation progression
        if self.animation_timer > 0:
            self.animation_timer -= 1
            if self.animation_timer == 0:
                # Snap to final position after animation
                for block in self.blocks:
                    block["vis_pos"] = self._grid_to_pixel(block["grid_pos"])
                # Sound effect placeholder: // sfx_block_snap

        # Process action only if not animating
        else:
            movement = action[0]
            if movement != 0: # 0 is no-op
                moved = self._perform_push(movement)
                if moved:
                    # Sound effect placeholder: // sfx_push
                    self.animation_timer = self.ANIMATION_DURATION_FRAMES
                    reward = self._calculate_reward()
                    self.score += reward
                    # Update state for the next move's reward calculation
                    self.total_distance_before_move = self._get_total_manhattan_distance()
                    self.blocks_on_target_before_move = self._count_blocks_on_target()

        terminated = self._check_termination()
        
        # Apply terminal rewards only once
        if terminated and not self.game_over:
            if self._count_blocks_on_target() == len(self.targets): # Win
                win_bonus = 50 + max(0, self.timer)
                self.score += win_bonus
                reward += win_bonus
                # Sound effect placeholder: // sfx_win_level
            elif self.timer <= 0: # Timeout
                loss_penalty = -100
                self.score += loss_penalty
                reward += loss_penalty
                # Sound effect placeholder: // sfx_lose_timeout
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_push(self, direction):
        original_positions = [tuple(b["grid_pos"]) for b in self.blocks]
        
        if direction in [1, 2]: # Up or Down
            for x in range(self.GRID_WIDTH):
                col_blocks = sorted([b for b in self.blocks if b["grid_pos"][0] == x], key=lambda b: b["grid_pos"][1])
                if direction == 2: # Down
                    col_blocks.reverse()
                
                next_free_y = 0 if direction == 1 else self.GRID_HEIGHT - 1
                for block in col_blocks:
                    block["grid_pos"] = (x, next_free_y)
                    block["start_vis_pos"] = block["vis_pos"]
                    next_free_y += 1 if direction == 1 else -1

        elif direction in [3, 4]: # Left or Right
            for y in range(self.GRID_HEIGHT):
                row_blocks = sorted([b for b in self.blocks if b["grid_pos"][1] == y], key=lambda b: b["grid_pos"][0])
                if direction == 4: # Right
                    row_blocks.reverse()

                next_free_x = 0 if direction == 3 else self.GRID_WIDTH - 1
                for block in row_blocks:
                    block["grid_pos"] = (next_free_x, y)
                    block["start_vis_pos"] = block["vis_pos"]
                    next_free_x += 1 if direction == 3 else -1
        
        new_positions = [tuple(b["grid_pos"]) for b in self.blocks]
        return original_positions != new_positions

    def _calculate_reward(self):
        # Distance-based reward
        current_dist = self._get_total_manhattan_distance()
        dist_change_reward = (self.total_distance_before_move - current_dist) * 0.1

        # On-target event reward
        current_on_target = self._count_blocks_on_target()
        on_target_reward = (current_on_target - self.blocks_on_target_before_move) * 1.0
        
        return dist_change_reward + on_target_reward

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if self._count_blocks_on_target() == len(self.targets):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _generate_level(self):
        self.blocks = []
        self.targets = []
        
        # Fixed level design for consistency
        level_targets = [(1, 1), (8, 1), (4, 4)]
        level_blocks = [(1, 4), (4, 1), (8, 4)]
        
        for i in range(len(level_targets)):
            grid_pos = level_blocks[i]
            vis_pos = self._grid_to_pixel(grid_pos)
            self.blocks.append({
                "grid_pos": list(grid_pos),
                "vis_pos": vis_pos,
                "start_vis_pos": vis_pos,
                "color_pair": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)],
                "target_idx": i
            })
            self.targets.append(level_targets[i])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw targets
        for i, pos in enumerate(self.targets):
            px, py = self._grid_to_pixel(pos)
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)][0]
            s = self.CELL_SIZE // 4
            pygame.draw.line(self.screen, color, (px - s, py), (px + s, py), 3)
            pygame.draw.line(self.screen, color, (px, py - s), (px, py + s), 3)

        # Draw blocks
        for block in self.blocks:
            if self.animation_timer > 0:
                progress = 1.0 - (self.animation_timer / self.ANIMATION_DURATION_FRAMES)
                progress = 1 - (1 - progress) ** 2 # Ease-out quadratic
                target_vis_pos = self._grid_to_pixel(block["grid_pos"])
                start_x, start_y = block["start_vis_pos"]
                end_x, end_y = target_vis_pos
                
                curr_x = start_x + (end_x - start_x) * progress
                curr_y = start_y + (end_y - start_y) * progress
                block["vis_pos"] = (curr_x, curr_y)

            px, py = block["vis_pos"]
            size = self.CELL_SIZE - self.BLOCK_PADDING * 2
            
            # Draw shadow/base
            base_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, block["color_pair"][1], base_rect, border_radius=6)
            
            # Draw top
            top_size = size - 8
            top_rect = pygame.Rect(px - top_size // 2, py - top_size // 2, top_size, top_size)
            pygame.draw.rect(self.screen, block["color_pair"][0], top_rect, border_radius=4)


    def _render_ui(self):
        # Render score/progress
        on_target = self._count_blocks_on_target()
        progress_text = f"{on_target} / {len(self.targets)}"
        text_surf = self.font_ui.render(progress_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (20, self.SCREEN_HEIGHT - 40))

        # Render timer
        time_str = f"{max(0, self.timer):.1f}"
        time_color = (100, 255, 100) if self.timer > 10 else (255, 100, 100)
        timer_surf = self.font_timer.render(time_str, True, time_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_surf, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "blocks_on_target": self._count_blocks_on_target(),
        }

    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        px = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _count_blocks_on_target(self):
        count = 0
        for block in self.blocks:
            if tuple(block["grid_pos"]) == self.targets[block["target_idx"]]:
                count += 1
        return count

    def _get_total_manhattan_distance(self):
        total_dist = 0
        for block in self.blocks:
            bx, by = block["grid_pos"]
            tx, ty = self.targets[block["target_idx"]]
            total_dist += abs(bx - tx) + abs(by - ty)
        return total_dist

    def close(self):
        pygame.quit()

    @staticmethod
    def validate_implementation(self):
        print("Running implementation validation...")
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        action = [movement, 0, 0] # Space and Shift are not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()