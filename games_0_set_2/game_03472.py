
# Generated: 2025-08-27T23:27:31.880839
# Source Brief: brief_03472.md
# Brief Index: 3472

        
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
        "Controls: Use arrow keys to push all blocks. Complete all 3 stages before running out of moves."
    )

    # Must be a user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all blocks onto their matching colored targets. Plan your moves carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Visual & Game Constants ---
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 7
        self.CELL_SIZE = 48
        
        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.screen_width - self.grid_pixel_width) // 2
        self.GRID_OFFSET_Y = (self.screen_height - self.grid_pixel_height) // 2 + 20

        # Colors (Dracula theme for good contrast)
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID = (68, 71, 90)
        self.COLOR_TEXT = (248, 248, 242)
        self.COLOR_TEXT_DIM = (150, 150, 150)
        self.BLOCK_COLORS = [
            (255, 85, 85),   # Red
            (80, 250, 123),  # Green
            (139, 233, 253), # Cyan
            (255, 184, 108), # Orange
        ]
        self.TARGET_COLORS = [
            (c[0] // 3, c[1] // 3, c[2] // 3) for c in self.BLOCK_COLORS
        ]

        # --- Level Design ---
        self.levels = self._define_levels()

        # --- State variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.moves_remaining = 0
        self.blocks = []
        self.targets = []
        self.particles = []
        self.victory = False

        self.reset()
        
        # Run validation check
        # self.validate_implementation()
    
    def _define_levels(self):
        return [
            { # Stage 1
                "moves": 35,
                "blocks": [[2, 3, 0], [4, 1, 1]],
                "targets": [[9, 3, 0], [7, 5, 1]],
            },
            { # Stage 2
                "moves": 30,
                "blocks": [[2, 1, 0], [2, 5, 1], [5, 3, 2]],
                "targets": [[9, 1, 0], [9, 5, 1], [6, 3, 2]],
            },
            { # Stage 3
                "moves": 25,
                "blocks": [[1, 1, 0], [1, 5, 1], [10, 1, 2], [10, 5, 3]],
                "targets": [[5, 1, 3], [5, 5, 2], [6, 1, 1], [6, 5, 0]],
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.stage = 1
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        level_index = self.stage - 1
        if level_index < len(self.levels):
            level_data = self.levels[level_index]
            self.moves_remaining = level_data["moves"]
            # Deep copy of lists of lists
            self.blocks = [list(b) for b in level_data["blocks"]]
            self.targets = [list(t) for t in level_data["targets"]]
        else:
            # This case means all levels are won
            self.victory = True
            self.game_over = True


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        terminated = False
        
        if movement != 0:
            self.moves_remaining -= 1
            # Store old positions for reward calculation
            old_positions = {tuple(b[:2]): i for i, b in enumerate(self.blocks)}
            old_distances = self._get_block_target_distances()

            self._push_blocks(movement)
            
            # --- Calculate Reward ---
            new_distances = self._get_block_target_distances()
            reward += self._calculate_distance_reward(old_distances, new_distances)
            reward += self._calculate_on_target_reward(old_positions)
            
            self.score += reward

            # --- Check for Stage Completion ---
            if self._check_stage_complete():
                # Sound effect placeholder: # sfx_stage_clear()
                self.stage += 1
                if self.stage > len(self.levels):
                    reward += 100 # Win game bonus
                    self.score += 100
                    self.victory = True
                    self.game_over = True
                    terminated = True
                else:
                    reward += 50 # Stage complete bonus
                    self.score += 50
                    self._setup_stage()
        
        self.steps += 1
        
        if not terminated and (self.moves_remaining <= 0 or self.steps >= 1000):
            # Sound effect placeholder: # sfx_game_over()
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_block_target_distances(self):
        distances = []
        target_map = {tuple(t[2:]): tuple(t[:2]) for t in self.targets}
        for block in self.blocks:
            block_id = block[2]
            target_pos = target_map.get((block_id,), None)
            if target_pos:
                dist = abs(block[0] - target_pos[0]) + abs(block[1] - target_pos[1])
                distances.append(dist)
        return distances

    def _calculate_distance_reward(self, old_d, new_d):
        reward = 0
        for i in range(len(self.blocks)):
            if new_d[i] < old_d[i]:
                reward += 0.1
            elif new_d[i] > old_d[i]:
                reward -= 0.1
        return reward

    def _calculate_on_target_reward(self, old_positions_map):
        reward = 0
        target_coords = {tuple(t[:2]) for t in self.targets}
        for block in self.blocks:
            block_pos = tuple(block[:2])
            is_on_target = block_pos in target_coords
            was_on_target = block_pos in old_positions_map
            if is_on_target and not was_on_target:
                reward += 5
        return reward

    def _push_blocks(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]

        # Sort blocks to push them correctly
        # Push away from direction of movement
        sort_key = lambda b: b[0] * -dx + b[1] * -dy
        sorted_blocks = sorted(self.blocks, key=sort_key)
        
        occupied = {tuple(b[:2]) for b in self.blocks}

        for block in sorted_blocks:
            # Sound effect placeholder: # sfx_block_slide()
            ox, oy, color_idx = block
            nx, ny = ox, oy
            
            occupied.remove((ox, oy))

            while True:
                next_x, next_y = nx + dx, ny + dy
                if not (0 <= next_x < self.GRID_WIDTH and 0 <= next_y < self.GRID_HEIGHT):
                    break # Wall collision
                if (next_x, next_y) in occupied:
                    break # Block collision
                nx, ny = next_x, next_y
            
            block[0], block[1] = nx, ny
            occupied.add((nx, ny))

            # Create particles if block moved
            if (ox, oy) != (nx, ny):
                self._create_push_particles(ox, oy, nx, ny, color_idx)

    def _create_push_particles(self, start_x, start_y, end_x, end_y, color_idx):
        dx, dy = end_x - start_x, end_y - start_y
        dist = max(abs(dx), abs(dy))
        
        for i in range(dist * 2):
            progress = i / (dist * 2)
            px = start_x + dx * progress
            py = start_y + dy * progress
            
            screen_x = self.GRID_OFFSET_X + px * self.CELL_SIZE + self.CELL_SIZE / 2
            screen_y = self.GRID_OFFSET_Y + py * self.CELL_SIZE + self.CELL_SIZE / 2
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(4, 8)
            lifespan = self.np_random.integers(15, 30)
            color = self.BLOCK_COLORS[color_idx]
            
            self.particles.append([ [screen_x, screen_y], vel, size, color, lifespan ])

    def _check_stage_complete(self):
        target_map = {tuple(t[:2]): t[2] for t in self.targets}
        for block in self.blocks:
            pos = tuple(block[:2])
            if pos not in target_map or target_map[pos] != block[2]:
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._update_and_render_particles()
        self._render_grid()
        self._render_targets()
        self._render_blocks()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.grid_pixel_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.grid_pixel_width, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_targets(self):
        for x, y, color_idx in self.targets:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + x * self.CELL_SIZE,
                self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.TARGET_COLORS[color_idx], rect)

    def _render_blocks(self):
        for x, y, color_idx in self.blocks:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + x * self.CELL_SIZE + 4,
                self.GRID_OFFSET_Y + y * self.CELL_SIZE + 4,
                self.CELL_SIZE - 8, self.CELL_SIZE - 8
            )
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_idx], rect, border_radius=4)
            # Add a slight 3D effect
            shadow_color = tuple(c // 1.5 for c in self.BLOCK_COLORS[color_idx])
            pygame.draw.rect(self.screen, shadow_color, rect.move(0, 3), border_radius=4)
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_idx], rect, border_radius=4)


    def _update_and_render_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[1][1] += 0.1     # gravity
            p[2] -= 0.2        # size shrink
            p[4] -= 1          # lifespan
            if p[2] > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p[0][0]), int(p[0][1]), int(p[2]), p[3]
                )
        self.particles = [p for p in self.particles if p[4] > 0 and p[2] > 0]

    def _render_ui(self):
        # Moves and Stage
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        stage_text = self.font_small.render(f"Stage: {self.stage}/{len(self.levels)}", True, self.COLOR_TEXT_DIM)
        self.screen.blit(stage_text, (20, 55))

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.victory else "GAME OVER"
            color = self.BLOCK_COLORS[1] if self.victory else self.BLOCK_COLORS[0]
            
            end_text = self.font_main.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "moves_remaining": self.moves_remaining,
            "victory": self.victory,
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if action[0] != 0 or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over. Press 'R' to restart or 'Q' to quit.")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()