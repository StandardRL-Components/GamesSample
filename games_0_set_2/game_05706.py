
# Generated: 2025-08-28T05:50:06.903528
# Source Brief: brief_05706.md
# Brief Index: 5706

        
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
        "Controls: Use arrow keys to push all blocks simultaneously. "
        "Solve the puzzle by moving each colored block to its matching target."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push colored blocks onto their matching targets. "
        "Plan your moves carefully, as all blocks move together. "
        "You have a limited number of moves to solve each puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_BLOCKS = 4
        self.MAX_MOVES = 30
        self.WIN_REWARD = 50.0
        self.MOVE_PENALTY = -0.1
        self.BLOCK_ON_TARGET_REWARD = 1.0

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_game_over = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        # --- Visuals ---
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_WIN = (180, 255, 180)
        self.COLOR_LOSS = (255, 180, 180)

        self.PALETTE = [
            (0, (41, 224, 224)),    # Cyan
            (1, (224, 41, 224)),    # Magenta
            (2, (224, 224, 41)),    # Yellow
            (3, (247, 148, 29)),    # Orange
            (4, (41, 224, 90)),     # Green
            (5, (224, 41, 41)),     # Red
            (6, (120, 41, 224)),    # Purple
            (7, (255, 255, 255)),  # White
        ]
        
        self.grid_rect = self._calculate_grid_rect()
        self.cell_size = self.grid_rect.width // self.GRID_SIZE

        # --- Game State (initialized in reset) ---
        self.blocks = []
        self.targets = []
        self.move_count = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.reset()
        self.validate_implementation()
    
    def _calculate_grid_rect(self):
        # Center the grid, making it a square based on the smaller screen dimension
        screen_min_dim = min(self.WIDTH, self.HEIGHT)
        grid_pixel_size = int(screen_min_dim * 0.85)
        grid_left = (self.WIDTH - grid_pixel_size) // 2
        grid_top = (self.HEIGHT - grid_pixel_size) // 2
        return pygame.Rect(grid_left, grid_top, grid_pixel_size, grid_pixel_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.move_count = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Generate unique positions for blocks and targets
        all_positions = [(r, c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)]
        chosen_indices = self.np_random.choice(len(all_positions), size=self.NUM_BLOCKS * 2, replace=False)
        chosen_pos = [all_positions[i] for i in chosen_indices]

        self.targets = []
        self.blocks = []
        
        # Use a permutation of colors to ensure each block has a unique target color
        color_indices = self.np_random.permutation(len(self.PALETTE))[:self.NUM_BLOCKS]

        for i in range(self.NUM_BLOCKS):
            self.targets.append({'pos': chosen_pos[i], 'color_idx': color_indices[i]})
            self.blocks.append({'pos': chosen_pos[i + self.NUM_BLOCKS], 'color_idx': color_indices[i]})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        if movement != 0:
            # Only apply penalty and increment move counter if a push is attempted
            reward += self.MOVE_PENALTY
            self.move_count += 1
            self._push_blocks(movement)
            # Sound effect placeholder
            # sfx_slide.play()

        # State-based reward for blocks on targets
        solved_blocks = self._get_solved_block_indices()
        reward += len(solved_blocks) * self.BLOCK_ON_TARGET_REWARD
        
        self.score += reward
        terminated = self._check_termination(solved_blocks)
        
        if terminated and self.win:
            reward += self.WIN_REWARD
            self.score += self.WIN_REWARD # Add to total score as well

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _push_blocks(self, movement):
        # 1:up, 2:down, 3:left, 4:right
        direction_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = direction_map[movement]

        # Sort blocks to push them correctly in sequence
        # e.g., for a right push (dc=1), sort by col descending (reverse=True)
        sorted_indices = sorted(
            range(len(self.blocks)),
            key=lambda i: self.blocks[i]['pos'][0] * dr + self.blocks[i]['pos'][1] * dc,
            reverse=True
        )

        occupied = {b['pos'] for b in self.blocks}

        for i in sorted_indices:
            r, c = self.blocks[i]['pos']
            next_r, next_c = r + dr, c + dc

            if not (0 <= next_r < self.GRID_SIZE and 0 <= next_c < self.GRID_SIZE):
                continue # Hit wall

            if (next_r, next_c) in occupied:
                continue # Blocked by another block

            self.blocks[i]['pos'] = (next_r, next_c)
            occupied.remove((r, c))
            occupied.add((next_r, next_c))

    def _get_solved_block_indices(self):
        solved_indices = set()
        block_positions = {b['pos']: b['color_idx'] for b in self.blocks}
        for target in self.targets:
            if target['pos'] in block_positions and block_positions[target['pos']] == target['color_idx']:
                # Find which block is on this target
                for j, block in enumerate(self.blocks):
                    if block['pos'] == target['pos']:
                        solved_indices.add(j)
                        break
        return solved_indices

    def _check_termination(self, solved_blocks):
        if len(solved_blocks) == self.NUM_BLOCKS:
            self.game_over = True
            self.win = True
            # Sound effect placeholder
            # sfx_win.play()
        elif self.move_count >= self.MAX_MOVES:
            self.game_over = True
            self.win = False
            # Sound effect placeholder
            # sfx_loss.play()
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x_start = self.grid_rect.left + i * self.cell_size
            y_start = self.grid_rect.top + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x_start, self.grid_rect.top), (x_start, self.grid_rect.bottom))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_rect.left, y_start), (self.grid_rect.right, y_start))

        # Draw targets
        for target in self.targets:
            r, c = target['pos']
            color = self.PALETTE[target['color_idx']][1]
            center_x = int(self.grid_rect.left + (c + 0.5) * self.cell_size)
            center_y = int(self.grid_rect.top + (r + 0.5) * self.cell_size)
            radius = int(self.cell_size * 0.35)
            
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            
            inner_color = (*color, 60)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, inner_color)

        # Draw blocks
        solved_indices = self._get_solved_block_indices()
        for i, block in enumerate(self.blocks):
            r, c = block['pos']
            color = self.PALETTE[block['color_idx']][1]
            
            block_rect = pygame.Rect(
                self.grid_rect.left + c * self.cell_size + self.cell_size * 0.1,
                self.grid_rect.top + r * self.cell_size + self.cell_size * 0.1,
                self.cell_size * 0.8,
                self.cell_size * 0.8
            )
            
            is_solved = i in solved_indices
            if is_solved:
                # Draw a glow effect for solved blocks
                glow_center = block_rect.center
                glow_radius = int(block_rect.width * 0.7)
                glow_color = (*color, 100) # Semi-transparent
                pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, glow_color)
                pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], glow_radius, glow_color)
            
            pygame.draw.rect(self.screen, color, block_rect, border_radius=int(self.cell_size * 0.15))
            
            border_color = tuple(max(0, val - 40) for val in color)
            pygame.draw.rect(self.screen, border_color, block_rect, width=2, border_radius=int(self.cell_size * 0.15))

    def _render_ui(self):
        # Moves display
        moves_text = f"Moves: {self.move_count} / {self.MAX_MOVES}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_surf, (20, 20))

        # Score display
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = "PUZZLE SOLVED!"
                color = self.COLOR_WIN
            else:
                end_text = "OUT OF MOVES"
                color = self.COLOR_LOSS
                
            text_surf = self.font_game_over.render(end_text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves": self.move_count,
            "max_moves": self.MAX_MOVES,
            "is_success": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Push Block Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                elif not done:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if action[0] != 0 and not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()