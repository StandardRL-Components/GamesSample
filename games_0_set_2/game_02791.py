
# Generated: 2025-08-28T05:57:31.672589
# Source Brief: brief_02791.md
# Brief Index: 2791

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks. Your goal is to move each colored block onto its matching target square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all blocks simultaneously to solve the puzzle within the move limit. Blocks lock in place on their matching targets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 150, 255),  # Blue
            (100, 220, 100), # Green
            (255, 200, 80),  # Yellow
            (200, 100, 255), # Purple
            (80, 220, 220),  # Cyan
        ]

        # Game state variables
        self.level = 1
        self.blocks = []
        self.targets = []
        self.grid_rows = 0
        self.grid_cols = 0
        self.max_moves = 0
        self.moves_remaining = 0
        self.cell_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self.newly_locked_blocks = []

        # Initialize state
        self.reset()

        # Run validation
        self.validate_implementation()

    def _generate_level(self):
        """Generates a new puzzle based on the current level."""
        # Difficulty scaling
        self.grid_rows = min(10 + self.level - 1, 15)
        self.grid_cols = min(10 + self.level - 1, 22)
        num_blocks = min(3 + self.level - 1, len(self.BLOCK_COLORS) * 2)
        self.max_moves = 20 + self.level * 5
        self.moves_remaining = self.max_moves

        # Determine grid rendering properties
        self.cell_size = int(min(
            (self.screen_width * 0.9) / self.grid_cols,
            (self.screen_height * 0.8) / self.grid_rows
        ))
        grid_width = self.grid_cols * self.cell_size
        grid_height = self.grid_rows * self.cell_size
        self.grid_offset_x = (self.screen_width - grid_width) // 2
        self.grid_offset_y = (self.screen_height - grid_height) // 2 + 30

        # Generate positions
        all_coords = [(x, y) for x in range(self.grid_cols) for y in range(self.grid_rows)]
        self.np_random.shuffle(all_coords)

        self.targets = []
        self.blocks = []
        
        used_colors = (self.BLOCK_COLORS * (num_blocks // len(self.BLOCK_COLORS) + 1))[:num_blocks]
        self.np_random.shuffle(used_colors)

        target_coords = all_coords[:num_blocks]
        block_coords = all_coords[num_blocks:num_blocks*2]

        for i in range(num_blocks):
            color = used_colors[i]
            self.targets.append({'pos': np.array(target_coords[i]), 'color': color})
            self.blocks.append({'pos': np.array(block_coords[i]), 'color': color, 'locked': False})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.newly_locked_blocks = []

        if options and 'level' in options:
            self.level = options['level']

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        self.newly_locked_blocks.clear()

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if movement != 0:
            self.moves_remaining -= 1
            reward -= 0.1  # Cost for making a move
            
            moved = self._perform_push(movement)
            if not moved:
                reward += 0.05 # Smaller penalty for non-moving pushes
            
            new_locks_reward = self._check_and_lock_blocks()
            reward += new_locks_reward
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            all_locked = all(b['locked'] for b in self.blocks)
            if all_locked:
                reward += 100.0  # Big reward for winning
                self.level += 1 # Progress to next level on next reset
            elif self.moves_remaining <= 0:
                reward -= 10.0 # Penalty for losing
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_push(self, movement_action):
        """Pushes all non-locked blocks in the given direction."""
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement_action]

        # Sort blocks to handle pushes correctly.
        # For up/left pushes, sort ascending. For down/right, sort descending.
        sort_reverse = (dx > 0 or dy > 0)
        sorted_indices = sorted(
            range(len(self.blocks)),
            key=lambda k: (self.blocks[k]['pos'][0], self.blocks[k]['pos'][1]),
            reverse=sort_reverse
        )
        
        any_block_moved = False
        
        current_block_positions = {tuple(b['pos']) for b in self.blocks}

        for i in sorted_indices:
            block = self.blocks[i]
            if block['locked']:
                continue

            original_pos = tuple(block['pos'])
            current_block_positions.remove(original_pos)
            
            new_pos = block['pos'].copy()
            while True:
                next_pos = new_pos + np.array([dx, dy])
                # Check grid boundaries
                if not (0 <= next_pos[0] < self.grid_cols and 0 <= next_pos[1] < self.grid_rows):
                    break
                # Check for collision with other blocks
                if tuple(next_pos) in current_block_positions:
                    break
                new_pos = next_pos
            
            if not np.array_equal(block['pos'], new_pos):
                any_block_moved = True

            block['pos'] = new_pos
            current_block_positions.add(tuple(new_pos))

        return any_block_moved

    def _check_and_lock_blocks(self):
        """Checks if any blocks are on their targets and locks them."""
        reward = 0
        for i, block in enumerate(self.blocks):
            if block['locked']:
                continue
            for target in self.targets:
                if np.array_equal(block['pos'], target['pos']) and block['color'] == target['color']:
                    block['locked'] = True
                    reward += 1.0  # Reward for placing a block
                    self.newly_locked_blocks.append(i)
                    break # Block can only match one target
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        
        all_locked = all(b['locked'] for b in self.blocks)
        out_of_moves = self.moves_remaining <= 0
        
        if all_locked or out_of_moves or self.steps >= 1000:
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
        # Draw grid lines
        for r in range(self.grid_rows + 1):
            y = self.grid_offset_y + r * self.cell_size
            start = (self.grid_offset_x, y)
            end = (self.grid_offset_x + self.grid_cols * self.cell_size, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.grid_cols + 1):
            x = self.grid_offset_x + c * self.cell_size
            start = (x, self.grid_offset_y)
            end = (x, self.grid_offset_y + self.grid_rows * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw targets
        for target in self.targets:
            x = self.grid_offset_x + target['pos'][0] * self.cell_size
            y = self.grid_offset_y + target['pos'][1] * self.cell_size
            rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, tuple(c*0.4 for c in target['color']), rect) # Darker color for targets

        # Draw blocks
        for i, block in enumerate(self.blocks):
            x = self.grid_offset_x + block['pos'][0] * self.cell_size
            y = self.grid_offset_y + block['pos'][1] * self.cell_size
            rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            
            # Draw a bright flash effect for newly locked blocks
            if i in self.newly_locked_blocks:
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(6, 6), border_radius=3)

            pygame.draw.rect(self.screen, block['color'], rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), rect, width=max(1, self.cell_size//8), border_radius=3)
            
            if block['locked']:
                # Draw a lock icon or symbol
                center_x, center_y = rect.center
                dot_radius = max(2, self.cell_size // 6)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, dot_radius, (255,255,255,180))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, dot_radius, (0,0,0,180))

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Level
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(midtop=(self.screen_width // 2, 20))
        self.screen.blit(level_text, level_rect)

        # Game Over Message
        if self.game_over:
            all_locked = all(b['locked'] for b in self.blocks)
            if all_locked:
                msg = "PUZZLE SOLVED!"
                color = self.COLOR_WIN
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "blocks_locked": sum(1 for b in self.blocks if b['locked']),
            "total_blocks": len(self.blocks),
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
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Block Pusher")
    
    terminated = False
    
    print(GameEnv.user_guide)

    while not terminated:
        # Convert numpy array back to pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        movement = 0 # No-op by default
        
        # Pygame event loop
        # This is for manual control, not for the agent
        wait_for_input = True
        while wait_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    wait_for_input = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        movement = 1
                        wait_for_input = False
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                        wait_for_input = False
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                        wait_for_input = False
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                        wait_for_input = False
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                        wait_for_input = False
                    elif event.key == pygame.K_q: # Quit
                        terminated = True
                        wait_for_input = False

        if terminated:
            break

        # Construct the action for the environment
        action = [movement, 0, 0] # space and shift are not used
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        print(f"Action: {movement}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        
        if terminated:
            # Display final frame for 2 seconds
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            
            # Reset for a new game
            obs, info = env.reset()
            terminated = False

    env.close()