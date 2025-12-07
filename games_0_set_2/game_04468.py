
# Generated: 2025-08-28T02:29:46.200884
# Source Brief: brief_04468.md
# Brief Index: 4468

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character (white square). "
        "Push colored blocks onto their matching target squares."
    )

    game_description = (
        "A fast-paced puzzle game. Push all 15 colored blocks to their matching targets "
        "before the 60-second timer runs out. Plan your moves carefully to avoid getting stuck!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_SIZE = 32
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    NUM_BLOCKS = 15
    TIME_LIMIT_SECONDS = 60.0
    MAX_STEPS = int(TIME_LIMIT_SECONDS * 30) # 30 FPS

    # --- Colors (Retro Palette) ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 45)
    
    BLOCK_COLORS = [
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (141, 21, 70),
        (0, 135, 81), (0, 0, 0), (95, 87, 79), (194, 195, 199),
        (255, 204, 170), (29, 43, 83), (126, 37, 83)
    ]
    
    # --- Data Structures ---
    Block = namedtuple("Block", ["id", "color", "grid_pos", "pixel_pos", "on_target"])
    Target = namedtuple("Target", ["id", "color", "grid_pos"])

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
        
        self.render_mode = render_mode
        self.np_random = None

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0.0
        self.player_grid_pos = (0, 0)
        self.player_pixel_pos = [0.0, 0.0]
        self.blocks = []
        self.targets = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Timing ---
        self.clock.tick(30)
        self.time_remaining -= 1.0 / 30.0
        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        reward = 0
        
        # --- Game Logic ---
        if dx != 0 or dy != 0:
            reward += self._handle_player_move(dx, dy)

        self._update_animations()

        # Check for newly placed blocks after move and animation
        newly_placed_reward = self._update_score_and_block_states()
        reward += newly_placed_reward

        # --- Termination Check ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        """Creates a new, solvable puzzle configuration."""
        grid_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(grid_cells)

        self.targets = []
        self.blocks = []
        
        # Place targets and blocks on top of them (solved state)
        target_positions = set()
        for i in range(self.NUM_BLOCKS):
            pos = grid_cells.pop()
            target_positions.add(pos)
            color = self.BLOCK_COLORS[i]
            self.targets.append(self.Target(id=i, color=color, grid_pos=pos))
            
            pixel_pos = [self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE]
            self.blocks.append(self.Block(id=i, color=color, grid_pos=pos, pixel_pos=pixel_pos, on_target=True))

        # Scramble the board by performing random "pulls"
        num_scramble_moves = self.np_random.integers(50, 100)
        for _ in range(num_scramble_moves):
            block_idx = self.np_random.integers(0, len(self.blocks))
            block_to_move = self.blocks[block_idx]
            
            # Try to pull the block into an empty adjacent space
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(possible_moves)
            
            for dx, dy in possible_moves:
                new_pos = (block_to_move.grid_pos[0] + dx, block_to_move.grid_pos[1] + dy)
                
                # Check if new position is valid and empty
                if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                    is_occupied = any(b.grid_pos == new_pos for b in self.blocks)
                    if not is_occupied:
                        # Perform the "pull"
                        new_pixel_pos = [self.GRID_OFFSET_X + new_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + new_pos[1] * self.CELL_SIZE]
                        self.blocks[block_idx] = block_to_move._replace(grid_pos=new_pos, pixel_pos=new_pixel_pos)
                        break

        # Place player in a random empty spot
        occupied_cells = {b.grid_pos for b in self.blocks}
        empty_cells = [cell for cell in grid_cells if cell not in occupied_cells]
        if not empty_cells: # Failsafe
             empty_cells = [(0,0)]
        self.player_grid_pos = self.np_random.choice(empty_cells)
        self.player_pixel_pos = [
            self.GRID_OFFSET_X + self.player_grid_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.player_grid_pos[1] * self.CELL_SIZE
        ]

        self._update_score_and_block_states()

    def _handle_player_move(self, dx, dy):
        """Handles player movement and block pushing logic."""
        target_x = self.player_grid_pos[0] + dx
        target_y = self.player_grid_pos[1] + dy

        # Wall collision
        if not (0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT):
            return 0

        # Check for block at target position
        block_at_target_idx = -1
        for i, block in enumerate(self.blocks):
            if block.grid_pos == (target_x, target_y):
                block_at_target_idx = i
                break

        # --- Case 1: Move to empty space ---
        if block_at_target_idx == -1:
            self.player_grid_pos = (target_x, target_y)
            return 0 # No reward for just moving

        # --- Case 2: Attempt to push block(s) ---
        push_chain = []
        current_pos = (target_x, target_y)
        
        while True:
            block_idx = -1
            for i, b in enumerate(self.blocks):
                if b.grid_pos == current_pos:
                    block_idx = i
                    break
            
            if block_idx != -1:
                push_chain.append(block_idx)
                next_pos_x = current_pos[0] + dx
                next_pos_y = current_pos[1] + dy
                
                # Check if the chain hits a wall
                if not (0 <= next_pos_x < self.GRID_WIDTH and 0 <= next_pos_y < self.GRID_HEIGHT):
                    return 0 # Push failed, hit wall
                current_pos = (next_pos_x, next_pos_y)
            else:
                break # End of chain is an empty space, push is valid

        # Push is valid, now execute it and calculate rewards
        reward = 0
        
        # Store pre-move distances for reward calculation
        pre_move_distances = {
            idx: self._manhattan_distance(self.blocks[idx].grid_pos, self.targets[self.blocks[idx].id].grid_pos)
            for idx in push_chain
        }

        # Move blocks in reverse order
        for block_idx in reversed(push_chain):
            block = self.blocks[block_idx]
            new_pos = (block.grid_pos[0] + dx, block.grid_pos[1] + dy)
            self.blocks[block_idx] = block._replace(grid_pos=new_pos)
            # SFX: // block_slide.wav

        # Move player
        self.player_grid_pos = (target_x, target_y)

        # Calculate rewards based on distance change
        for idx in push_chain:
            block = self.blocks[idx]
            target = self.targets[block.id]
            post_move_distance = self._manhattan_distance(block.grid_pos, target.grid_pos)
            delta_dist = post_move_distance - pre_move_distances[idx]
            
            if delta_dist < 0:
                reward += 0.1  # Moved closer
            elif delta_dist > 0:
                reward -= 0.1  # Moved further

        return reward

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _update_score_and_block_states(self):
        """Updates block 'on_target' status and score. Returns reward for new placements."""
        reward = 0
        current_score = 0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            target = self.targets[block.id]
            is_on_target = block.grid_pos == target.grid_pos
            
            if is_on_target and not block.on_target:
                reward += 1.0  # Event reward for placing a block
                # SFX: // success_chime.wav
            
            self.blocks[i] = block._replace(on_target=is_on_target)
            if is_on_target:
                current_score += 1
        
        self.score = current_score
        return reward

    def _check_termination(self):
        """Checks for win/loss conditions and returns terminal reward."""
        if self.score == self.NUM_BLOCKS:
            # SFX: // level_complete.wav
            return True, 100.0  # Win
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            # SFX: // game_over.wav
            return True, -100.0 # Lose
        return False, 0.0

    def _update_animations(self):
        """Smoothly interpolates visual positions towards logical grid positions."""
        lerp_rate = 0.4
        
        # Player
        target_px_x = self.GRID_OFFSET_X + self.player_grid_pos[0] * self.CELL_SIZE
        target_px_y = self.GRID_OFFSET_Y + self.player_grid_pos[1] * self.CELL_SIZE
        self.player_pixel_pos[0] += (target_px_x - self.player_pixel_pos[0]) * lerp_rate
        self.player_pixel_pos[1] += (target_px_y - self.player_pixel_pos[1]) * lerp_rate

        # Blocks
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            target_bx_x = self.GRID_OFFSET_X + block.grid_pos[0] * self.CELL_SIZE
            target_bx_y = self.GRID_OFFSET_Y + block.grid_pos[1] * self.CELL_SIZE
            
            new_pixel_pos_x = block.pixel_pos[0] + (target_bx_x - block.pixel_pos[0]) * lerp_rate
            new_pixel_pos_y = block.pixel_pos[1] + (target_bx_y - block.pixel_pos[1]) * lerp_rate
            
            self.blocks[i] = block._replace(pixel_pos=[new_pixel_pos_x, new_pixel_pos_y])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw targets
        for target in self.targets:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + target.grid_pos[0] * self.CELL_SIZE,
                self.GRID_OFFSET_Y + target.grid_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect) # BG for target
            pygame.draw.rect(self.screen, target.color, rect.inflate(-self.CELL_SIZE * 0.7, -self.CELL_SIZE * 0.7))

        # Draw blocks
        for block in self.blocks:
            rect = pygame.Rect(
                int(block.pixel_pos[0]), int(block.pixel_pos[1]),
                self.CELL_SIZE, self.CELL_SIZE
            )
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, block.color, inner_rect, border_radius=4)
            
            if block.on_target:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, inner_rect, width=2, border_radius=4)

        # Draw player
        player_size = self.CELL_SIZE // 3
        player_rect = pygame.Rect(
            int(self.player_pixel_pos[0] + (self.CELL_SIZE - player_size) / 2),
            int(self.player_pixel_pos[1] + (self.CELL_SIZE - player_size) / 2),
            player_size, player_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.GRID_OFFSET_Y - 5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score} / {self.NUM_BLOCKS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Time
        time_color = self.COLOR_UI_TEXT
        if self.time_remaining < 10:
            time_color = self.BLOCK_COLORS[0] # Red
        time_str = f"TIME: {max(0, self.time_remaining):.1f}"
        time_text = self.font_large.render(time_str, True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(time_text, time_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a display window
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not terminated:
        # --- Human Input ---
        # Default action is NO-OP
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

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

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    env.close()