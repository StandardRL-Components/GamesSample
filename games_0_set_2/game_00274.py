
# Generated: 2025-08-27T13:08:57.796404
# Source Brief: brief_00274.md
# Brief Index: 274

        
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


# --- Constants ---
WIDTH, HEIGHT = 640, 400
GRID_WIDTH, GRID_HEIGHT = 16, 10
CELL_SIZE = 40
MAX_MOVES = 20
NUM_BLOCKS = 3
SCRAMBLE_MOVES = 75 # How many random moves to make when generating a puzzle

# --- Colors ---
COLOR_BG = (20, 20, 30)
COLOR_GRID = (40, 40, 50)
COLOR_PLAYER = (50, 150, 255)
BLOCK_COLORS = [(255, 80, 80), (80, 255, 80), (255, 255, 80)]
TARGET_OUTLINE_COLOR = (255, 255, 255)
TEXT_COLOR = (220, 220, 220)
GAMEOVER_WIN_COLOR = (100, 255, 100)
GAMEOVER_LOSE_COLOR = (255, 100, 100)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Push blocks onto their matching targets."
    )

    game_description = (
        "A minimalist puzzle game. Push all blocks to their targets within the move limit."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.block_positions = None
        self.target_positions = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks_on_target_before_move = set()

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._generate_puzzle()
        self.blocks_on_target_before_move = self._get_on_target_blocks()
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        # 1. Place targets away from edges
        possible_coords = []
        for x in range(1, GRID_WIDTH - 1):
            for y in range(1, GRID_HEIGHT - 1):
                possible_coords.append((x, y))
        
        target_indices = self.np_random.choice(len(possible_coords), NUM_BLOCKS, replace=False)
        self.target_positions = [possible_coords[i] for i in target_indices]

        # 2. Place blocks on targets (solved state)
        self.block_positions = list(self.target_positions)

        # 3. Place player in a random empty spot
        empty_cells = self._get_empty_cells()
        player_idx = self.np_random.choice(len(empty_cells))
        self.player_pos = empty_cells[player_idx]

        # 4. Scramble the puzzle with random reverse moves to ensure solvability
        for _ in range(SCRAMBLE_MOVES):
            block_idx = self.np_random.integers(0, NUM_BLOCKS)
            block_pos = self.block_positions[block_idx]

            # Choose a random direction to "pull" the block from
            # This is equivalent to the direction the block will move
            move_options = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.np_random.shuffle(move_options)

            for dx, dy in move_options:
                pull_from_pos = (block_pos[0] - dx, block_pos[1] - dy)
                new_block_pos = (block_pos[0] + dx, block_pos[1] + dy)
                
                # Check if the move is valid in reverse
                if self._is_valid_and_empty(new_block_pos, exclude_player=True) and \
                   self._is_valid_and_empty(pull_from_pos, exclude_player=True):
                    # Execute the reverse move
                    self.block_positions[block_idx] = new_block_pos
                    self.player_pos = block_pos # Player occupies the block's old spot
                    break # Move to next scramble step

        # Final check to ensure player is not on a block
        if self.player_pos in self.block_positions:
            empty_cells = self._get_empty_cells()
            if empty_cells:
                self.player_pos = empty_cells[self.np_random.choice(len(empty_cells))]

    def _get_empty_cells(self):
        occupied = set(self.block_positions)
        if self.player_pos:
            occupied.add(self.player_pos)
        empty = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if (x, y) not in occupied:
                    empty.append((x, y))
        return empty

    def _is_valid_and_empty(self, pos, exclude_player=False):
        x, y = pos
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return False
        if pos in self.block_positions:
            return False
        if not exclude_player and pos == self.player_pos:
            return False
        return True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Per-step penalty
        self.steps += 1
        
        if movement != 0: # 0 is a no-op that still consumes a turn
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = move_map[movement]

            px, py = self.player_pos
            next_player_pos = (px + dx, py + dy)

            # Case 1: Pushing a block
            if next_player_pos in self.block_positions:
                block_idx = self.block_positions.index(next_player_pos)
                block_behind_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)

                if self._is_valid_and_empty(block_behind_pos, exclude_player=True):
                    # Valid push, calculate distance-based reward
                    old_dist = self._manhattan_distance(next_player_pos, self.target_positions[block_idx])
                    new_dist = self._manhattan_distance(block_behind_pos, self.target_positions[block_idx])
                    if new_dist > old_dist:
                        reward -= 0.2
                    
                    # Move block and player
                    self.block_positions[block_idx] = block_behind_pos
                    self.player_pos = next_player_pos
                    # sfx: push_block.wav

            # Case 2: Moving to an empty space
            elif self._is_valid_and_empty(next_player_pos):
                self.player_pos = next_player_pos
                # sfx: player_move.wav
            # else: sfx: bump_wall.wav

        # Calculate reward for blocks landing on targets
        blocks_on_target_after_move = self._get_on_target_blocks()
        newly_on_target = blocks_on_target_after_move - self.blocks_on_target_before_move
        if newly_on_target:
            reward += len(newly_on_target) * 1.0
            # sfx: block_on_target.wav
        
        self.blocks_on_target_before_move = blocks_on_target_after_move

        # Check for termination conditions
        terminated = False
        puzzle_solved = len(self.blocks_on_target_before_move) == NUM_BLOCKS
        
        if puzzle_solved:
            terminated = True
            self.game_over = True
            reward += 10.0 # Goal-oriented reward
            # sfx: puzzle_solved.wav
        elif self.steps >= MAX_MOVES:
            terminated = True
            self.game_over = True
            reward -= 10.0 # Goal-oriented penalty
            # sfx: puzzle_failed.wav

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_on_target_blocks(self):
        return {i for i, b_pos in enumerate(self.block_positions) if b_pos == self.target_positions[i]}

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, WIDTH + 1, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT + 1, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (WIDTH, y))

        # Draw targets
        for i, (tx, ty) in enumerate(self.target_positions):
            rect = pygame.Rect(tx * CELL_SIZE, ty * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            inner_rect = rect.inflate(-8, -8)
            # Faint fill matching block color
            target_color = BLOCK_COLORS[i]
            faint_color = (target_color[0] // 4, target_color[1] // 4, target_color[2] // 4)
            pygame.draw.rect(self.screen, faint_color, inner_rect, border_radius=4)
            # Bright outline
            pygame.draw.rect(self.screen, target_color, inner_rect, width=2, border_radius=4)

        # Draw blocks
        for i, (bx, by) in enumerate(self.block_positions):
            rect = pygame.Rect(bx * CELL_SIZE, by * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, BLOCK_COLORS[i], inner_rect, border_radius=6)
            # Subtle 3D highlight
            highlight_rect = pygame.Rect(inner_rect.left + 2, inner_rect.top + 2, inner_rect.width - 4, 4)
            pygame.draw.rect(self.screen, (255, 255, 255, 50), highlight_rect, border_top_left_radius=4, border_top_right_radius=4)

        # Draw player
        px, py = self.player_pos
        rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        inner_rect = rect.inflate(-8, -8)
        pygame.draw.rect(self.screen, COLOR_PLAYER, inner_rect, border_radius=6)
        highlight_rect = pygame.Rect(inner_rect.left + 2, inner_rect.top + 2, inner_rect.width - 4, 4)
        pygame.draw.rect(self.screen, (255, 255, 255, 80), highlight_rect, border_top_left_radius=4, border_top_right_radius=4)


    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.steps:02d}/{MAX_MOVES}", True, TEXT_COLOR)
        self.screen.blit(moves_text, (15, 10))

        score_text = self.font_main.render(f"Score: {self.score: >5.1f}", True, TEXT_COLOR)
        score_rect = score_text.get_rect(topright=(WIDTH - 15, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            puzzle_solved = len(self._get_on_target_blocks()) == NUM_BLOCKS
            text, color = ("PUZZLE SOLVED!", GAMEOVER_WIN_COLOR) if puzzle_solved else ("OUT OF MOVES", GAMEOVER_LOSE_COLOR)
            
            gameover_surf = self.font_gameover.render(text, True, color)
            gameover_rect = gameover_surf.get_rect(center=(WIDTH / 2, HEIGHT / 2))
            self.screen.blit(gameover_surf, gameover_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_on_target": len(self._get_on_target_blocks()),
            "puzzle_solved": len(self._get_on_target_blocks()) == NUM_BLOCKS
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sokoban Puzzle")
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    continue
                
                # Take step only on key press for turn-based feel
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")
                    if terminated:
                        done = True

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

        if done and info.get('puzzle_solved', False):
            print("Congratulations! You solved the puzzle.")
        elif done:
            print("Game over. Press 'r' to try again or close the window.")
            # Allow reset after game over
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                        wait_for_reset = False
                clock.tick(10)


    env.close()