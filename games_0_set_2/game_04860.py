
# Generated: 2025-08-28T03:14:38.036463
# Source Brief: brief_04860.md
# Brief Index: 4860

        
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
    user_guide = "Controls: Use arrow keys to push all blocks. Goal: Match blocks to targets."

    # Must be a short, user-facing description of the game:
    game_description = "Push colored blocks onto their matching targets in this grid-based puzzle. You have a limited number of moves to solve each puzzle."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_W, GRID_H = 10, 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_W * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_H * CELL_SIZE) // 2
    
    MAX_MOVES = 20
    NUM_BLOCKS = 10
    SCRAMBLE_MOVES = 15
    ANIMATION_FRAMES = 8

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (230, 230, 230)
    
    BLOCK_COLORS = [
        (255, 87, 87),    # Red
        (87, 187, 255),   # Blue
        (87, 255, 87),    # Green
        (255, 255, 87),   # Yellow
        (255, 87, 255),   # Magenta
        (87, 255, 255),   # Cyan
        (255, 165, 0),    # Orange
        (180, 87, 255),   # Purple
        (200, 200, 200),  # White
        (100, 255, 180),  # Sea Green
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_gameover = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.blocks = []
        self.targets = []
        self.moves_left = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.game_state = 'IDLE' # IDLE, ANIMATING
        self.animation_progress = 0.0
        self.anim_start_pos = []
        self.anim_end_pos = []
        self.num_on_target_last_turn = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0.0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.game_state = 'IDLE'
        self.animation_progress = 0.0
        
        self._generate_puzzle()

        # Calculate initial on-target count
        self.num_on_target_last_turn = self._count_on_target()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        movement = action[0]

        if self.game_state == 'IDLE' and not self.game_over:
            if movement != 0: # A push action is attempted
                self.moves_left -= 1
                reward -= 0.1
                # sound: "click"
                
                new_positions, has_changed = self._calculate_push(movement)

                if has_changed:
                    self.anim_start_pos = [b['pos'] for b in self.blocks]
                    self.anim_end_pos = new_positions
                    for i, block in enumerate(self.blocks):
                        block['pos'] = new_positions[i]
                    
                    self.game_state = 'ANIMATING'
                    self.animation_progress = 0.0
                else: # Move was attempted but all blocks were blocked
                    if self.moves_left <= 0:
                        self.game_over = True
                        self.win = False
                        reward -= 50
                        # sound: "lose_whistle"

        if self.game_state == 'ANIMATING':
            self.animation_progress += 1.0 / self.ANIMATION_FRAMES
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.game_state = 'IDLE'
                # sound: "thud"
                
                post_move_reward, is_win = self._check_puzzle_state()
                reward += post_move_reward
                
                if is_win:
                    self.game_over = True
                    self.win = True
                    reward += 50
                    # sound: "win_fanfare"
                elif self.moves_left <= 0:
                    self.game_over = True
                    self.win = False
                    reward -= 50
                    # sound: "lose_whistle"

        self.score += reward
        terminated = self.game_over
        self.clock.tick(30) # For auto_advance=True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _generate_puzzle(self):
        # 1. Get all possible grid cells
        all_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_coords)

        # 2. Assign targets and initial block positions (solved state)
        self.targets = []
        self.blocks = []
        target_coords = all_coords[:self.NUM_BLOCKS]
        
        for i in range(self.NUM_BLOCKS):
            pos = target_coords[i]
            color = self.BLOCK_COLORS[i]
            self.targets.append({'pos': list(pos), 'color': color})
            self.blocks.append({'pos': list(pos), 'color': color})

        # 3. Scramble the puzzle by applying random moves
        for _ in range(self.SCRAMBLE_MOVES):
            rand_move = self.np_random.integers(1, 5)
            new_positions, _ = self._calculate_push(rand_move)
            for i, block in enumerate(self.blocks):
                block['pos'] = new_positions[i]
    
    def _calculate_push(self, direction):
        if direction == 0:
            return [b['pos'] for b in self.blocks], False

        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        sorted_blocks = sorted(
            self.blocks, 
            key=lambda b: b['pos'][0] * dx + b['pos'][1] * dy, 
            reverse=True
        )
        
        occupied = {tuple(b['pos']) for b in self.blocks}
        new_positions_map = {}
        original_indices = {tuple(b['pos']): i for i, b in enumerate(self.blocks)}

        has_changed = False
        for block in sorted_blocks:
            orig_pos_tuple = tuple(block['pos'])
            
            occupied.remove(orig_pos_tuple)
            
            current_pos = list(orig_pos_tuple)
            while True:
                next_pos = [current_pos[0] + dx, current_pos[1] + dy]
                
                if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                    break
                
                if tuple(next_pos) in occupied:
                    break
                
                current_pos = next_pos
            
            occupied.add(tuple(current_pos))
            
            block_index = original_indices[orig_pos_tuple]
            new_positions_map[block_index] = current_pos
            
            if tuple(current_pos) != orig_pos_tuple:
                has_changed = True

        final_positions = [new_positions_map[i] for i in range(len(self.blocks))]
        return final_positions, has_changed

    def _count_on_target(self):
        count = 0
        for block in self.blocks:
            for target in self.targets:
                if block['color'] == target['color'] and block['pos'] == target['pos']:
                    count += 1
                    break
        return count

    def _check_puzzle_state(self):
        num_on_target = self._count_on_target()
        
        newly_placed = num_on_target - self.num_on_target_last_turn
        if newly_placed > 0:
            # sound: "place_block"
            pass
        self.num_on_target_last_turn = num_on_target
        
        reward = newly_placed * 1.0
        is_win = (num_on_target == self.NUM_BLOCKS)
        
        return reward, is_win

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_H * self.CELL_SIZE))
        for y in range(self.GRID_H + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_W * self.CELL_SIZE, py))

        # Draw targets
        for target in self.targets:
            tx, ty = target['pos']
            color = target['color']
            rect = pygame.Rect(
                self.GRID_X_OFFSET + tx * self.CELL_SIZE,
                self.GRID_Y_OFFSET + ty * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, color, rect, 3)

        # Draw blocks
        for i, block in enumerate(self.blocks):
            color = block['color']
            
            if self.game_state == 'ANIMATING':
                start_x, start_y = self.anim_start_pos[i]
                end_x, end_y = self.anim_end_pos[i]
                
                # Lerp
                prog = self.animation_progress
                curr_x = start_x + (end_x - start_x) * prog
                curr_y = start_y + (end_y - start_y) * prog
                
                px = self.GRID_X_OFFSET + curr_x * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + curr_y * self.CELL_SIZE
            else:
                bx, by = block['pos']
                px = self.GRID_X_OFFSET + bx * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + by * self.CELL_SIZE

            rect = pygame.Rect(int(px) + 3, int(py) + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.gfxdraw.rectangle(self.screen, rect, tuple(c//2 for c in color))

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves Left: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg = "PUZZLE SOLVED!"
                color = (100, 255, 100)
            else:
                msg = "OUT OF MOVES"
                color = (255, 100, 100)
                
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "blocks_on_target": self.num_on_target_last_turn,
        }

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game directly
    # It will not be executed by the evaluation system
    # but is useful for testing.
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Key mapping
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    
    running = True
    while running:
        action_movement = 0 # Default to no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action_movement = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q: # Quit on 'q'
                    running = False

        gym_action = [action_movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(gym_action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            pygame.time.wait(1500)
            obs, info = env.reset()

    env.close()