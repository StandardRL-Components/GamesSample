
# Generated: 2025-08-28T03:57:34.644621
# Source Brief: brief_05098.md
# Brief Index: 5098

        
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
        "Controls: Use Space to cycle selection. Use ↑↓←→ to push the selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks onto their matching targets. Solve the puzzle in under 50 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 14
    GRID_HEIGHT = 9
    CELL_SIZE = 40
    NUM_BLOCKS = 10
    MAX_MOVES = 50
    ANIMATION_STEPS = 8

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (45, 45, 55)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SELECTION = (255, 255, 0)
    
    BLOCK_COLORS = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
        (250, 190, 212), (0, 128, 128)
    ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self.blocks = []
        self.moves_made = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.selected_block_idx = 0
        self.last_space_state = 0
        self.animation_state = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_made = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.selected_block_idx = 0
        self.last_space_state = 0
        self.animation_state = None
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.blocks = []
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        target_coords = all_coords[:self.NUM_BLOCKS]
        block_start_coords = all_coords[self.NUM_BLOCKS:self.NUM_BLOCKS * 2]

        for i in range(self.NUM_BLOCKS):
            self.blocks.append({
                'id': i,
                'color': self.BLOCK_COLORS[i],
                'pos': block_start_coords[i],
                'target': target_coords[i],
                'on_target': False
            })
        
        # Scramble puzzle by starting from a solved state and moving blocks randomly
        # This guarantees solvability.
        available_coords = set(all_coords)
        for block in self.blocks:
            available_coords.remove(block['target'])
        
        # Place blocks on targets
        for block in self.blocks:
             block['pos'] = block['target']
             block['on_target'] = True

        # Shuffle blocks
        for _ in range(50): # Number of shuffle moves
            block_to_move_idx = self.np_random.integers(0, self.NUM_BLOCKS)
            block_to_move = self.blocks[block_to_move_idx]
            
            # Find empty adjacent cells
            x, y = block_to_move['pos']
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    is_occupied = any(b['pos'] == (nx, ny) for b in self.blocks)
                    if not is_occupied:
                        possible_moves.append((nx, ny))
            
            if possible_moves:
                new_pos = self.np_random.choice(possible_moves, 1)[0]
                block_to_move['pos'] = (new_pos[0], new_pos[1])

        # Recalculate initial on_target status after shuffling
        for block in self.blocks:
            block['on_target'] = (block['pos'] == block['target'])


    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle ongoing animations
        if self.animation_state:
            self.animation_state['progress'] += 1
            if self.animation_state['progress'] >= self.ANIMATION_STEPS:
                # Animation finished
                block = self.blocks[self.animation_state['block_idx']]
                block['pos'] = self.animation_state['end_pos']
                self.animation_state = None

                # Check if it landed on a target
                was_on_target = block['on_target']
                is_on_target = block['pos'] == block['target']
                block['on_target'] = is_on_target

                if is_on_target and not was_on_target:
                    reward += 1.0  # Reward for placing a block
                    # sfx: block_placed.wav

                # Check for win condition
                if all(b['on_target'] for b in self.blocks):
                    self.win_state = True
                    self.game_over = True
                    terminated = True
                    reward += 50.0 # Win bonus
                    # sfx: win_jingle.wav
            
            self.score += reward
            return self._get_observation(), reward, terminated, False, self._get_info()

        # 2. Process new action if not animating
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle selection cycling (on space press)
        if space_held and not self.last_space_state:
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            # sfx: select_tick.wav
        self.last_space_state = space_held

        # Handle block push
        if movement > 0:
            self.moves_made += 1
            reward -= 0.1  # Cost per move

            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = direction_map[movement]

            selected_block = self.blocks[self.selected_block_idx]
            start_pos = selected_block['pos']
            current_pos = start_pos

            # Determine push destination
            while True:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    break # Hit a wall
                
                is_occupied = any(b['pos'] == next_pos for b in self.blocks)
                if is_occupied:
                    break # Hit another block

                current_pos = next_pos
            
            end_pos = current_pos

            if start_pos != end_pos:
                # Start animation
                self.animation_state = {
                    'block_idx': self.selected_block_idx,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'progress': 0
                }
                # sfx: block_slide.wav

        # Check for loss condition
        if self.moves_made >= self.MAX_MOVES and not self.game_over:
            self.game_over = True
            terminated = True
            reward -= 50.0 # Loss penalty
            # sfx: lose_sound.wav
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.CELL_SIZE
        y = self.grid_offset_y + grid_pos[1] * self.CELL_SIZE
        return x, y
    
    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_offset_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_offset_y), (px, self.grid_offset_y + self.grid_pixel_height))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_offset_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, py), (self.grid_offset_x + self.grid_pixel_width, py))

        # Draw targets
        for block in self.blocks:
            px, py = self._grid_to_pixel(block['target'])
            center_x, center_y = px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 3
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, block['color'])
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*block['color'][:3], 50))


        # Draw blocks
        for i, block in enumerate(self.blocks):
            if self.animation_state and self.animation_state['block_idx'] == i:
                # Interpolate position for animation
                anim = self.animation_state
                start_px, start_py = self._grid_to_pixel(anim['start_pos'])
                end_px, end_py = self._grid_to_pixel(anim['end_pos'])
                t = anim['progress'] / self.ANIMATION_STEPS
                
                # Ease-out curve for smoother stop
                t = 1 - (1 - t)**3
                
                px = self._lerp(start_px, end_px, t)
                py = self._lerp(start_py, end_py, t)
            else:
                px, py = self._grid_to_pixel(block['pos'])

            rect = pygame.Rect(int(px) + 3, int(py) + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            
            # Draw selection highlight
            if i == self.selected_block_idx and not self.game_over:
                highlight_rect = rect.inflate(8, 8)
                pygame.draw.rect(self.screen, self.COLOR_SELECTION, highlight_rect, border_radius=8)
                highlight_rect2 = rect.inflate(4, 4)
                pygame.draw.rect(self.screen, (*self.COLOR_SELECTION, 150), highlight_rect2, border_radius=6)

            pygame.draw.rect(self.screen, block['color'], rect, border_radius=6)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), rect, width=2, border_radius=6)

    def _render_ui(self):
        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.moves_made}/{self.MAX_MOVES}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Score display
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                end_text = self.font_large.render("PUZZLE SOLVED!", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("OUT OF MOVES", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_made": self.moves_made,
            "blocks_on_target": sum(1 for b in self.blocks if b['on_target'])
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
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # 0=none
        space_press = 0 # 0=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_press = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # If a move key is pressed, we only need one step for that action
        if movement > 0 or space_press > 0:
             action = [movement, space_press, 0]
             obs, reward, terminated, truncated, info = env.step(action)
        else:
             # If no key is pressed, we still step with a no-op to advance animation
             obs, reward, terminated, truncated, info = env.step([0, 0, 0])

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Moves: {info['moves_made']}")
            # The env will now be in a 'game_over' state. We can just watch the end screen.
            # To play again, the user should press 'r'.

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()