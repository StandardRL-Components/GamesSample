
# Generated: 2025-08-27T18:11:26.105502
# Source Brief: brief_01756.md
# Brief Index: 1756

        
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
        "Controls: Use arrow keys to push all blocks in a direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based puzzle game. Push colored blocks onto their matching "
        "target squares within the move limit. All blocks move together."
    )

    # Should frames auto-advance or wait for user input?
    # True for smooth animation, but logic remains turn-based.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_SIZE = 8
        self.NUM_BLOCKS = 5
        self.INITIAL_MOVES = 20
        self.SCRAMBLE_MOVES = 25
        self.ANIMATION_FRAMES = 8

        # Visuals
        self.FONT_MAIN = pygame.font.SysFont("monospace", 24, bold=True)
        self.FONT_MSG = pygame.font.SysFont("monospace", 48, bold=True)
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Rewards
        self.REWARD_PER_MOVE = -0.05
        self.REWARD_BLOCK_PLACED = 1.0
        self.REWARD_BLOCK_REMOVED = -0.2
        self.REWARD_WIN = 50.0
        self.REWARD_LOSE = -20.0
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.blocks = []
        self.targets = []
        self.is_animating = False
        self.animation_progress = 0.0
        self.game_over_message = ""

        # Calculate grid rendering properties
        self.grid_area_size = min(self.screen_width - 40, self.screen_height - 80)
        self.cell_size = self.grid_area_size // self.GRID_SIZE
        self.grid_render_size = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = (self.screen_width - self.grid_render_size) // 2
        self.grid_offset_y = (self.screen_height - self.grid_render_size) // 2 + 20

        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.INITIAL_MOVES
        self.is_animating = False
        self.animation_progress = 0.0
        self.game_over_message = ""
        
        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = self.game_over

        if self.is_animating:
            # Advance animation, no new logic
            self.animation_progress += 1.0 / self.ANIMATION_FRAMES
            if self.animation_progress >= 1.0:
                self.is_animating = False
                self.animation_progress = 0.0
                # Snap blocks to final logical positions
                for block in self.blocks:
                    block['pos'] = block['end_pos']
                
                # Check for win condition post-animation
                if self._check_win_condition():
                    self.game_over = True
                    terminated = True
                    reward += self.REWARD_WIN
                    self.game_over_message = "YOU WIN!"
                    # Sound placeholder: pygame.mixer.Sound.play(self.win_sound)
        
        elif not self.game_over:
            movement = action[0]  # 0-4: none/up/down/left/right
            
            if movement > 0: # Is a push action
                self.moves_remaining -= 1
                reward += self.REWARD_PER_MOVE
                # Sound placeholder: pygame.mixer.Sound.play(self.push_sound)

                blocks_on_target_before = self._count_blocks_on_target()
                moved = self._push_blocks(movement)
                
                if moved:
                    self.is_animating = True
                    self.animation_progress = 0.0
                    
                    blocks_on_target_after = self._count_blocks_on_target(check_end_pos=True)
                    
                    placed = max(0, blocks_on_target_after - blocks_on_target_before)
                    removed = max(0, blocks_on_target_before - blocks_on_target_after)
                    
                    if placed > 0:
                        # Sound placeholder: pygame.mixer.Sound.play(self.place_sound)
                        reward += placed * self.REWARD_BLOCK_PLACED
                    if removed > 0:
                        reward += removed * self.REWARD_BLOCK_REMOVED
                else: # Push resulted in no movement
                    # No animation, but move is still consumed
                    pass

            # Check for loss condition
            if self.moves_remaining <= 0 and not self._check_win_condition():
                self.game_over = True
                terminated = True
                reward += self.REWARD_LOSE
                self.game_over_message = "OUT OF MOVES"
                # Sound placeholder: pygame.mixer.Sound.play(self.lose_sound)

        self.steps += 1
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_level(self):
        self.targets = []
        self.blocks = []
        
        available_pos = list(np.ndindex((self.GRID_SIZE, self.GRID_SIZE)))
        self.np_random.shuffle(available_pos)

        # Place targets and blocks in solved state
        for i in range(self.NUM_BLOCKS):
            pos = available_pos.pop()
            self.targets.append({'pos': pos, 'color_idx': i})
            self.blocks.append({'pos': pos, 'color_idx': i})
        
        # Scramble the board by performing random pushes
        for _ in range(self.SCRAMBLE_MOVES):
            direction = self.np_random.integers(1, 5)
            self._push_blocks(direction, execute_move=True)

        # Finalize block state for the start of the game
        for block in self.blocks:
            block['start_pos'] = block['pos']
            block['end_pos'] = block['pos']

    def _push_blocks(self, direction, execute_move=False):
        # direction: 1=U, 2=D, 3=L, 4=R
        if direction == 1: # Up
            sort_order = sorted(self.blocks, key=lambda b: b['pos'][0])
            delta = (-1, 0)
        elif direction == 2: # Down
            sort_order = sorted(self.blocks, key=lambda b: b['pos'][0], reverse=True)
            delta = (1, 0)
        elif direction == 3: # Left
            sort_order = sorted(self.blocks, key=lambda b: b['pos'][1])
            delta = (0, -1)
        elif direction == 4: # Right
            sort_order = sorted(self.blocks, key=lambda b: b['pos'][1], reverse=True)
            delta = (0, 1)
        else:
            return False

        moved_any_block = False
        occupied = {tuple(b['pos']) for b in self.blocks}

        for block in sort_order:
            current_pos = block['pos']
            final_pos = current_pos
            
            while True:
                next_pos = (final_pos[0] + delta[0], final_pos[1] + delta[1])
                
                if not (0 <= next_pos[0] < self.GRID_SIZE and 0 <= next_pos[1] < self.GRID_SIZE):
                    break
                if next_pos in occupied:
                    break
                
                final_pos = next_pos
            
            if final_pos != current_pos:
                moved_any_block = True
                occupied.remove(current_pos)
                occupied.add(final_pos)
            
            if execute_move:
                block['pos'] = final_pos
            else:
                block['start_pos'] = block['pos']
                block['end_pos'] = final_pos

        return moved_any_block

    def _check_win_condition(self):
        if len(self.blocks) != len(self.targets): return False
        on_target_count = self._count_blocks_on_target()
        return on_target_count == self.NUM_BLOCKS

    def _count_blocks_on_target(self, check_end_pos=False):
        count = 0
        target_map = {t['pos']: t['color_idx'] for t in self.targets}
        for block in self.blocks:
            pos = block['end_pos'] if check_end_pos else block['pos']
            if pos in target_map and target_map[pos] == block['color_idx']:
                count += 1
        return count

    def _grid_to_pixel(self, grid_pos):
        r, c = grid_pos
        x = self.grid_offset_x + c * self.cell_size
        y = self.grid_offset_y + r * self.cell_size
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_render_size))
            # Horizontal
            start_y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_render_size, start_y))

        # Draw targets
        for target in self.targets:
            px, py = self._grid_to_pixel(target['pos'])
            color = self.BLOCK_COLORS[target['color_idx']]
            target_color = tuple(int(c * 0.4) for c in color)
            rect = (px, py, self.cell_size, self.cell_size)
            pygame.gfxdraw.box(self.screen, rect, target_color)
            pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GRID)

        # Draw blocks
        target_map = {t['pos']: t['color_idx'] for t in self.targets}
        for block in self.blocks:
            if self.is_animating:
                p = self.animation_progress
                p = p * p * (3 - 2 * p)  # Smoothstep easing
                start_x, start_y = self._grid_to_pixel(block['start_pos'])
                end_x, end_y = self._grid_to_pixel(block['end_pos'])
                vis_x = start_x + (end_x - start_x) * p
                vis_y = start_y + (end_y - start_y) * p
                current_pos_for_check = block['end_pos']
            else:
                vis_x, vis_y = self._grid_to_pixel(block['pos'])
                current_pos_for_check = block['pos']
            
            color = self.BLOCK_COLORS[block['color_idx']]
            rect = pygame.Rect(int(vis_x) + 2, int(vis_y) + 2, self.cell_size - 4, self.cell_size - 4)
            
            # Draw block with a subtle 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, shadow, rect.move(2, 2))
            pygame.draw.rect(self.screen, highlight, rect.move(-2, -2))
            pygame.draw.rect(self.screen, color, rect)

            # Draw checkmark if on correct target
            if current_pos_for_check in target_map and target_map[current_pos_for_check] == block['color_idx']:
                cx, cy = rect.center
                points = [(cx - 10, cy), (cx - 2, cy + 8), (cx + 10, cy - 8)]
                pygame.draw.lines(self.screen, (255, 255, 255), False, points, 3)

    def _render_ui(self):
        # Render moves remaining
        moves_text = self.FONT_MAIN.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Render score
        score_text = self.FONT_MAIN.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_color = self.COLOR_WIN if self._check_win_condition() else self.COLOR_LOSE
            msg_text = self.FONT_MSG.render(self.game_over_message, True, msg_color)
            msg_rect = msg_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "blocks_on_target": self._count_blocks_on_target()
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    action = np.array([0, 0, 0]) # No-op
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
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
            if event.type == pygame.KEYUP:
                 if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0

        # In auto_advance=True mode, we continuously step.
        # The logic inside step() handles turn-based actions correctly.
        obs, reward, terminated, truncated, info = env.step(action)

        # For human play, we only take an action once per key press
        if action[0] != 0:
            action[0] = 0

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    # Display final screen for a moment
    print(f"Game Over! Final Score: {info['score']:.2f}")
    pygame.time.wait(2000)
    
    env.close()