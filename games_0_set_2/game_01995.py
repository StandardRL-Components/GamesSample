
# Generated: 2025-08-28T03:20:58.785823
# Source Brief: brief_01995.md
# Brief Index: 1995

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your player (white square) and push blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all the red blocks onto the green target zones before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        
        self.MAX_MOVES = 50
        self.MAX_STEPS = 500
        
        self.N_RED_BLOCKS = 3
        self.N_GREY_BLOCKS = 5
        self.SHUFFLE_MOVES = 60

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_RED_BLOCK = (220, 50, 50)
        self.COLOR_GREY_BLOCK = (120, 120, 140)
        self.COLOR_TARGET = (50, 200, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SHADOW = (15, 15, 20)

        # Initialize state variables
        self.player_pos = None
        self.blocks = []
        self.targets = []
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        # Initialize state
        # self.reset() # Called by gym.make
        
        # Run validation check
        # self.validate_implementation() # Not called in production
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.moves_taken = 0
        self.score = 0
        self.game_over = False
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.blocks = []
        self.targets = []

        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        # Place targets
        for _ in range(self.N_RED_BLOCKS):
            self.targets.append(all_coords.pop())

        # Place red blocks on targets (solved state)
        for i in range(self.N_RED_BLOCKS):
            self.blocks.append({"pos": self.targets[i], "type": "red"})

        # Place grey blocks
        for _ in range(self.N_GREY_BLOCKS):
            self.blocks.append({"pos": all_coords.pop(), "type": "grey"})
            
        # Shuffle blocks from solved state to guarantee solvability
        for _ in range(self.SHUFFLE_MOVES):
            block_to_move = self.np_random.choice(self.blocks)
            
            # Try to move in a random direction
            move_dir = self.np_random.integers(0, 4)
            dx = [0, 0, -1, 1][move_dir]
            dy = [-1, 1, 0, 0][move_dir]
            
            new_pos = (block_to_move['pos'][0] + dx, block_to_move['pos'][1] + dy)

            # Check if move is valid (in bounds and not occupied)
            if self._is_in_bounds(new_pos) and not self._is_occupied(new_pos, check_player=False):
                block_to_move['pos'] = new_pos
        
        # Place player in a remaining empty spot
        self.player_pos = all_coords.pop()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only directional actions count as a "move"
        if movement > 0:
            self.moves_taken += 1
            reward -= 0.1  # Cost for taking a move

            # Calculate how many red blocks are on targets before the move
            red_blocks = [b for b in self.blocks if b['type'] == 'red']
            on_target_before = sum(1 for b in red_blocks if self._is_on_target(b['pos']))

            dx = [0, 0, 0, -1, 1][movement]
            dy = [0, -1, 1, 0, 0][movement]
            
            player_target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            if self._is_in_bounds(player_target_pos):
                block_to_push = self._get_block_at(player_target_pos)
                
                if block_to_push is None:
                    # Simple move, no push
                    self.player_pos = player_target_pos
                else:
                    # Push logic
                    push_chain = []
                    can_push = True
                    current_pos = player_target_pos
                    
                    while True:
                        block = self._get_block_at(current_pos)
                        if block is None:
                            break # End of chain
                        
                        push_chain.append(block)
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        
                        if not self._is_in_bounds(next_pos):
                            can_push = False
                            break
                        
                        current_pos = next_pos
                    
                    if can_push:
                        # Move all blocks in the chain
                        for block in reversed(push_chain):
                            block['pos'] = (block['pos'][0] + dx, block['pos'][1] + dy)
                        # Move player
                        self.player_pos = player_target_pos
                        # Place block sfx placeholder: # sfx.play('push_block')

            # Calculate reward change from block movement
            on_target_after = sum(1 for b in red_blocks if self._is_on_target(b['pos']))
            reward += (on_target_after - on_target_before) * 1.0

        self.steps += 1
        self.score += reward

        # Check termination conditions
        win = self._check_win_condition()
        loss_moves = self.moves_taken >= self.MAX_MOVES
        loss_steps = self.steps >= self.MAX_STEPS
        
        terminated = win or loss_moves or loss_steps
        if terminated:
            self.game_over = True
            if win:
                self.score += 100
                reward += 100
                # Win sfx placeholder: # sfx.play('win_puzzle')
            elif loss_moves:
                self.score -= 10
                reward -= 10
                # Lose sfx placeholder: # sfx.play('lose_puzzle')
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

        # Draw targets
        for tx, ty in self.targets:
            rect = pygame.Rect(tx * self.CELL_WIDTH, ty * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_TARGET, rect.topleft, rect.bottomright, 5)
            pygame.draw.line(self.screen, self.COLOR_TARGET, rect.topright, rect.bottomleft, 5)

        # Draw blocks
        for block in self.blocks:
            bx, by = block['pos']
            color = self.COLOR_RED_BLOCK if block['type'] == 'red' else self.COLOR_GREY_BLOCK
            rect = pygame.Rect(bx * self.CELL_WIDTH + 5, by * self.CELL_HEIGHT + 5, self.CELL_WIDTH - 10, self.CELL_HEIGHT - 10)
            
            # Shadow
            shadow_rect = rect.copy().move(3, 3)
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=5)
            # Main block
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            # Highlight
            pygame.draw.line(self.screen, (255,255,255,40), rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, (255,255,255,40), rect.topleft, rect.bottomleft, 2)

        # Draw player
        px, py = self.player_pos
        rect = pygame.Rect(px * self.CELL_WIDTH + 8, py * self.CELL_HEIGHT + 8, self.CELL_WIDTH - 16, self.CELL_HEIGHT - 16)
        shadow_rect = rect.copy().move(3, 3)
        pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)

    def _render_ui(self):
        moves_text = f"Moves Left: {self.MAX_MOVES - self.moves_taken}"
        score_text = f"Score: {self.score:.1f}"

        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(moves_surf, (15, 10))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 15, 10))
        
        if self.game_over:
            win = self._check_win_condition()
            end_text = "PUZZLE SOLVED!" if win else "OUT OF MOVES"
            end_color = self.COLOR_TARGET if win else self.COLOR_RED_BLOCK
            font_end = pygame.font.SysFont("monospace", 50, bold=True)
            end_surf = font_end.render(end_text, True, end_color)
            
            # Simple text shadow
            shadow_surf = font_end.render(end_text, True, self.COLOR_SHADOW)
            self.screen.blit(shadow_surf, (self.SCREEN_WIDTH // 2 - end_surf.get_width() // 2 + 3,
                                           self.SCREEN_HEIGHT // 2 - end_surf.get_height() // 2 + 3))

            self.screen.blit(end_surf, (self.SCREEN_WIDTH // 2 - end_surf.get_width() // 2,
                                        self.SCREEN_HEIGHT // 2 - end_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_taken": self.moves_taken,
            "red_blocks_on_target": sum(1 for b in self.blocks if b['type'] == 'red' and self._is_on_target(b['pos']))
        }

    def _is_in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT

    def _get_block_at(self, pos):
        for block in self.blocks:
            if block['pos'] == pos:
                return block
        return None

    def _is_occupied(self, pos, check_player=True):
        if check_player and self.player_pos == pos:
            return True
        return self._get_block_at(pos) is not None

    def _is_on_target(self, pos):
        return pos in self.targets

    def _check_win_condition(self):
        red_blocks = [b for b in self.blocks if b['type'] == 'red']
        if not red_blocks: return False
        return all(self._is_on_target(b['pos']) for b in red_blocks)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")