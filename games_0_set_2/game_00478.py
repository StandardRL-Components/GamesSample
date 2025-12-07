
# Generated: 2025-08-27T13:45:56.225081
# Source Brief: brief_00478.md
# Brief Index: 478

        
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
        "Controls: Use arrow keys to push all blocks simultaneously. Solve the puzzle before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all colored blocks to their matching goal zones. Each push counts against your move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 10
        self.GRID_ROWS = 6
        self.GRID_MARGIN_X = (self.SCREEN_WIDTH - self.GRID_COLS * 50) // 2
        self.GRID_MARGIN_Y = (self.SCREEN_HEIGHT - self.GRID_ROWS * 50) // 2
        self.CELL_SIZE = 50
        self.MAX_STEPS = 1000
        self.NUM_BLOCKS = 4
        self.INITIAL_MOVES_FACTOR = 2.0

        # Colors
        self.COLOR_BG = (44, 62, 80) # Dark blue-gray
        self.COLOR_GRID = (52, 73, 94) # Slightly lighter blue-gray
        self.COLOR_TEXT = (236, 240, 241) # White
        self.BLOCK_GOAL_COLORS = [
            ((231, 76, 60), (192, 57, 43)),   # Red
            ((46, 204, 113), (39, 174, 96)),  # Green
            ((52, 152, 219), (41, 128, 185)), # Blue
            ((241, 196, 15), (243, 156, 18)), # Yellow
        ]

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
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Game state variables
        self.blocks = []
        self.goals = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = 0
        self.max_moves = 0
        self.win_message = ""
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def _generate_puzzle(self):
        # Generate a solvable puzzle by starting with the solution and making random moves
        all_pos = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_pos)
        
        goal_positions = [all_pos.pop() for _ in range(self.NUM_BLOCKS)]
        self.goals = [{'pos': pos, 'color_idx': i} for i, pos in enumerate(goal_positions)]
        
        # Start with blocks on goals
        block_positions = list(goal_positions)
        
        # Perform random moves to scramble the puzzle
        scramble_moves = self.np_random.integers(8, 15)
        for _ in range(scramble_moves):
            direction = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            self._apply_push_logic(block_positions, direction)

        self.blocks = []
        for i, pos in enumerate(block_positions):
            self.blocks.append({
                'pos': list(pos), 
                'start_pos': list(pos), 
                'color_idx': i
            })
        
        self.max_moves = int(scramble_moves * self.INITIAL_MOVES_FACTOR)
        self.remaining_moves = self.max_moves

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        
        # Any action, even a no-op, consumes a move.
        self.remaining_moves -= 1
        
        # Store pre-move state for reward calculation
        pre_move_distances = self._get_total_distance_to_goals()
        pre_move_on_goal_state = {i: self._is_on_goal(block) for i, block in enumerate(self.blocks)}

        if movement > 0: # 1-4
            # # sfx: block_push.wav
            current_positions = [tuple(b['pos']) for b in self.blocks]
            new_positions = self._apply_push_logic(current_positions, movement)
            for i, block in enumerate(self.blocks):
                block['start_pos'] = list(block['pos'])
                block['pos'] = list(new_positions[i])
        else: # no-op
            # # sfx: error_or_wasted_turn.wav
            for block in self.blocks:
                block['start_pos'] = list(block['pos']) # No change

        # Calculate rewards
        post_move_distances = self._get_total_distance_to_goals()
        distance_change = pre_move_distances - post_move_distances
        reward += distance_change # +1 for each cell closer, -1 for each cell further

        for i, block in enumerate(self.blocks):
            is_now_on_goal = self._is_on_goal(block)
            was_on_goal = pre_move_on_goal_state[i]
            if is_now_on_goal and not was_on_goal:
                reward += 10 # Event-based reward for placing a block
                # # sfx: goal_achieved.wav

        self.score += reward
        
        terminated, terminal_reward = self._check_termination()
        self.score += terminal_reward
        reward += terminal_reward
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_push_logic(self, positions, direction):
        # direction: 1=up, 2=down, 3=left, 4=right
        if direction == 1: # Up
            sort_order = sorted(range(self.NUM_BLOCKS), key=lambda k: positions[k][1])
            dx, dy = 0, -1
        elif direction == 2: # Down
            sort_order = sorted(range(self.NUM_BLOCKS), key=lambda k: positions[k][1], reverse=True)
            dx, dy = 0, 1
        elif direction == 3: # Left
            sort_order = sorted(range(self.NUM_BLOCKS), key=lambda k: positions[k][0])
            dx, dy = -1, 0
        else: # Right
            sort_order = sorted(range(self.NUM_BLOCKS), key=lambda k: positions[k][0], reverse=True)
            dx, dy = 1, 0

        new_positions = list(positions)
        for i in sort_order:
            current_pos = new_positions[i]
            target_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check for wall collision
            if not (0 <= target_pos[0] < self.GRID_COLS and 0 <= target_pos[1] < self.GRID_ROWS):
                continue
            
            # Check for block collision
            if target_pos in new_positions:
                continue
            
            new_positions[i] = target_pos
        return new_positions

    def _get_total_distance_to_goals(self):
        total_dist = 0
        for block in self.blocks:
            goal_pos = self.goals[block['color_idx']]['pos']
            dist = abs(block['pos'][0] - goal_pos[0]) + abs(block['pos'][1] - goal_pos[1])
            total_dist += dist
        return total_dist

    def _is_on_goal(self, block):
        goal = self.goals[block['color_idx']]
        return tuple(block['pos']) == goal['pos']

    def _check_termination(self):
        is_win = all(self._is_on_goal(b) for b in self.blocks)
        if is_win:
            self.game_over = True
            self.win_message = "PUZZLE SOLVED!"
            # # sfx: victory_fanfare.wav
            return True, 50.0

        if self.remaining_moves <= 0:
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            # # sfx: failure_sound.wav
            return True, -50.0
        
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixels(self, pos):
        x = self.GRID_MARGIN_X + pos[0] * self.CELL_SIZE
        y = self.GRID_MARGIN_Y + pos[1] * self.CELL_SIZE
        return x, y

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_MARGIN_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_ROWS * self.CELL_SIZE), 2)
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_MARGIN_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_COLS * self.CELL_SIZE, py), 2)

        # Draw goals
        for goal in self.goals:
            px, py = self._grid_to_pixels(goal['pos'])
            color = self.BLOCK_GOAL_COLORS[goal['color_idx']][0]
            rect = pygame.Rect(px + 5, py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            pygame.draw.rect(self.screen, color, rect, 4, border_radius=8)

        # Draw block trails and blocks
        for block in self.blocks:
            start_px, start_py = self._grid_to_pixels(block['start_pos'])
            end_px, end_py = self._grid_to_pixels(block['pos'])
            
            # Draw trail if moved
            if (start_px, start_py) != (end_px, end_py):
                trail_color = list(self.BLOCK_GOAL_COLORS[block['color_idx']][0])
                trail_color.append(50) # Add alpha
                trail_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                trail_rect = pygame.Rect(self.CELL_SIZE * 0.2, self.CELL_SIZE * 0.2, self.CELL_SIZE * 0.6, self.CELL_SIZE * 0.6)
                pygame.draw.rect(trail_surf, trail_color, trail_rect, 0, border_radius=10)
                self.screen.blit(trail_surf, (start_px, start_py))

            # Draw block
            bright_color, dark_color = self.BLOCK_GOAL_COLORS[block['color_idx']]
            rect = pygame.Rect(end_px + 5, end_py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            pygame.draw.rect(self.screen, dark_color, rect.move(0, 3), 0, border_radius=12)
            pygame.draw.rect(self.screen, bright_color, rect, 0, border_radius=12)

    def _render_ui(self):
        # Render Moves
        moves_text = self.font_medium.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Render Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_text_render = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            win_rect = win_text_render.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(win_text_render, win_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "is_success": all(self._is_on_goal(b) for b in self.blocks)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate pygame screen for display
    pygame.display.set_caption("Push Block Puzzle")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    while running:
        # Convert observation back to a surface for display
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if terminated:
            # Wait for a key press to reset
            print(f"Game Over. Final Score: {info['score']}. Success: {info['is_success']}")
            wait_for_key = True
            while wait_for_key:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        wait_for_key = False
                    if event.type == pygame.KEYDOWN:
                        obs, info = env.reset()
                        terminated = False
                        wait_for_key = False
            continue

        action = [0, 0, 0] # Default action is no-op
        
        # Simple event handling for manual play
        event_processed = False
        while not event_processed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    event_processed = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    # Allow quitting with escape
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    
                    # Any keydown event triggers a step
                    event_processed = True
            
            if not running:
                break
        
        if not running:
            break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
    env.close()