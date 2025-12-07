
# Generated: 2025-08-28T04:24:26.405507
# Source Brief: brief_05236.md
# Brief Index: 5236

        
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

    # Short, user-facing control string:
    user_guide = "Controls: Use arrow keys to push all blocks simultaneously. No-op, space, and shift have no effect."

    # Short, user-facing description of the game:
    game_description = "A minimalist puzzle game. Push colored blocks to their matching goal zones before you run out of moves. Plan your pushes carefully, as every block moves at once."

    # Frames only advance when an action is received.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.CELL_SIZE = 40
        self.NUM_BLOCKS = 3
        self.MAX_MOVES = 20
        self.SCRAMBLE_MOVES_MIN = 6
        self.SCRAMBLE_MOVES_MAX = 12

        self.GRID_PIXEL_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_PIXEL_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_PIXEL_HEIGHT) // 2
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (45, 48, 52)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (15, 18, 22, 200)

        self.BLOCK_COLORS = [
            (255, 70, 70),   # Red
            (70, 150, 255),  # Blue
            (80, 220, 100),  # Green
            (255, 200, 50)   # Yellow
        ]
        self.GOAL_COLORS = [
            (80, 25, 25),
            (25, 50, 80),
            (30, 75, 35),
            (80, 65, 20)
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)
        
        # --- State Initialization ---
        self.blocks = []
        self.goals = []
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_push_direction = None
        self.last_positions = {}
        
        self.reset()

        # --- Implementation Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_push_direction = None
        
        self._generate_level()
        self.last_positions = {b['id']: b['pos'] for b in self.blocks}
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.blocks = []
        self.goals = []
        
        possible_cells = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        
        # Ensure we don't try to pick more cells than available
        num_to_generate = min(self.NUM_BLOCKS, len(possible_cells))
        
        # Use generator for random choices
        chosen_indices = self.np_random.choice(len(possible_cells), size=num_to_generate, replace=False)
        goal_positions = [possible_cells[i] for i in chosen_indices]

        # Create goals and corresponding solved blocks
        for i in range(num_to_generate):
            color_idx = i % len(self.BLOCK_COLORS)
            pos = goal_positions[i]
            
            self.goals.append({
                'id': i,
                'pos': pos,
                'color': self.GOAL_COLORS[color_idx]
            })
            self.blocks.append({
                'id': i,
                'pos': pos,
                'color': self.BLOCK_COLORS[color_idx]
            })
            
        # Scramble the board by applying random pushes
        num_scrambles = self.np_random.integers(self.SCRAMBLE_MOVES_MIN, self.SCRAMBLE_MOVES_MAX + 1)
        for _ in range(num_scrambles):
            direction = self.np_random.integers(1, 5) # 1-4 for push directions
            self._apply_push(direction)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.last_push_direction = None
        
        # Only process push actions
        if movement in [1, 2, 3, 4]: # 1:Up, 2:Down, 3:Left, 4:Right
            self.moves_left -= 1
            self.last_push_direction = movement
            
            old_positions = {b['id']: b['pos'] for b in self.blocks}
            
            self._apply_push(movement) # Sound: block_slide.wav
            
            new_positions = {b['id']: b['pos'] for b in self.blocks}
            
            reward += self._calculate_reward(old_positions, new_positions)
            self.last_positions = old_positions
        
        self.score += reward
        
        terminated, win_reward, loss_penalty = self._check_termination()
        reward += win_reward + loss_penalty
        self.score += win_reward + loss_penalty
        
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_push(self, direction):
        # Determine sorting order based on push direction
        if direction == 1: # Up
            sort_key = lambda b: b['pos'][1]
            sort_reverse = False
        elif direction == 2: # Down
            sort_key = lambda b: b['pos'][1]
            sort_reverse = True
        elif direction == 3: # Left
            sort_key = lambda b: b['pos'][0]
            sort_reverse = False
        else: # Right (4)
            sort_key = lambda b: b['pos'][0]
            sort_reverse = True
            
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=sort_reverse)
        
        for block in sorted_blocks:
            while True:
                current_pos = block['pos']
                next_pos = list(current_pos)
                
                if direction == 1: next_pos[1] -= 1
                elif direction == 2: next_pos[1] += 1
                elif direction == 3: next_pos[0] -= 1
                else: next_pos[0] += 1
                
                # Check boundaries
                if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS):
                    break
                
                # Check for collision with other blocks
                if self._is_cell_occupied(tuple(next_pos)):
                    break
                
                block['pos'] = tuple(next_pos)

    def _calculate_reward(self, old_positions, new_positions):
        reward = 0
        for block in self.blocks:
            goal = self.goals[block['id']]
            old_dist = self._manhattan_distance(old_positions[block['id']], goal['pos'])
            new_dist = self._manhattan_distance(new_positions[block['id']], goal['pos'])
            
            # Distance-based reward
            reward += (old_dist - new_dist)
            
            # Event-based reward for reaching goal
            if new_dist == 0 and old_dist > 0:
                reward += 5 # Sound: goal_reached.wav
        return reward

    def _check_termination(self):
        win_reward = 0
        loss_penalty = 0
        
        # Win condition: all blocks on their goals
        on_goal_count = sum(1 for b in self.blocks if b['pos'] == self.goals[b['id']]['pos'])
        is_win = on_goal_count == len(self.blocks)
        
        if is_win:
            self.win_state = True
            win_reward = 100 # Sound: level_complete.wav
            return True, win_reward, loss_penalty
            
        # Loss condition: out of moves
        if self.moves_left <= 0:
            loss_penalty = -100 # Sound: game_over.wav
            return True, win_reward, loss_penalty
            
        return False, win_reward, loss_penalty

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_PIXEL_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, py))
            
        # Draw goals
        for goal in self.goals:
            rect = self._grid_to_pixel_rect(goal['pos'])
            pygame.draw.rect(self.screen, goal['color'], rect)

        # Draw blocks with push effect
        for block in self.blocks:
            # Draw motion trail if pushed
            if self.last_push_direction and self.last_positions:
                old_pos = self.last_positions.get(block['id'])
                if old_pos and old_pos != block['pos']:
                    start_rect = self._grid_to_pixel_rect(old_pos)
                    end_rect = self._grid_to_pixel_rect(block['pos'])
                    
                    num_trails = 4
                    for i in range(1, num_trails + 1):
                        alpha = 100 - (i * 20)
                        trail_color = block['color'] + (alpha,)
                        
                        interp_x = start_rect.x + (end_rect.x - start_rect.x) * (i / num_trails)
                        interp_y = start_rect.y + (end_rect.y - start_rect.y) * (i / num_trails)
                        
                        trail_surface = pygame.Surface(start_rect.size, pygame.SRCALPHA)
                        pygame.draw.rect(trail_surface, trail_color, (0, 0, start_rect.width, start_rect.height), border_radius=4)
                        self.screen.blit(trail_surface, (interp_x, interp_y))

            # Draw block
            rect = self._grid_to_pixel_rect(block['pos'])
            border_size = 3
            
            # Outer, slightly darker part for depth
            pygame.draw.rect(self.screen, tuple(max(0, c-40) for c in block['color']), rect, border_radius=6)
            # Main color
            inner_rect = rect.inflate(-border_size*2, -border_size*2)
            pygame.draw.rect(self.screen, block['color'], inner_rect, border_radius=4)

    def _render_ui(self):
        # Panel for UI text
        ui_panel = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 12))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(right=self.WIDTH - 15, top=12)
        self.screen.blit(score_text, score_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            
            if self.win_state:
                msg = "LEVEL COMPLETE!"
                color = self.BLOCK_COLORS[2] # Green
            else:
                msg = "OUT OF MOVES"
                color = self.BLOCK_COLORS[0] # Red
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            overlay.blit(end_text, text_rect)
            self.screen.blit(overlay, (0, 0))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.MAX_MOVES - self.moves_left,
            "moves_left": self.moves_left,
            "win": self.win_state,
        }

    # --- Helper Methods ---
    def _grid_to_pixel_rect(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)

    def _is_cell_occupied(self, grid_pos):
        return any(b['pos'] == grid_pos for b in self.blocks)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # Set up a window to visualize the rgb_array output
    pygame.display.set_caption("Sokoban Push Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
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
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    done = False
                
                # If a push key was pressed or reset was hit
                if action[0] != 0 or event.key == pygame.K_r:
                    if not done:
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated
                        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                    
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            # Wait a bit on the end screen before allowing a reset
            pygame.time.wait(1000)

    env.close()