
# Generated: 2025-08-28T03:52:57.685656
# Source Brief: brief_02145.md
# Brief Index: 2145

        
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
        "Controls: Use arrow keys to highlight a block. Press space to select it. "
        "Use arrows again to push the selected block. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro puzzle game. Push pixel blocks on the grid to match the target pattern "
        "in the top-right corner before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.BLOCK_SIZE = 36
        self.GRID_LINE_WIDTH = 2
        
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HIGHLIGHT = (255, 255, 100)
        self.COLOR_SELECT = (100, 255, 150)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 150, 255), (255, 255, 80),
            (255, 80, 255), (80, 255, 255), (255, 150, 80), (150, 80, 255)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.level = 0
        self.max_moves = 0
        self.moves_left = 0
        self.grid = None
        self.target_grid = None
        self.movable_blocks = []
        self.target_positions = {}
        self.highlighted_idx = 0
        self.selected_idx = -1
        self.last_action_feedback = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Create a new generator if one doesn't exist
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.level = options.get("level", 1) if options else 1
        
        self._generate_level()
        
        self.highlighted_idx = 0
        self.selected_idx = -1
        self.last_action_feedback = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.last_action_feedback = None

        action_taken = False

        if shift_held and self.selected_idx != -1:
            self.selected_idx = -1
            action_taken = True
            # Small penalty for deselecting? Or neutral. Let's keep it neutral.
        elif space_held and self.selected_idx == -1 and self.movable_blocks:
            self.selected_idx = self.highlighted_idx
            action_taken = True
        elif movement != 0:
            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            direction = direction_map.get(movement)
            
            if direction:
                if self.selected_idx != -1:
                    # --- PUSH ACTION ---
                    action_taken = True
                    self.moves_left -= 1
                    
                    pre_push_state = self._get_block_state()
                    push_successful = self._execute_push(self.selected_idx, direction)
                    
                    if push_successful:
                        # sound: block_slide.wav
                        post_push_state = self._get_block_state()
                        reward += self._calculate_push_reward(pre_push_state, post_push_state)
                        self.last_action_feedback = ("push", self.movable_blocks[self.selected_idx][1])
                    else:
                        # sound: push_fail.wav
                        reward -= 0.1 # Small penalty for trying an invalid push
                        self.last_action_feedback = ("fail", self.movable_blocks[self.selected_idx][1])
                        
                elif self.movable_blocks:
                    # --- HIGHLIGHT CYCLE ACTION ---
                    action_taken = True
                    # This logic cycles based on direction, which is more intuitive
                    current_x, current_y = self.movable_blocks[self.highlighted_idx][1]
                    best_next_idx = -1
                    min_dist = float('inf')

                    for i, (_, (x, y)) in enumerate(self.movable_blocks):
                        if i == self.highlighted_idx: continue
                        dx, dy = x - current_x, y - current_y
                        dist = dx*dx + dy*dy
                        # Check if block is roughly in the chosen direction
                        if (direction == (0, -1) and dy < 0 and abs(dy) > abs(dx)) or \
                           (direction == (0, 1) and dy > 0 and abs(dy) > abs(dx)) or \
                           (direction == (-1, 0) and dx < 0 and abs(dx) > abs(dy)) or \
                           (direction == (1, 0) and dx > 0 and abs(dx) > abs(dy)):
                            if dist < min_dist:
                                min_dist = dist
                                best_next_idx = i
                    
                    if best_next_idx != -1:
                        self.highlighted_idx = best_next_idx
                    else: # Fallback to simple cycling if no block is in that direction
                        self.highlighted_idx = (self.highlighted_idx + (1 if movement in [2,4] else -1)) % len(self.movable_blocks)

        if not action_taken and movement == 0:
            reward -= 0.01 # Small penalty for no-op to encourage action

        self.steps += 1
        self.score += reward

        # --- Check Termination Conditions ---
        if np.array_equal(self.grid, self.target_grid):
            # sound: level_complete.wav
            reward += 100
            self.score += 100
            terminated = True
        elif self.moves_left <= 0:
            # sound: game_over.wav
            reward -= 50
            self.score -= 50
            terminated = True
        
        if self.steps >= 1000:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32)
        self.target_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32)
        
        num_blocks = min(self.GRID_WIDTH * self.GRID_HEIGHT - 2, 3 + (self.level - 1))
        self.max_moves = max(15, 40 - (self.level - 1))
        self.moves_left = self.max_moves

        possible_coords = [(y, x) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH)]
        self.np_random.shuffle(possible_coords)
        
        self.target_positions.clear()
        for i in range(num_blocks):
            y, x = possible_coords[i]
            block_id = i + 1
            self.target_grid[y, x] = block_id
            self.target_positions[block_id] = (x, y)

        # Create a solvable starting state by shuffling from the solved state
        self.grid = np.copy(self.target_grid)
        num_shuffles = min(20, 5 + self.level * 2)
        for _ in range(num_shuffles):
            temp_blocks = self._get_current_movable_blocks()
            if not temp_blocks: break
            
            block_to_shuffle_idx = self.np_random.integers(0, len(temp_blocks))
            shuffle_dir = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            self._execute_push(block_to_shuffle_idx, shuffle_dir, update_state=True)

        self.movable_blocks = self._get_current_movable_blocks()

    def _execute_push(self, block_idx, direction, update_state=False):
        if not self.movable_blocks: return False
        
        block_id, (start_x, start_y) = self.movable_blocks[block_idx]
        dx, dy = direction
        
        line_to_push = []
        cx, cy = start_x, start_y
        
        while 0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT:
            if self.grid[cy, cx] > 0:
                line_to_push.append((self.grid[cy, cx], (cx, cy)))
                cx += dx
                cy += dy
            else:
                break # Hit an empty space
        
        # Check if push is blocked by edge of grid
        if not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
            return False

        # Execute the push by moving blocks backwards
        self.grid[start_y, start_x] = 0
        for b_id, (x, y) in reversed(line_to_push):
            nx, ny = x + dx, y + dy
            self.grid[ny, nx] = b_id
        
        if update_state: # Used for shuffling
            self.movable_blocks = self._get_current_movable_blocks()
        else: # Normal gameplay push
            self.movable_blocks = self._get_current_movable_blocks()
            # Update selected_idx to follow the pushed block
            for i, (b_id, _) in enumerate(self.movable_blocks):
                if b_id == block_id:
                    self.selected_idx = i
                    self.highlighted_idx = i
                    break
        return True

    def _get_current_movable_blocks(self):
        blocks = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_id = self.grid[y, x]
                if block_id > 0:
                    blocks.append((block_id, (x, y)))
        blocks.sort() # Ensure consistent order
        return blocks

    def _get_block_state(self):
        state = {}
        for block_id, pos in self._get_current_movable_blocks():
            state[block_id] = pos
        return state

    def _calculate_push_reward(self, pre_state, post_state):
        reward = 0
        dist_change = 0
        
        all_block_ids = set(pre_state.keys()) | set(post_state.keys())

        for block_id in all_block_ids:
            if block_id in self.target_positions:
                target_x, target_y = self.target_positions[block_id]
                
                # Distance change reward
                if block_id in pre_state and block_id in post_state:
                    pre_x, pre_y = pre_state[block_id]
                    post_x, post_y = post_state[block_id]
                    pre_dist = abs(pre_x - target_x) + abs(pre_y - target_y)
                    post_dist = abs(post_x - target_x) + abs(post_y - target_y)
                    dist_change += (pre_dist - post_dist)

                # Correct placement event reward
                if block_id in post_state:
                    post_x, post_y = post_state[block_id]
                    if (post_x, post_y) == (target_x, target_y):
                        # Check if it wasn't already correct
                        if block_id not in pre_state or pre_state.get(block_id) != (target_x, target_y):
                            reward += 5
                            # sound: correct_place.wav
        
        # Scale distance change to be a smaller but significant reward
        reward += dist_change * 0.5
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT * self.BLOCK_SIZE), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE, y), self.GRID_LINE_WIDTH)

        # Draw blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_id = self.grid[y, x]
                if block_id > 0:
                    color = self.BLOCK_COLORS[(block_id - 1) % len(self.BLOCK_COLORS)]
                    rect = pygame.Rect(self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y + y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    darker_color = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, darker_color, rect, width=3, border_radius=4)

        # Draw highlight and selection
        if self.movable_blocks:
            if self.selected_idx != -1:
                _, (x, y) = self.movable_blocks[self.selected_idx]
                rect = pygame.Rect(self.GRID_X + x * self.BLOCK_SIZE - 2, self.GRID_Y + y * self.BLOCK_SIZE - 2, self.BLOCK_SIZE + 4, self.BLOCK_SIZE + 4)
                pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, width=3, border_radius=6)
            elif self.highlighted_idx < len(self.movable_blocks):
                _, (x, y) = self.movable_blocks[self.highlighted_idx]
                rect = pygame.Rect(self.GRID_X + x * self.BLOCK_SIZE - 2, self.GRID_Y + y * self.BLOCK_SIZE - 2, self.BLOCK_SIZE + 4, self.BLOCK_SIZE + 4)
                pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, width=3, border_radius=6)

    def _render_ui(self):
        # --- Render Text ---
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 20, 20))

        # --- Render Target Preview ---
        preview_size = 12
        preview_x = self.SCREEN_WIDTH - (self.GRID_WIDTH * preview_size) - 20
        preview_y = 50
        
        target_label = self.font_small.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_label, (preview_x, preview_y))
        preview_y += 20

        preview_bg_rect = pygame.Rect(preview_x - 2, preview_y - 2, self.GRID_WIDTH * preview_size + 4, self.GRID_HEIGHT * preview_size + 4)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_bg_rect, border_radius=3)
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_id = self.target_grid[y, x]
                if block_id > 0:
                    color = self.BLOCK_COLORS[(block_id - 1) % len(self.BLOCK_COLORS)]
                    rect = pygame.Rect(preview_x + x * preview_size, preview_y + y * preview_size, preview_size - 1, preview_size - 1)
                    pygame.draw.rect(self.screen, color, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Pusher")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing or resetting
            pygame.time.wait(2000)

        # Since auto_advance is False, we need to control the loop speed
        env.clock.tick(10) # Limit to 10 actions per second for human play
        
    env.close()