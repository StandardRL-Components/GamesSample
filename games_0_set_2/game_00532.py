
# Generated: 2025-08-27T13:56:22.183126
# Source Brief: brief_00532.md
# Brief Index: 532

        
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


# Helper class for game objects
class GameObject:
    """A simple data class for blocks and targets."""
    def __init__(self, x, y, color):
        self.grid_x = x
        self.grid_y = y
        self.prev_grid_x = x
        self.prev_grid_y = y
        self.color = color
        self.on_target = False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all blocks simultaneously."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks onto their matching targets in as few moves as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 9
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        
        # Colors (A clean, high-contrast palette)
        self.COLOR_BG = (36, 39, 58)
        self.COLOR_GRID = (69, 73, 94)
        self.COLOR_TEXT = (202, 211, 245)
        self.BLOCK_COLORS = [
            (237, 135, 150), # Red
            (166, 218, 149), # Green
            (138, 173, 244), # Blue
            (245, 224, 179), # Yellow
            (183, 189, 248), # Lavender
        ]
        self.NUM_BLOCKS = 4
        self.MOVE_LIMIT = 25
        self.MAX_STEPS = 1000

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.moves_left = 0
        self.game_over = False
        self.blocks = []
        self.targets = []
        self.last_push_direction = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.moves_left = self.MOVE_LIMIT
        self.game_over = False
        self.last_push_direction = None

        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0.0
        self.last_push_direction = None

        if movement > 0: # An actual push action
            self.moves_left -= 1
            reward -= 0.1 # Penalty for using a move

            blocks_on_target_before = sum(1 for b in self.blocks if b.on_target)
            
            self._execute_push(movement)
            self.last_push_direction = movement

            self._update_on_target_status()
            blocks_on_target_after = sum(1 for b in self.blocks if b.on_target)
            
            newly_on_target = blocks_on_target_after - blocks_on_target_before
            
            if newly_on_target > 0:
                reward += newly_on_target * 1.0 # Reward for placing blocks
            elif newly_on_target < 0:
                # Negative reward for moving a block off its target
                reward += newly_on_target * -1 * -0.2

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if all(b.on_target for b in self.blocks):
                reward += 100.0 # Win bonus
            elif self.moves_left <= 0:
                reward += -10.0 # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        # A simple score for human players
        score = sum(10 for b in self.blocks if b.on_target) - (self.MOVE_LIMIT - self.moves_left)
        return {
            "score": score,
            "moves_left": self.moves_left,
            "steps": self.steps,
        }

    def _generate_puzzle(self):
        """Creates a new, solvable puzzle."""
        self.blocks.clear()
        self.targets.clear()

        available_positions = set((c, r) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))
        
        chosen_colors = self.np_random.choice(len(self.BLOCK_COLORS), self.NUM_BLOCKS, replace=False)
        
        # Use .choice on a list of tuples for compatibility
        target_positions_list = self.np_random.choice(list(available_positions), size=self.NUM_BLOCKS, replace=False)
        
        for i in range(self.NUM_BLOCKS):
            color_idx = chosen_colors[i]
            pos = tuple(target_positions_list[i])
            self.targets.append(GameObject(pos[0], pos[1], self.BLOCK_COLORS[color_idx]))
            self.blocks.append(GameObject(pos[0], pos[1], self.BLOCK_COLORS[color_idx]))
            available_positions.remove(pos)

        # Shuffle by making random inverse moves to guarantee solvability
        num_shuffles = self.np_random.integers(8, 15)
        for _ in range(num_shuffles):
            move_dir = self.np_random.integers(1, 5) # 1-4
            self._execute_push(move_dir, is_setup=True)
        
        for block in self.blocks:
            block.prev_grid_x = block.grid_x
            block.prev_grid_y = block.grid_y
        
        self._update_on_target_status()

    def _execute_push(self, direction, is_setup=False):
        """Moves all blocks in a given direction, handling collisions."""
        # 1=up, 2=down, 3=left, 4=right
        if direction == 1: dx, dy, sort_key, rev = 0, -1, lambda b: b.grid_y, False
        elif direction == 2: dx, dy, sort_key, rev = 0, 1, lambda b: b.grid_y, True
        elif direction == 3: dx, dy, sort_key, rev = -1, 0, lambda b: b.grid_x, False
        else: dx, dy, sort_key, rev = 1, 0, lambda b: b.grid_x, True
            
        sorted_blocks = sorted(self.blocks, key=sort_key, reverse=rev)
        
        if not is_setup:
            for block in self.blocks:
                block.prev_grid_x, block.prev_grid_y = block.grid_x, block.grid_y

        occupied = {(b.grid_x, b.grid_y) for b in self.blocks}

        for block in sorted_blocks:
            curr_pos = (block.grid_x, block.grid_y)
            next_pos = (block.grid_x + dx, block.grid_y + dy)

            if not (0 <= next_pos[0] < self.GRID_COLS and 0 <= next_pos[1] < self.GRID_ROWS): continue
            if next_pos in occupied: continue

            block.grid_x, block.grid_y = next_pos
            occupied.remove(curr_pos)
            occupied.add(next_pos)

    def _update_on_target_status(self):
        """Checks which blocks are on their correct targets."""
        target_map = {(t.grid_x, t.grid_y): t.color for t in self.targets}
        for block in self.blocks:
            pos = (block.grid_x, block.grid_y)
            block.on_target = pos in target_map and target_map[pos] == block.color

    def _check_termination(self):
        """Checks for win or loss conditions."""
        if all(b.on_target for b in self.blocks) or self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _render_game(self):
        """Renders the main game grid, targets, and blocks."""
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GRID_HEIGHT))

        # Draw targets
        for target in self.targets:
            rect = pygame.Rect(target.grid_x * self.CELL_SIZE, target.grid_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect)
            inner_rect = rect.inflate(-self.CELL_SIZE * 0.2, -self.CELL_SIZE * 0.2)
            pygame.draw.rect(self.screen, target.color, inner_rect, border_radius=4)
        
        # Draw block trails for motion feedback
        if self.last_push_direction:
            for block in self.blocks:
                if block.grid_x != block.prev_grid_x or block.grid_y != block.prev_grid_y:
                    trail_color = tuple(c * 0.4 for c in block.color)
                    trail_rect = pygame.Rect(block.prev_grid_x * self.CELL_SIZE, block.prev_grid_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE).inflate(-4, -4)
                    pygame.draw.rect(self.screen, trail_color, trail_rect, border_radius=8)

        # Draw blocks
        for block in self.blocks:
            rect = pygame.Rect(block.grid_x * self.CELL_SIZE, block.grid_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE).inflate(-4, -4)
            pygame.draw.rect(self.screen, block.color, rect, border_radius=8)
            
            # Highlight for 3D effect
            highlight_color = tuple(min(255, c + 30) for c in block.color)
            highlight_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height * 0.4)
            pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_top_left_radius=8, border_top_right_radius=8)
            
            # "Solved" indicator
            if block.on_target:
                center = rect.center
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 5, (255, 255, 255, 180))
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 5, (255, 255, 255, 180))

    def _render_ui(self):
        """Renders UI elements like score and moves left."""
        ui_panel_rect = pygame.Rect(0, self.GRID_HEIGHT, self.WIDTH, self.HEIGHT - self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG, ui_panel_rect)

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, self.GRID_HEIGHT + 5))

        info = self._get_info()
        score_text = self.font_large.render(f"Score: {info['score']}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(right=self.WIDTH - 20, top=self.GRID_HEIGHT + 5)
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win = all(b.on_target for b in self.blocks)
            msg = "PUZZLE SOLVED!" if win else "OUT OF MOVES"
            color = self.BLOCK_COLORS[1] if win else self.BLOCK_COLORS[0]
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,200), end_rect.inflate(20,20), border_radius=10)
            self.screen.blit(end_text, end_rect)
            
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Pusher")
    
    running = True
    while running:
        action = [0, 0, 0] # Default: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                elif not env.game_over:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()