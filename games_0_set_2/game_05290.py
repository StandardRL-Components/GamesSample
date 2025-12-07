import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selected block. "
        "Press space to cycle selection forward, shift to cycle backward."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Slide colored blocks to fill the entire grid. "
        "You have a limited number of moves to solve each puzzle."
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
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAX_MOVES = 50
        self.MAX_STEPS = 500

        # Visuals
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_EMPTY = (40, 50, 60)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Bright Red
            (80, 255, 80),   # Bright Green
            (80, 150, 255),  # Bright Blue
            (255, 220, 80),  # Bright Yellow
        ]
        self.COLOR_SELECT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        # Use Pygame's default font for better portability
        self.font_main = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 52)

        # State variables
        self.grid = None
        self.blocks = []
        self.selected_block_idx = 0
        self.moves_made = 0
        self.score = 0.0
        self.game_over = False
        self.steps = 0
        self.last_move_info = None
        
        # Initialize state variables
        # The first reset call will use an unseeded RNG, which is standard.
        # Subsequent calls can be seeded.
        if render_mode == "rgb_array":
            self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_over = False
        self.score = 0.0
        self.moves_made = 0
        self.steps = 0
        self.last_move_info = None

        # Generate a grid with a guaranteed 40-60% fill rate efficiently
        grid_area = self.GRID_SIZE * self.GRID_SIZE
        grid_flat = np.zeros(grid_area, dtype=int)
        
        num_blocks = self.np_random.integers(
            low=int(grid_area * 0.4), 
            high=int(grid_area * 0.6) + 1
        )
        
        block_indices = self.np_random.choice(grid_area, size=num_blocks, replace=False)
        block_colors = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1, size=num_blocks)
        
        grid_flat[block_indices] = block_colors
        self.grid = grid_flat.reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        self._update_block_list()

        # With the new generation logic, there will always be blocks.
        self.selected_block_idx = self.np_random.integers(0, len(self.blocks))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        self.last_move_info = None  # Clear visual effect from previous step
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_fill_pct = self._get_fill_percentage()

        # Prioritize movement over selection
        if movement > 0 and self.blocks:
            moved = self._execute_move(movement)
            if moved:
                new_fill_pct = self._get_fill_percentage()
                # Reward for progress, penalized by move cost
                reward = (new_fill_pct - old_fill_pct) - 0.2
        elif (space_held or shift_held) and self.blocks:
            if space_held and not shift_held:
                self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
            elif shift_held and not space_held:
                self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
            # No reward for selection change
        
        self.score += reward

        terminated = self._check_termination()
        if terminated and self._get_fill_percentage() == 100.0:
            reward += 50.0  # Big bonus for winning
            self.score += 50.0

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _execute_move(self, move_direction):
        block_to_move = self.blocks[self.selected_block_idx]
        r, c = block_to_move['r'], block_to_move['c']
        color_idx = block_to_move['type'] - 1
        
        # 1=up, 2=down, 3=left, 4=right
        dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = dirs[move_direction]

        # Find destination
        nr, nc = r, c
        while True:
            next_r, next_c = nr + dr, nc + dc
            if not (0 <= next_r < self.GRID_SIZE and 0 <= next_c < self.GRID_SIZE):
                break  # Hit edge
            if self.grid[next_r, next_c] != 0:
                break  # Hit another block
            nr, nc = next_r, next_c

        if (nr, nc) != (r, c):
            self.grid[r, c] = 0
            self.grid[nr, nc] = block_to_move['type']
            self.moves_made += 1
            
            self.last_move_info = {'from': (r, c), 'to': (nr, nc), 'color_idx': color_idx}
            
            # Rebuild block list and keep selection on the moved block
            self._update_block_list()
            new_idx = next((i for i, b in enumerate(self.blocks) if b['r'] == nr and b['c'] == nc), 0)
            self.selected_block_idx = new_idx
            return True
        return False

    def _update_block_list(self):
        self.blocks.clear()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0:
                    self.blocks.append({'r': r, 'c': c, 'type': self.grid[r, c]})

    def _get_fill_percentage(self):
        if self.grid is None or self.grid.size == 0: return 0.0
        return np.count_nonzero(self.grid) * 100.0 / self.grid.size

    def _check_termination(self):
        if self.moves_made >= self.MAX_MOVES:
            self.game_over = True
        if self._get_fill_percentage() == 100.0:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y, 
                                self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw cells and blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(self.GRID_ORIGIN_X + c * self.CELL_SIZE,
                                        self.GRID_ORIGIN_Y + r * self.CELL_SIZE,
                                        self.CELL_SIZE, self.CELL_SIZE)
                
                block_type = self.grid[r, c]
                color = self.COLOR_EMPTY if block_type == 0 else self.BLOCK_COLORS[block_type - 1]
                
                pygame.gfxdraw.box(self.screen, cell_rect.inflate(-2, -2), color)

        # Draw move trail effect
        if self.last_move_info:
            from_r, from_c = self.last_move_info['from']
            to_r, to_c = self.last_move_info['to']
            color = self.BLOCK_COLORS[self.last_move_info['color_idx']]
            
            start_pos = (self.GRID_ORIGIN_X + from_c * self.CELL_SIZE + self.CELL_SIZE // 2,
                         self.GRID_ORIGIN_Y + from_r * self.CELL_SIZE + self.CELL_SIZE // 2)
            end_pos = (self.GRID_ORIGIN_X + to_c * self.CELL_SIZE + self.CELL_SIZE // 2,
                       self.GRID_ORIGIN_Y + to_r * self.CELL_SIZE + self.CELL_SIZE // 2)

            dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
            dist = max(1, math.hypot(dx, dy))
            steps = max(1, int(dist / (self.CELL_SIZE / 4)))
            for i in range(steps + 1):
                t = i / steps
                x = int(start_pos[0] + dx * t)
                y = int(start_pos[1] + dy * t)
                alpha = int(128 * (1 - t))
                radius = int((self.CELL_SIZE / 3) * (1 - t * 0.5))
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (*color, alpha))

        # Draw selection highlight
        if self.blocks and not self.game_over:
            selected_block = self.blocks[self.selected_block_idx]
            r, c = selected_block['r'], selected_block['c']
            
            sel_rect = pygame.Rect(self.GRID_ORIGIN_X + c * self.CELL_SIZE,
                                   self.GRID_ORIGIN_Y + r * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
            
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            line_width = int(2 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, sel_rect, line_width, border_radius=3)

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_made:02d}/{self.MAX_MOVES}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (20, 20))

        fill_pct = self._get_fill_percentage()
        fill_text = f"Fill: {fill_pct:.0f}%"
        fill_surf = self.font_main.render(fill_text, True, self.COLOR_TEXT)
        self.screen.blit(fill_surf, (self.WIDTH - fill_surf.get_width() - 20, 20))
        
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 50))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self._get_fill_percentage() == 100.0:
                msg = "GRID COMPLETE!"
                color = self.BLOCK_COLORS[1]
            else:
                msg = "GAME OVER"
                color = self.BLOCK_COLORS[0]
            
            msg_surf = self.font_big.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_made": self.moves_made,
            "fill_percentage": self._get_fill_percentage(),
        }

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # --- Pygame loop for human play ---
    running = True
    pygame.display.set_caption("Block Filler")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    running = False

        # --- Action polling for turn-based input ---
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space_held, shift_held])
        
        # Step only if an action is taken or if the game auto-advances
        if env.auto_advance or np.any(current_action):
            obs, reward, terminated, truncated, info = env.step(current_action)
            print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")
            if terminated or truncated:
                print("Game Over! Press 'R' to reset or 'Q' to quit.")

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap FPS to reduce CPU usage
        clock.tick(30)

    pygame.quit()
    env.close()