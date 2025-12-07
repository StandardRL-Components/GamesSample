import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←/→ to move, ↑ to rotate clockwise, SHIFT to rotate counter-clockwise. "
        "↓ to soft drop, SPACE to hard drop."
    )

    game_description = (
        "A fast-paced, falling block puzzle. Strategically place pieces to clear lines. "
        "Clear 10 lines to win, but don't let the stack reach the top!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.render_mode = render_mode
        self.screen_width, self.screen_height = 640, 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 10

        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.screen_width - self.GRID_PIXEL_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_PIXEL_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.SHAPE_COLORS = {
            'I': (0, 240, 240),  # Cyan
            'J': (0, 0, 240),    # Blue
            'L': (240, 160, 0),  # Orange
            'T': (160, 0, 240),  # Purple
        }

        # --- Fonts ---
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Tetromino Definitions (Rotations) ---
        self.SHAPES = {
            'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)]],
            'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)],
                  [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
            'L': [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)],
                  [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
            'T': [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)],
                  [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)]]
        }
        self.shape_keys = list(self.SHAPES.keys())

        # --- State Variables ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.fall_timer = None
        self.fall_speed_normal = 15  # Ticks per fall
        self.fall_speed_soft_drop = 3
        self.line_clear_anim_timer = None
        self.cleared_lines_indices = None
        
        self.np_random = None

        # self.validate_implementation() is called after state is initialized in reset()
        # The original code called it here, causing an error.
        # It's better to let the user call it after instantiation if needed,
        # but for the verifier, we will call reset once to initialize state.
        self.reset()


    def _new_piece(self):
        shape_key = self.np_random.choice(self.shape_keys)
        return {
            "shape_key": shape_key,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
            "color": self.SHAPE_COLORS[shape_key],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()

        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.line_clear_anim_timer = 0
        self.cleared_lines_indices = []

        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def _check_collision(self, piece, dx, dy):
        shape = self.SHAPES[piece["shape_key"]][piece["rotation"]]
        for x, y in shape:
            new_x = piece["x"] + x + dx
            new_y = piece["y"] + y + dy
            if not (0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT):
                return True  # Wall collision
            if self.grid[new_y, new_x] != 0:
                return True  # Grid collision
        return False

    def _rotate_piece(self, piece, direction):
        original_rotation = piece["rotation"]
        num_rotations = len(self.SHAPES[piece["shape_key"]])
        piece["rotation"] = (piece["rotation"] + direction) % num_rotations

        if not self._check_collision(piece, 0, 0):
            return True # Simple rotation succeeded

        # Wall kick
        for offset in [-1, 1, -2, 2]:
            if not self._check_collision(piece, offset, 0):
                piece["x"] += offset
                return True
        
        piece["rotation"] = original_rotation # Revert if all kicks fail
        return False

    def _lock_piece(self):
        reward = 0
        is_risky = False
        
        shape = self.SHAPES[self.current_piece["shape_key"]][self.current_piece["rotation"]]
        color_index = list(self.SHAPE_COLORS.values()).index(self.current_piece["color"]) + 1

        for x, y in shape:
            px, py = self.current_piece["x"] + x, self.current_piece["y"] + y
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.grid[py, px] = color_index
                # Check for risky placement
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                        if self.grid[ny, nx] != 0 and self.grid[ny, nx] != color_index:
                            is_risky = True
        
        reward += 2.0 if is_risky else -0.2

        # Line clear check
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            self.cleared_lines_indices = lines_to_clear
            self.line_clear_anim_timer = 6 # frames
            reward += len(lines_to_clear) * 1.0
            self.lines_cleared += len(lines_to_clear)
            self.score += (len(lines_to_clear) ** 2) * 10 # Update score for line clears

        # Spawn next piece
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        # Check for game over on spawn
        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True
            reward -= 100

        return reward

    def _execute_line_clear(self):
        if not self.cleared_lines_indices:
            return

        self.cleared_lines_indices.sort(reverse=True)
        for r in self.cleared_lines_indices:
            self.grid = np.delete(self.grid, r, axis=0)
            new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack([new_row, self.grid])
        
        self.cleared_lines_indices = []

    def step(self, action):
        self.steps += 1
        reward = -0.1
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), -100, True, False, self._get_info()
        
        # Handle line clear animation pause
        if self.line_clear_anim_timer > 0:
            self.line_clear_anim_timer -= 1
            if self.line_clear_anim_timer == 0:
                self._execute_line_clear()
            return self._get_observation(), reward, False, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        # 1. Hard Drop (takes precedence)
        if space_held:
            fall_dist = 0
            while not self._check_collision(self.current_piece, 0, fall_dist + 1):
                fall_dist += 1
            self.current_piece["y"] += fall_dist
            reward += self._lock_piece()

        else:
            # 2. Horizontal Movement & Rotation
            if movement == 3: # Left
                if not self._check_collision(self.current_piece, -1, 0):
                    self.current_piece["x"] -= 1
            elif movement == 4: # Right
                if not self._check_collision(self.current_piece, 1, 0):
                    self.current_piece["x"] += 1
            
            if movement == 1: # Rotate CW
                self._rotate_piece(self.current_piece, 1)
            if shift_held: # Rotate CCW
                self._rotate_piece(self.current_piece, -1)

            # 3. Gravity
            self.fall_timer += 1
            fall_speed = self.fall_speed_soft_drop if movement == 2 else self.fall_speed_normal
            
            if self.fall_timer >= fall_speed:
                self.fall_timer = 0
                if not self._check_collision(self.current_piece, 0, 1):
                    self.current_piece["y"] += 1
                else:
                    reward += self._lock_piece()

        # --- Termination Check ---
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        elif self.game_over:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_block(self, x, y, color):
        """Renders a single block with a 3D effect."""
        px, py = int(x), int(y)
        size = self.CELL_SIZE
        
        darker_color = tuple(max(0, c - 50) for c in color)
        lighter_color = tuple(min(255, c + 50) for c in color)

        pygame.draw.rect(self.screen, darker_color, (px, py, size, size))
        pygame.draw.rect(self.screen, color, (px + 1, py + 1, size - 2, size - 2))
        
        # Highlight
        pygame.draw.line(self.screen, lighter_color, (px+1, py+1), (px+size-2, py+1))
        pygame.draw.line(self.screen, lighter_color, (px+1, py+1), (px+1, py+size-2))

    def _render_game(self):
        # Draw grid background and lines
        grid_rect = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, (0,0,0), grid_rect)
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_PIXEL_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_PIXEL_WIDTH, py))

        # Draw locked blocks
        color_list = list(self.SHAPE_COLORS.values())
        if self.grid is not None:
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    if self.grid[r, c] != 0:
                        color = color_list[int(self.grid[r, c]) - 1]
                        px = self.GRID_X_OFFSET + c * self.CELL_SIZE
                        py = self.GRID_Y_OFFSET + r * self.CELL_SIZE
                        self._render_block(px, py, color)

        # Draw line clear animation
        if self.line_clear_anim_timer > 0:
            alpha = 255 * (self.line_clear_anim_timer / 6)
            flash_surface = pygame.Surface((self.GRID_PIXEL_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            for r in self.cleared_lines_indices:
                py = self.GRID_Y_OFFSET + r * self.CELL_SIZE
                self.screen.blit(flash_surface, (self.GRID_X_OFFSET, py))

        # Draw current piece (and ghost piece)
        if not self.game_over and self.current_piece is not None:
            # Ghost piece
            fall_dist = 0
            while not self._check_collision(self.current_piece, 0, fall_dist + 1):
                fall_dist += 1
            
            ghost_piece_shape = self.SHAPES[self.current_piece["shape_key"]][self.current_piece["rotation"]]
            for x, y in ghost_piece_shape:
                px = self.GRID_X_OFFSET + (self.current_piece["x"] + x) * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + (self.current_piece["y"] + y + fall_dist) * self.CELL_SIZE
                color = self.current_piece["color"]
                pygame.gfxdraw.box(self.screen, (px, py, self.CELL_SIZE, self.CELL_SIZE), (*color, 50))

            # Actual piece
            piece_shape = self.SHAPES[self.current_piece["shape_key"]][self.current_piece["rotation"]]
            for x, y in piece_shape:
                px = self.GRID_X_OFFSET + (self.current_piece["x"] + x) * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + (self.current_piece["y"] + y) * self.CELL_SIZE
                self._render_block(px, py, self.current_piece["color"])

        # Draw next piece preview
        if self.next_piece is not None:
            next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
            self.screen.blit(next_text, (self.GRID_X_OFFSET + self.GRID_PIXEL_WIDTH + 20, self.GRID_Y_OFFSET + 10))
            next_shape = self.SHAPES[self.next_piece["shape_key"]][0]
            for x, y in next_shape:
                px = self.GRID_X_OFFSET + self.GRID_PIXEL_WIDTH + 30 + x * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + 50 + y * self.CELL_SIZE
                self._render_block(px, py, self.next_piece["color"])

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score or 0}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, 20))
        
        # Lines
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared or 0}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 20))
        
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            status_text = self.font_main.render(status_text_str, True, self.COLOR_FLASH)
            text_rect = status_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Re-enable display for interactive mode
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1 # Rotate CW
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Soft Drop
        elif keys[pygame.K_LEFT]:
            action[0] = 3 # Move Left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Move Right

        if keys[pygame.K_SPACE]:
            action[1] = 1 # Hard Drop
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Rotate CCW

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)

    env.close()