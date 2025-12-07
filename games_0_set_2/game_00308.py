
# Generated: 2025-08-27T13:16:32.593076
# Source Brief: brief_00308.md
# Brief Index: 308

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate CW, Shift to rotate CCW. ↓ for soft drop, Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling shapes to clear lines and reach the target height. Manage your stack to avoid reaching the top!"
    )

    # Frames auto-advance for real-time gameplay.
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 22
        self.CELL_SIZE = 16
        self.GAME_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GAME_AREA_HEIGHT = (self.GRID_HEIGHT - 2) * self.CELL_SIZE # Hide top 2 rows
        self.GAME_AREA_X = (self.screen_width - self.GAME_AREA_WIDTH) // 2
        self.GAME_AREA_Y = (self.screen_height - self.GAME_AREA_HEIGHT) // 2

        self.TARGET_HEIGHT = 18 # Win condition
        self.MAX_STEPS = 18000 # 10 minutes at 30fps

        # Colors
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_GRID = (30, 40, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TARGET_LINE = (255, 200, 0)
        
        # Fonts
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Tetromino shapes and colors
        self.SHAPES = {
            'I': [[(0, -1), (0, 0), (0, 1), (0, 2)], [(-1, 0), (0, 0), (1, 0), (2, 0)]],
            'J': [[(-1, -1), (0, -1), (0, 0), (0, 1)], [(-1, 1), (-1, 0), (0, 0), (1, 0)], [(1, 1), (0, 1), (0, 0), (0, -1)], [(1, -1), (1, 0), (0, 0), (-1, 0)]],
            'L': [[(1, -1), (0, -1), (0, 0), (0, 1)], [(-1, -1), (-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (0, 0), (0, -1)], [(1, 1), (1, 0), (0, 0), (-1, 0)]],
            'O': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
            'S': [[(0, 1), (0, 0), (1, 0), (1, -1)], [(-1, 0), (0, 0), (0, 1), (1, 1)]],
            'T': [[(-1, 0), (0, 0), (1, 0), (0, -1)], [(0, -1), (0, 0), (0, 1), (-1, 0)], [(-1, 0), (0, 0), (1, 0), (0, 1)], [(0, -1), (0, 0), (0, 1), (1, 0)]],
            'Z': [[(0, -1), (0, 0), (1, 0), (1, 1)], [(-1, 1), (0, 1), (0, 0), (1, 0)]]
        }
        self.SHAPE_KEYS = list(self.SHAPES.keys())
        self.SHAPE_COLORS = {
            'I': (0, 240, 240), 'J': (0, 0, 240), 'L': (240, 160, 0), 'O': (240, 240, 0),
            'S': (0, 240, 0), 'T': (160, 0, 240), 'Z': (240, 0, 0)
        }

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.lines_cleared_total = None
        self.current_height = None
        self.fall_speed_level = None
        self.fall_counter = None
        self.particles = None
        self.last_action = None
        self.risk_level = None
        self.reward_this_step = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[(0,0,0) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.lines_cleared_total = 0
        self.current_height = 0
        self.fall_speed_level = 1.0
        self.fall_counter = 0
        self.particles = []
        self.risk_level = 0.0
        self.reward_this_step = 0.0
        
        self.last_action = np.array([0, 0, 0])
        self._new_piece()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0.0

        self._handle_input(action)
        self._update_game_physics()
        self._update_particles()
        
        terminated = self._check_termination()
        reward = self.reward_this_step
        
        if terminated:
            if self.current_height >= self.TARGET_HEIGHT:
                reward += 100  # Win
            else:
                reward -= 100  # Lose
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        prev_movement = self.last_action[0]
        prev_shift_held = self.last_action[2] == 1
        
        # Hard drop (highest priority)
        if space_held:
            self._hard_drop()
            return # Skip other inputs for this frame

        # Horizontal movement
        if movement == 3: # Left
            if self._move_piece(-1, 0): self.reward_this_step -= 0.02
        elif movement == 4: # Right
            if self._move_piece(1, 0): self.reward_this_step -= 0.02
        
        # Soft drop
        if movement == 2: # Down
            if self._move_piece(0, 1): self.reward_this_step += 0.01

        # Rotation (on press)
        if movement == 1 and prev_movement != 1: # Rotate CW
            self._rotate_piece(1)
        if shift_held and not prev_shift_held: # Rotate CCW
            self._rotate_piece(-1)
            
        self.last_action = action

    def _update_game_physics(self):
        self.fall_counter += 1
        ticks_to_drop = max(3, 30.0 / self.fall_speed_level)
        
        if self.fall_counter >= ticks_to_drop:
            self.fall_counter = 0
            if not self._move_piece(0, 1):
                self._place_piece()

    def _new_piece(self):
        shape_key = self.np_random.choice(self.SHAPE_KEYS)
        self.current_piece = {
            'key': shape_key,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2,
            'y': 1,
            'color': self.SHAPE_COLORS[shape_key]
        }
        if self._check_collision():
            self.game_over = True

    def _check_collision(self, offset_x=0, offset_y=0, rotation=None):
        piece_rot = rotation if rotation is not None else self.current_piece['rotation']
        shape_coords = self.SHAPES[self.current_piece['key']][piece_rot]
        
        for x, y in shape_coords:
            grid_x = self.current_piece['x'] + x + offset_x
            grid_y = self.current_piece['y'] + y + offset_y
            
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return True # Wall collision
            if self.grid[grid_y][grid_x] != (0,0,0):
                return True # Piece collision
        return False

    def _move_piece(self, dx, dy):
        if not self._check_collision(offset_x=dx, offset_y=dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self, direction):
        key = self.current_piece['key']
        num_rotations = len(self.SHAPES[key])
        new_rotation = (self.current_piece['rotation'] + direction) % num_rotations
        
        # Wall kick checks
        for offset_x in [0, 1, -1, 2, -2]:
            if not self._check_collision(offset_x=offset_x, rotation=new_rotation):
                self.current_piece['rotation'] = new_rotation
                self.current_piece['x'] += offset_x
                # sfx: rotate
                return True
        # sfx: rotate_fail
        return False

    def _hard_drop(self):
        # sfx: hard_drop
        while self._move_piece(0, 1):
            self.reward_this_step += 0.01 # Small reward for each cell dropped
        self._place_piece()

    def _place_piece(self):
        # sfx: place_piece
        shape_coords = self.SHAPES[self.current_piece['key']][self.current_piece['rotation']]
        
        # Calculate hole penalty before placing
        holes_created = 0
        for x, y in shape_coords:
            px, py = self.current_piece['x'] + x, self.current_piece['y'] + y
            if py < self.GRID_HEIGHT - 1 and self.grid[py + 1][px] == (0,0,0):
                is_covered = False
                for ox, oy in shape_coords:
                    if px == self.current_piece['x'] + ox and py + 1 == self.current_piece['y'] + oy:
                        is_covered = True
                        break
                if not is_covered:
                    holes_created += 1
        
        self.reward_this_step -= holes_created * 0.2

        # Place the piece
        for x, y in shape_coords:
            grid_x = self.current_piece['x'] + x
            grid_y = self.current_piece['y'] + y
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y][grid_x] = self.current_piece['color']

        self._clear_lines()
        self._update_height_and_risk()
        self._new_piece()

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if all(cell != (0,0,0) for cell in self.grid[y]):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return

        # sfx: line_clear_multi
        for y in lines_to_clear:
            self.grid.pop(y)
            self.grid.insert(0, [(0,0,0) for _ in range(self.GRID_WIDTH)])
            self._spawn_particles(y)
        
        num_cleared = len(lines_to_clear)
        self.lines_cleared_total += num_cleared
        
        # Reward for line clears
        rewards = {1: 1, 2: 3, 3: 5, 4: 10}
        self.reward_this_step += rewards.get(num_cleared, 0)
        self.score += rewards.get(num_cleared, 0) * 100 * (self.lines_cleared_total // 50 + 1)
        
        # Update fall speed
        new_speed = 1.0 + (self.lines_cleared_total // 10) * 0.2
        if new_speed > self.fall_speed_level:
            self.fall_speed_level = new_speed
            # sfx: level_up

    def _update_height_and_risk(self):
        self.current_height = 0
        holes = 0
        for y in range(self.GRID_HEIGHT):
            row_has_block = any(cell != (0,0,0) for cell in self.grid[y])
            if row_has_block:
                self.current_height = self.GRID_HEIGHT - y
                # Count holes below the current line
                for x in range(self.GRID_WIDTH):
                    if self.grid[y][x] != (0,0,0):
                        for H_y in range(y + 1, self.GRID_HEIGHT):
                            if self.grid[H_y][x] == (0,0,0):
                                holes += 1
                break
        self.risk_level = min(1.0, holes / 50.0) # Normalize risk

    def _check_termination(self):
        if self.game_over: return True
        if self.current_height >= self.TARGET_HEIGHT: return True
        if self.steps >= self.MAX_STEPS: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.current_height, "lines": self.lines_cleared_total}

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GAME_AREA_X + x * self.CELL_SIZE, self.GAME_AREA_Y)
            end = (self.GAME_AREA_X + x * self.CELL_SIZE, self.GAME_AREA_Y + self.GAME_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT - 1):
            start = (self.GAME_AREA_X, self.GAME_AREA_Y + y * self.CELL_SIZE)
            end = (self.GAME_AREA_X + self.GAME_AREA_WIDTH, self.GAME_AREA_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw target line
        target_y = self.GAME_AREA_Y + (self.GRID_HEIGHT - 2 - self.TARGET_HEIGHT) * self.CELL_SIZE
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (self.GAME_AREA_X, target_y), (self.GAME_AREA_X + self.GAME_AREA_WIDTH, target_y), 2)

        # Draw risk glow
        if self.risk_level > 0.1:
            r = int(255 * min(1.0, self.risk_level * 2))
            g = int(255 * (1 - self.risk_level))
            glow_color = (r, g, 0)
            glow_radius = int(self.GAME_AREA_WIDTH * 0.6 * self.risk_level)
            glow_center = (self.GAME_AREA_X + self.GAME_AREA_WIDTH // 2, self.GAME_AREA_Y + self.GAME_AREA_HEIGHT)
            for i in range(glow_radius, 0, -2):
                alpha = int(30 * (1 - i/glow_radius))
                if alpha > 0:
                    pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, (*glow_color, alpha))
        
        # Draw placed pieces
        for y in range(2, self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] != (0,0,0):
                    self._draw_block(x, y - 2, self.grid[y][x])
        
        if not self.game_over:
            # Draw ghost piece
            ghost_y = self.current_piece['y']
            while not self._check_collision(offset_y=ghost_y - self.current_piece['y'] + 1):
                ghost_y += 1
            shape_coords = self.SHAPES[self.current_piece['key']][self.current_piece['rotation']]
            for x, y in shape_coords:
                if self.current_piece['y'] + y >= 2:
                    self._draw_block(self.current_piece['x'] + x, ghost_y + y - 2, self.current_piece['color'], is_ghost=True)

            # Draw current piece
            for x, y in shape_coords:
                if self.current_piece['y'] + y >= 2:
                    self._draw_block(self.current_piece['x'] + x, self.current_piece['y'] + y - 2, self.current_piece['color'])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _draw_block(self, x, y, color, is_ghost=False):
        rect = pygame.Rect(
            self.GAME_AREA_X + x * self.CELL_SIZE,
            self.GAME_AREA_Y + y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 1)
        else:
            pygame.draw.rect(self.screen, color, rect)
            l_color = tuple(min(255, c + 50) for c in color)
            d_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.line(self.screen, l_color, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, l_color, rect.topleft, rect.bottomleft)
            pygame.draw.line(self.screen, d_color, rect.bottomright, rect.topright)
            pygame.draw.line(self.screen, d_color, rect.bottomright, rect.bottomleft)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        height_text = self.font_main.render(f"HEIGHT: {self.current_height}/{self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (self.screen_width - height_text.get_width() - 20, 20))

        lines_text = self.font_small.render(f"LINES: {self.lines_cleared_total}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 55))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.current_height >= self.TARGET_HEIGHT else "GAME OVER"
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_y):
        y_pos = self.GAME_AREA_Y + (grid_y - 2) * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(30):
            self.particles.append({
                'pos': [self.GAME_AREA_X + self.np_random.random() * self.GAME_AREA_WIDTH, y_pos],
                'vel': [(self.np_random.random() - 0.5) * 4, (self.np_random.random() - 0.7) * 3],
                'life': self.np_random.integers(20, 40),
                'size': self.np_random.random() * 3 + 1,
                'color': (200, 200, 255)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] -= 0.05
            p['color'] = (max(0, p['color'][0]-5), max(0, p['color'][1]-5), 255)
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
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
    # Set Pygame to run headlessly if no display is available
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # Run for a few steps with random actions
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Game over! Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

    print("Example run completed.")

    # To visualize the game, you would need a display and run a loop like this:
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    # pygame.display.set_caption("Geometric Stacker")
    # display_screen = pygame.display.set_mode((640, 400))
    # running = True
    # env.reset()
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #     
    #     # This is where you'd get an action from a player or an agent
    #     # For this demo, we'll use a no-op
    #     action = np.array([0, 0, 0]) 
    #     obs, _, terminated, _, _ = env.step(action)
    #     
    #     if terminated:
    #         env.reset()
    #
    #     # Blit the observation from the env to the display screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     display_screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #     
    #     env.clock.tick(30) # Maintain 30 FPS
    # pygame.quit()