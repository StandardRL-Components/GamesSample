
# Generated: 2025-08-27T17:33:08.720505
# Source Brief: brief_01570.md
# Brief Index: 1570

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block group. Hold Shift to restart."
    )

    game_description = (
        "Clear the grid by selecting groups of 3 or more same-colored blocks. You have a limited number of moves. Plan your moves to create chain reactions and maximize your score!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.GRID_X_OFFSET, self.GRID_Y_OFFSET = 160, 40
        self.BLOCK_SIZE = 40
        self.INITIAL_MOVES = 10
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 80, 80),   # 1: Red
            (80, 255, 80),   # 2: Green
            (80, 150, 255),  # 3: Blue
            (255, 255, 80),  # 4: Yellow
            (200, 80, 255),  # 5: Purple
            (100, 110, 120), # 6: Obstacle
        ]
        self.COLOR_OBSTACLE_BORDER = (80, 90, 100)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)

        # --- Fonts ---
        try:
            self.FONT_LARGE = pygame.font.Font(None, 48)
            self.FONT_MEDIUM = pygame.font.Font(None, 32)
            self.FONT_SMALL = pygame.font.Font(None, 24)
        except pygame.error:
            self.FONT_LARGE = pygame.font.SysFont("sans", 48)
            self.FONT_MEDIUM = pygame.font.SysFont("sans", 32)
            self.FONT_SMALL = pygame.font.SysFont("sans", 24)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.last_space_held = False
        self.particles = []
        self.falling_blocks = []
        self.selection_feedback_timer = 0
        self.selection_feedback_valid = False
        self.steps = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        while True:
            self._create_grid()
            if self._has_valid_moves():
                break

        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.last_space_held = False
        self.particles = []
        self.falling_blocks = []
        self.selection_feedback_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0
        terminated = False

        if shift_held:
            obs, info = self.reset()
            return obs, -1, True, False, info # Penalize agent for resetting

        self._update_animations()

        # Only process new actions if no animations are running
        if not self.falling_blocks:
            # --- Handle Movement ---
            if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
            elif movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1
            elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
            elif movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1

            # --- Handle Selection ---
            space_pressed = space_held and not self.last_space_held
            if space_pressed:
                reward, match_found = self._process_selection()
                self.selection_feedback_timer = 30 # 1 second at 30fps
                self.selection_feedback_valid = match_found

        self.last_space_held = space_held

        # --- Check Termination Conditions ---
        if not self.game_over:
            is_won = self._check_win_condition()
            no_more_moves = not self._has_valid_moves()
            
            if self.moves_left <= 0 or is_won or (no_more_moves and not self.falling_blocks):
                self.game_over = True
                terminated = True
                if is_won:
                    reward += 100  # Win bonus
                else:
                    reward -= 100 # Lose penalty
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_selection(self):
        if self.moves_left <= 0:
            return 0, False

        self.moves_left -= 1
        y, x = self.cursor_pos
        color_idx = self.grid[y, x]

        if color_idx == 0 or color_idx == 6: # Empty or Obstacle
            # sfx: invalid_move.wav
            return -1.0, False 

        connected_blocks = self._find_connected_blocks(self.cursor_pos)

        if len(connected_blocks) < 3:
            # sfx: invalid_move.wav
            return -1.0, False

        # --- Valid Match Found ---
        # sfx: match_success.wav
        reward = len(connected_blocks)
        if len(connected_blocks) >= 5:
            reward += 5 # Bonus

        for r, c in connected_blocks:
            self._create_particles(r, c, self.COLORS[self.grid[r, c]])
            self.grid[r, c] = 0
        
        self._resolve_gravity()
        return float(reward), True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Blocks that are NOT currently falling
        animating_blocks_coords = {(b['to_pos'][0], b['to_pos'][1]) for b in self.falling_blocks}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in animating_blocks_coords:
                    self._draw_block(r, c, self.grid[r, c])

        # Falling blocks
        for block in self.falling_blocks:
            from_r, from_c = block['from_pos']
            to_r, to_c = block['to_pos']
            
            interp_r = from_r + (to_r - from_r) * block['progress']
            self._draw_block(interp_r, to_c, block['color_idx'], is_pixel_pos=False)

        # Particles
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

        self.particles = [p for p in self.particles if p['life'] > 0]

        # Cursor
        if self.selection_feedback_timer > 0:
            self.selection_feedback_timer -= 1
            color = (0, 255, 0) if self.selection_feedback_valid else (255, 0, 0)
            alpha = int(150 * (self.selection_feedback_timer / 30))
        else:
            color = self.COLOR_CURSOR
            alpha = int(100 + 50 * math.sin(pygame.time.get_ticks() * 0.01))

        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + self.cursor_pos[1] * self.BLOCK_SIZE,
            self.GRID_Y_OFFSET + self.cursor_pos[0] * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*color, alpha), (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=5)
        pygame.draw.rect(cursor_surface, color, (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), 3, border_radius=5)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _draw_block(self, r, c, color_idx, is_pixel_pos=False):
        if color_idx == 0: return

        if not is_pixel_pos:
            px = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
            py = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
        else:
            px, py = r, c

        border_size = 3
        color = self.COLORS[color_idx]
        
        rect = pygame.Rect(int(px), int(py), self.BLOCK_SIZE, self.BLOCK_SIZE)
        inner_rect = pygame.Rect(int(px + border_size), int(py + border_size), self.BLOCK_SIZE - 2 * border_size, self.BLOCK_SIZE - 2 * border_size)

        if color_idx == 6: # Obstacle
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_BORDER, rect, border_radius=4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)
        else: # Regular block
            light_color = tuple(min(255, val + 40) for val in color)
            pygame.draw.rect(self.screen, light_color, rect, border_radius=6)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.FONT_LARGE.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        score_label = self.FONT_SMALL.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (20, 50))

        # Moves
        moves_text = self.FONT_LARGE.render(f"{self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        moves_label = self.FONT_SMALL.render("MOVES", True, self.COLOR_TEXT)
        moves_label_rect = moves_label.get_rect(topright=(self.WIDTH - 20, 50))
        self.screen.blit(moves_label, moves_label_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._check_win_condition():
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.FONT_LARGE.render(msg, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _create_grid(self):
        self.grid = self.np_random.integers(1, len(self.COLORS) - 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        num_obstacles = self.np_random.integers(3, 7)
        for _ in range(num_obstacles):
            r, c = self.np_random.integers(0, self.GRID_HEIGHT), self.np_random.integers(0, self.GRID_WIDTH)
            self.grid[r, c] = 6 # Obstacle

    def _has_valid_moves(self):
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in visited and self.grid[r,c] not in [0, 6]:
                    connected = self._find_connected_blocks([r, c])
                    if len(connected) >= 3:
                        return True
                    visited.update(connected)
        return False

    def _find_connected_blocks(self, start_pos):
        y_start, x_start = start_pos
        target_color = self.grid[y_start, x_start]
        if target_color == 0 or target_color == 6:
            return []

        q = deque([start_pos])
        connected = {tuple(start_pos)}
        
        while q:
            y, x = q.popleft()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and \
                   self.grid[ny, nx] == target_color and (ny, nx) not in connected:
                    connected.add((ny, nx))
                    q.append([ny, nx])
        return list(connected)

    def _resolve_gravity(self):
        # sfx: blocks_fall.wav
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        # Create animation
                        self.falling_blocks.append({
                            'from_pos': [r, c],
                            'to_pos': [empty_row, c],
                            'color_idx': self.grid[r, c],
                            'progress': 0.0
                        })
                        # Move block in grid state
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _update_animations(self):
        if not self.falling_blocks:
            return

        for block in self.falling_blocks:
            block['progress'] += 0.15 # Animation speed
        
        finished_blocks = [b for b in self.falling_blocks if b['progress'] >= 1.0]
        self.falling_blocks = [b for b in self.falling_blocks if b['progress'] < 1.0]
        
        # if finished_blocks and not self.falling_blocks:
            # sfx: blocks_land.wav

    def _create_particles(self, r, c, color):
        px = self.GRID_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'size': random.randint(3, 6)
            })

    def _check_win_condition(self):
        return not np.any((self.grid > 0) & (self.grid < 6))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert info['moves_left'] == self.INITIAL_MOVES
        assert info['score'] == 0
        assert self._has_valid_moves(), "Reset failed to generate a valid board"
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Play Loop ---
    print("\n" + "="*30)
    print("      BLOCK BREAKER      ")
    print("="*30)
    print(env.user_guide)
    print("Press ESC or close window to quit.")

    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    running = True
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = False
    shift_held = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key states
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        if terminated or truncated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            
            # Display final frame for 2 seconds before reset
            render_screen.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            
            obs, info = env.reset()
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance=False, we need to control the step rate for human play
        clock.tick(30) # Run at 30 FPS

    env.close()