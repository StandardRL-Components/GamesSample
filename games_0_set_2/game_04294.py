
# Generated: 2025-08-28T01:57:07.259879
# Source Brief: brief_04294.md
# Brief Index: 4294

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select and clear a group of same-colored blocks."
    )

    game_description = (
        "Clear the board by matching groups of adjacent colored blocks before you run out of moves. Larger groups give more points!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.MAX_MOVES = 15
        self.MAX_STEPS = 1000
        self.MIN_GROUP_SIZE = 3

        # Visuals
        self.FONT_MAIN = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_TITLE = pygame.font.SysFont("Consolas", 48, bold=True)
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_HIGHLIGHT = (255, 255, 255, 100)
        self.BLOCK_COLORS = [
            None,  # 0 is empty
            (255, 80, 80),   # 1: Red
            (80, 255, 80),   # 2: Green
            (80, 150, 255),  # 3: Blue
            (255, 255, 80),  # 4: Yellow
            (200, 80, 255),  # 5: Purple
            (255, 150, 50),  # 6: Orange
        ]
        self.BLOCK_SHADOW_FACTOR = 0.6
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = 20
        self.BLOCK_HEIGHT = 25
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 120

        # State variables are initialized in reset()
        self.board = []
        self.cursor_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.particles = []
        
        self.board = self._generate_board()
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        self._move_cursor(movement)

        if space_pressed and not self.game_over:
            self.moves_left -= 1
            reward += self._attempt_clear()

        self.steps += 1
        self._update_particles()
        
        board_cleared = all(self.board[r][c] == 0 for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))
        
        terminated = False
        if board_cleared:
            reward += 100.0  # Goal-oriented reward for clearing the board
            terminated = True
        elif self.moves_left <= 0:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _to_iso(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH / 2
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT / 2
        return int(x), int(y)

    def _generate_board(self):
        while True:
            board = [
                [self.np_random.integers(1, len(self.BLOCK_COLORS)) for _ in range(self.GRID_COLS)]
                for _ in range(self.GRID_ROWS)
            ]
            if self._has_valid_move(board):
                return board

    def _has_valid_move(self, board):
        visited = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in visited:
                    group = self._find_connected_on_board((r, c), board)
                    if len(group) >= self.MIN_GROUP_SIZE:
                        return True
                    visited.update(group)
        return False
        
    def _find_connected_on_board(self, start_pos, board):
        r_start, c_start = start_pos
        if not (0 <= r_start < self.GRID_ROWS and 0 <= c_start < self.GRID_COLS):
            return []
        
        target_color = board[r_start][c_start]
        if target_color == 0:
            return []

        q = deque([start_pos])
        visited = {tuple(start_pos)}
        group = []

        while q:
            r, c = q.popleft()
            group.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and
                        (nr, nc) not in visited and board[nr][nc] == target_color):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return group

    def _move_cursor(self, movement):
        r, c = self.cursor_pos
        if movement == 1: # Up (iso up-left)
            self.cursor_pos[0] = max(0, r - 1)
        elif movement == 2: # Down (iso down-right)
            self.cursor_pos[0] = min(self.GRID_ROWS - 1, r + 1)
        elif movement == 3: # Left (iso down-left)
            self.cursor_pos[1] = max(0, c - 1)
        elif movement == 4: # Right (iso up-right)
            self.cursor_pos[1] = min(self.GRID_COLS - 1, c + 1)

    def _attempt_clear(self):
        group = self._find_connected_on_board(self.cursor_pos, self.board)
        
        if len(group) >= self.MIN_GROUP_SIZE:
            # Sound: Block clear
            num_cleared = len(group)
            base_reward = num_cleared  # +1 per block
            bonus_reward = 5.0 if num_cleared >= 4 else 0.0

            color_index = self.board[group[0][0]][group[0][1]]
            for r, c in group:
                self.board[r][c] = 0
                self._create_particles(r, c, color_index)
            
            self._apply_gravity()
            self.score += num_cleared
            return base_reward + bonus_reward
        else:
            # Sound: Invalid move
            return -0.1

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.board[r][c] != 0:
                    if r != empty_row:
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = 0
                    empty_row -= 1

    def _create_particles(self, r, c, color_index):
        x, y = self._to_iso(r, c)
        y -= self.BLOCK_HEIGHT // 2
        base_color = self.BLOCK_COLORS[color_index]
        for _ in range(10):
            self.particles.append({
                'pos': [x, y],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-4, -1)],
                'life': self.np_random.integers(20, 40),
                'color': base_color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid_and_blocks()
        self._render_highlight_and_cursor()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_blocks(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                x, y = self._to_iso(r, c)
                color_index = self.board[r][c]
                if color_index != 0:
                    self._draw_iso_block(x, y, color_index)
                else:
                    # Draw floor tile
                    points = [
                        (x, y),
                        (x + self.TILE_WIDTH / 2, y + self.TILE_HEIGHT / 2),
                        (x, y + self.TILE_HEIGHT),
                        (x - self.TILE_WIDTH / 2, y + self.TILE_HEIGHT / 2),
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _draw_iso_block(self, x, y, color_index):
        color = self.BLOCK_COLORS[color_index]
        shadow_color = tuple(int(c * self.BLOCK_SHADOW_FACTOR) for c in color)
        darker_shadow_color = tuple(int(c * self.BLOCK_SHADOW_FACTOR * 0.8) for c in color)

        w, h = self.TILE_WIDTH / 2, self.TILE_HEIGHT / 2
        bh = self.BLOCK_HEIGHT

        # Top face
        top_points = [(x, y - bh), (x + w, y - bh + h), (x, y - bh + 2 * h), (x - w, y - bh + h)]
        pygame.draw.polygon(self.screen, color, top_points)
        
        # Left face
        left_points = [(x - w, y - bh + h), (x, y - bh + 2 * h), (x, y + 2 * h), (x - w, y + h)]
        pygame.draw.polygon(self.screen, shadow_color, left_points)
        
        # Right face
        right_points = [(x + w, y - bh + h), (x, y - bh + 2 * h), (x, y + 2 * h), (x + w, y + h)]
        pygame.draw.polygon(self.screen, darker_shadow_color, right_points)

    def _render_highlight_and_cursor(self):
        if self.game_over: return

        # Highlight connected group
        group = self._find_connected_on_board(self.cursor_pos, self.board)
        if len(group) >= self.MIN_GROUP_SIZE:
            for r, c in group:
                x, y = self._to_iso(r, c)
                w, h = self.TILE_WIDTH / 2, self.TILE_HEIGHT / 2
                bh = self.BLOCK_HEIGHT
                points = [(x, y - bh), (x + w, y - bh + h), (x, y - bh + 2 * h), (x - w, y - bh + h)]
                
                # Use a surface for alpha blending
                highlight_surf = pygame.Surface((self.TILE_WIDTH, self.TILE_HEIGHT + self.BLOCK_HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(highlight_surf, self.COLOR_HIGHLIGHT, [(p[0]-x+w, p[1]-y+bh) for p in points])
                self.screen.blit(highlight_surf, (x-w, y-bh))

        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._to_iso(r, c)
        w, h = self.TILE_WIDTH / 2, self.TILE_HEIGHT / 2
        bh = self.BLOCK_HEIGHT
        
        points = [(x, y - bh), (x + w, y - bh + h), (x, y - bh + 2 * h), (x - w, y - bh + h)]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = p['color'] + (alpha,)
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))
            
    def _render_ui(self):
        score_text = self.FONT_MAIN.render(f"SCORE: {self.score}", True, (255, 255, 255))
        moves_text = self.FONT_MAIN.render(f"MOVES: {self.moves_left}", True, (255, 255, 255))
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            board_cleared = all(self.board[r][c] == 0 for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS))
            end_text_str = "YOU WIN!" if board_cleared else "GAME OVER"
            end_text = self.FONT_TITLE.render(end_text_str, True, self.COLOR_CURSOR)
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Block Matcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    print("--- Game Started ---")
    print(env.user_guide)
    print("--------------------")

    while not done:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        # Only step if an action is taken in this turn-based game
        if movement != 0 or space != 0 or shift != 0:
            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {env.moves_left}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it from the env if no action was taken
        if movement == 0 and space == 0 and shift == 0:
            obs = env._get_observation()

        # Transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    print("--- Game Over ---")
    print(f"Final Score: {env.score}")
    
    # Keep the window open for a few seconds to show the final state
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)
        
    env.close()