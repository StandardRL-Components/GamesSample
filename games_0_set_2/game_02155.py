
# Generated: 2025-08-28T03:55:18.353803
# Source Brief: brief_02155.md
# Brief Index: 2155

        
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
        "Controls: Arrow keys to move cursor. Space to select a block group. Shift to reshuffle the board."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid by selecting groups of 3 or more matching blocks. Plan your moves to create large combos and maximize your score before you run out of moves."
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
        self.NUM_COLORS = 5
        self.MIN_GROUP_SIZE = 3
        self.INITIAL_MOVES = 50
        self.MAX_STEPS = 1000

        # Visual constants
        self.BLOCK_SIZE = 36
        self.GRID_LINE_WIDTH = 2
        self.PARTICLE_LIFETIME = 30
        self.PARTICLE_COUNT = 15

        self.GRID_WIDTH = self.GRID_SIZE * self.BLOCK_SIZE + (self.GRID_SIZE + 1) * self.GRID_LINE_WIDTH
        self.GRID_HEIGHT = self.GRID_SIZE * self.BLOCK_SIZE + (self.GRID_SIZE + 1) * self.GRID_LINE_WIDTH
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID_BG = (40, 45, 55)
        self.COLOR_GRID_LINE = (50, 55, 65)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TEXT_SHADOW = (10, 10, 15)
        self.BLOCK_COLORS = [
            (227, 89, 89),   # Red
            (89, 139, 227),  # Blue
            (89, 227, 139),  # Green
            (227, 227, 89),  # Yellow
            (180, 89, 227),  # Purple
        ]
        self.BLOCK_SHADOW_COLORS = [
            (160, 60, 60),
            (60, 90, 160),
            (60, 160, 90),
            (160, 160, 60),
            (120, 60, 160),
        ]

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_combo_info = None
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        self.last_combo_info = {"size": 0, "timer": 0}

        self._generate_board()
        while not self._find_any_valid_move():
            self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._move_cursor(movement)

        action_taken = False
        if shift_held and self.moves_left > 0:
            # sfx: board_shuffle
            self.moves_left -= 1
            self._reshuffle_board_inplace()
            reward = -5  # Penalty for manual reshuffle
            action_taken = True
        elif space_held and self.moves_left > 0:
            reward = self._handle_click()
            action_taken = True

        if action_taken:
            # Check for win condition
            if np.all(self.grid == 0):
                reward += 100
                terminated = True
                self.game_over = True
            # Check for loss condition
            elif self.moves_left <= 0:
                reward += -10
                terminated = True
                self.game_over = True
            # Check for auto-reshuffle if no moves are left
            elif not self._find_any_valid_move():
                # sfx: auto_reshuffle
                self._reshuffle_board_inplace()
                reward -= 1 # Small penalty for getting stuck
        
        # Check for max steps termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_click(self):
        self.moves_left -= 1
        x, y = self.cursor_pos
        
        if self.grid[y, x] == 0:
            # sfx: invalid_move
            return -0.2

        group = self._find_group(x, y)
        
        if len(group) >= self.MIN_GROUP_SIZE:
            # sfx: clear_blocks_positive
            num_cleared = len(group)
            self.score += num_cleared
            
            # --- Reward Calculation ---
            reward = num_cleared  # +1 per block
            
            # Combo bonus
            combo_bonus = max(0, num_cleared - self.MIN_GROUP_SIZE)
            reward += combo_bonus
            self.score += combo_bonus
            
            # Risk bonus (if few moves were available before this one)
            if self._count_valid_moves() < 3:
                reward += 2

            # Trigger UI flash
            self.last_combo_info = {"size": num_cleared, "timer": 30}
            
            # Clear blocks and create particles
            for gx, gy in group:
                self._create_particles(gx, gy, self.grid[gy, gx])
                self.grid[gy, gx] = 0
            
            self._apply_gravity()
            return reward
        else:
            # sfx: invalid_move_negative
            return -0.2

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

    def _find_group(self, start_x, start_y):
        target_color = self.grid[start_y, start_x]
        if target_color == 0:
            return []

        q = [(start_x, start_y)]
        visited = set(q)
        group = []

        while q:
            x, y = q.pop(0)
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and \
                   (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return group

    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] != 0:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = 0
                    empty_row -= 1
    
    def _reshuffle_board_inplace(self):
        blocks = self.grid[self.grid > 0].flatten().tolist()
        self.np_random.shuffle(blocks)
        self.grid.fill(0)
        
        idx = 0
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if idx < len(blocks):
                    self.grid[y, x] = blocks[idx]
                    idx += 1
                else:
                    break
            if idx >= len(blocks):
                break

    def _find_any_valid_move(self):
        return self._count_valid_moves(stop_at_one=True) > 0

    def _count_valid_moves(self, stop_at_one=False):
        checked = np.zeros_like(self.grid, dtype=bool)
        count = 0
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] > 0 and not checked[y, x]:
                    group = self._find_group(x, y)
                    for gx, gy in group:
                        checked[gy, gx] = True
                    if len(group) >= self.MIN_GROUP_SIZE:
                        count += 1
                        if stop_at_one:
                            return count
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_background()
        self._update_and_draw_particles()
        self._render_blocks()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_background(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), border_radius=5)
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            x = self.GRID_X + i * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), self.GRID_LINE_WIDTH)
            # Horizontal lines
            y = self.GRID_Y + i * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), self.GRID_LINE_WIDTH)

    def _render_blocks(self):
        shadow_offset = int(self.BLOCK_SIZE * 0.1)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.grid[y, x]
                if color_idx > 0:
                    px = self.GRID_X + self.GRID_LINE_WIDTH + x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
                    py = self.GRID_Y + self.GRID_LINE_WIDTH + y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
                    
                    # Shadow
                    pygame.draw.rect(self.screen, self.BLOCK_SHADOW_COLORS[color_idx-1], (px, py, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=4)
                    # Main color
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_idx-1], (px, py, self.BLOCK_SIZE - shadow_offset, self.BLOCK_SIZE - shadow_offset), border_radius=4)

    def _render_cursor(self):
        x, y = self.cursor_pos
        px = self.GRID_X + self.GRID_LINE_WIDTH + x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        py = self.GRID_Y + self.GRID_LINE_WIDTH + y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        
        cursor_rect = pygame.Rect(px - 3, py - 3, self.BLOCK_SIZE + 6, self.BLOCK_SIZE + 6)
        
        # Pulsing alpha for glow effect
        alpha = 100 + int(math.sin(pygame.time.get_ticks() * 0.005) * 50)
        
        glow_surface = pygame.Surface((self.BLOCK_SIZE + 12, self.BLOCK_SIZE + 12), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, alpha), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, (px - 6, py - 6))

        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        draw_text(moves_text, self.font_medium, self.COLOR_TEXT, (20, 20), self.COLOR_TEXT_SHADOW)

        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        draw_text(score_text, self.font_large, self.COLOR_TEXT, (self.WIDTH - score_surf.get_width() - 20, 20), self.COLOR_TEXT_SHADOW)

        # Combo display
        if self.last_combo_info["timer"] > 0:
            size = self.last_combo_info["size"]
            timer = self.last_combo_info["timer"]
            
            scale = 1 + (timer / 30.0) * 0.5
            alpha = int((timer / 30.0) * 255)
            
            combo_font = pygame.font.Font(None, int(24 * scale))
            color = (*self.BLOCK_COLORS[size % self.NUM_COLORS], alpha)

            combo_text = f"Combo x{size}!"
            text_surf = combo_font.render(combo_text, True, color)
            pos_x = self.WIDTH - text_surf.get_width() - 20
            pos_y = 70
            self.screen.blit(text_surf, (pos_x, pos_y))
            
            self.last_combo_info["timer"] -= 1

    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_X + self.GRID_LINE_WIDTH + grid_x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
        py = self.GRID_Y + self.GRID_LINE_WIDTH + grid_y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_index-1]

        for _ in range(self.PARTICLE_COUNT):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [px, py],
                "vel": vel,
                "life": self.PARTICLE_LIFETIME,
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
            if p["life"] > 0:
                alpha = int(255 * (p["life"] / self.PARTICLE_LIFETIME))
                color = (*p["color"], alpha)
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color
                )
                active_particles.append(p)
        self.particles = active_particles
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "valid_moves": self._count_valid_moves()
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Clear Puzzle")
    
    running = True
    clock = pygame.time.Clock()
    
    # Game loop
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # Only step if an action is taken (for this human-playable demo)
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            if terminated or truncated:
                print("Game Over!")
                obs, info = env.reset()
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(15) # Limit frame rate for human playability
        
    env.close()