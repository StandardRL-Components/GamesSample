
# Generated: 2025-08-27T20:37:18.401324
# Source Brief: brief_02524.md
# Brief Index: 2524

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select and remove a crystal cluster."
    )

    game_description = (
        "An isometric puzzle game. Collect crystals by clicking on matching clusters to maximize your score before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    WIN_SCORE = 50
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_GRID = (45, 40, 55)
    
    CRYSTAL_COLORS = {
        1: {'base': (255, 80, 80), 'light': (255, 150, 150), 'shadow': (180, 50, 50)}, # Red
        2: {'base': (80, 255, 80), 'light': (150, 255, 150), 'shadow': (50, 180, 50)}, # Green
        3: {'base': (80, 120, 255), 'light': (150, 180, 255), 'shadow': (50, 80, 180)}, # Blue
    }
    CRYSTAL_POINTS = {1: 1, 2: 2, 3: 3}
    
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width, self.screen_height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.tile_width = 32
        self.tile_height = 16
        self.origin_x = self.screen_width // 2
        self.origin_y = 100

        self.grid = None
        self.cursor_pos = None
        self.steps = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win_message = ""
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self._generate_valid_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        self._update_particles()

        if not self.game_over:
            # Handle cursor movement
            if movement == 1: self.cursor_pos[0] -= 1  # Up
            elif movement == 2: self.cursor_pos[0] += 1  # Down
            elif movement == 3: self.cursor_pos[1] -= 1  # Left
            elif movement == 4: self.cursor_pos[1] += 1  # Right
            
            self.cursor_pos[0] %= self.GRID_HEIGHT
            self.cursor_pos[1] %= self.GRID_WIDTH

            # Handle selection
            if space_pressed:
                reward += self._process_selection()

        self.steps += 1

        # Check for termination conditions
        if not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100
                self.win_message = "YOU WIN!"
            elif self.moves_left <= 0:
                self.game_over = True
                terminated = True
                reward -= 10
                self.win_message = "GAME OVER"
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if not self.win_message:
                self.win_message = "TIME UP"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _process_selection(self):
        r, c = self.cursor_pos
        if self.grid[r][c] == 0:
            return 0.0 # Clicked on empty space

        cluster = self._find_cluster(r, c)
        
        if len(cluster) < 2:
            return 0.0 # Don't allow breaking single crystals

        self.moves_left -= 1
        
        points_this_turn = 0
        reward = 0.0

        for r_cluster, c_cluster in cluster:
            crystal_type = self.grid[r_cluster][c_cluster]
            points_this_turn += self.CRYSTAL_POINTS[crystal_type]
            
            screen_pos = self._iso_to_screen(r_cluster, c_cluster)
            self._create_particles(screen_pos, self.CRYSTAL_COLORS[crystal_type]['base'], 10)
            
            self.grid[r_cluster][c_cluster] = 0
            # sfx: crystal break sound

        self.score += points_this_turn
        reward += len(cluster) * 0.1
        if len(cluster) >= 3:
            reward += 1.0

        self._apply_gravity_and_refill()
        return reward

    def _generate_valid_grid(self):
        while True:
            self.grid = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH)).tolist()
            if self._count_potential_moves(self.grid) >= self.MAX_MOVES:
                break

    def _count_potential_moves(self, grid):
        count = 0
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in visited:
                    cluster = self._find_cluster(r, c, grid)
                    if len(cluster) > 1:
                        count += 1
                    for pos in cluster:
                        visited.add(pos)
        return count

    def _find_cluster(self, r_start, c_start, grid=None):
        if grid is None:
            grid = self.grid

        target_color = grid[r_start][c_start]
        if target_color == 0:
            return []

        q = [(r_start, c_start)]
        cluster = set(q)
        
        while q:
            r, c = q.pop(0)
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and (nr, nc) not in cluster:
                    if grid[nr][nc] == target_color:
                        cluster.add((nr, nc))
                        q.append((nr, nc))
        return list(cluster)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r][c] != 0:
                    self.grid[empty_row][c], self.grid[r][c] = self.grid[r][c], self.grid[empty_row][c]
                    empty_row -= 1
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] == 0:
                    self.grid[r][c] = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _iso_to_screen(self, r, c):
        x = self.origin_x + (c - r) * self.tile_width / 2
        y = self.origin_y + (c + r) * self.tile_height / 2
        return int(x), int(y)

    def _render_game(self):
        self._render_grid_lines()
        self._render_crystals()
        self._render_cursor()
        self._render_particles()

    def _render_grid_lines(self):
        for r in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(r, 0)
            end = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for c in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(0, c)
            end = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_crystals(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[r][c]
                if crystal_type != 0:
                    self._draw_iso_cube(r, c, self.CRYSTAL_COLORS[crystal_type])

    def _draw_iso_cube(self, r, c, colors):
        x, y = self._iso_to_screen(r, c)
        w, h = self.tile_width, self.tile_height
        
        # Points for the cube
        p_top = (x, y - h)
        p_mid = (x, y)
        p_left = (x - w / 2, y - h / 2)
        p_right = (x + w / 2, y - h / 2)
        p_bottom_left = (x - w / 2, y + h / 2)
        p_bottom_right = (x + w / 2, y + h / 2)
        p_bottom = (x, y + h)

        # Draw faces with antialiasing
        # Top face
        top_points = [p_top, p_right, p_mid, p_left]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, colors['light'])
        pygame.gfxdraw.aapolygon(self.screen, top_points, colors['light'])
        
        # Left face
        left_points = [p_left, p_mid, p_bottom, p_bottom_left]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, colors['base'])
        pygame.gfxdraw.aapolygon(self.screen, left_points, colors['base'])
        
        # Right face
        right_points = [p_right, p_mid, p_bottom, p_bottom_right]
        pygame.gfxdraw.filled_polygon(self.screen, right_points, colors['shadow'])
        pygame.gfxdraw.aapolygon(self.screen, right_points, colors['shadow'])
    
    def _render_cursor(self):
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        w, h = self.tile_width, self.tile_height
        
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y)
        ]
        
        # Pulsing glow effect
        glow_alpha = (math.sin(self.steps * 0.2) + 1) / 2 * 150 + 50
        glow_color = (*self.COLOR_CURSOR, int(glow_alpha))
        
        # Create a temporary surface for the glow
        temp_surface = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
        glow_points = [(p[0] - x + w, p[1] - y + h) for p in points]
        
        pygame.draw.polygon(temp_surface, glow_color, glow_points, 0)
        
        # Scale up for a blurred effect
        temp_surface = pygame.transform.smoothscale(temp_surface, (int(w*2.5), int(h*2.5)))
        self.screen.blit(temp_surface, (x - w*1.25, y - h*1.25))

        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = max(0, p['lifetime'] / 6)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(size))

    def _render_ui(self):
        # Score
        self._draw_text(f"Score: {self.score}", (20, 20), self.font_medium)
        
        # Moves
        moves_text = f"Moves: {self.moves_left}"
        text_width = self.font_medium.size(moves_text)[0]
        self._draw_text(moves_text, (self.screen_width - text_width - 20, 20), self.font_medium)
        
        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, (0, 0))
            self._draw_text(self.win_message, (self.screen_width // 2, self.screen_height // 2 - 30), self.font_large, center=True)
            self._draw_text("Press 'R' to Reset", (self.screen_width // 2, self.screen_height // 2 + 20), self.font_small, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surface = font.render(text, True, color)
        text_shadow = font.render(text, True, shadow_color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Collector")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    while running:
        movement = 0
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        
        action = [movement, space_pressed, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()