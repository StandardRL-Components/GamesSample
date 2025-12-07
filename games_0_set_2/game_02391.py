import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to tilt the cavern and slide the crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Tilt the cavern to slide crystals. Align 3 or more of the same color to score points. Win by creating 20 sets before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.NUM_CRYSTALS_TO_SPAWN = 45
        self.NUM_CRYSTAL_TYPES = 8

        # Game parameters
        self.MAX_MOVES = 30
        self.WIN_SETS = 20
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 40, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 65)
        self.COLOR_UI_BG = (30, 40, 65, 180)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_MOVES_HIGH = (100, 255, 100)
        self.COLOR_MOVES_MID = (255, 255, 100)
        self.COLOR_MOVES_LOW = (255, 100, 100)

        # Crystal Colors (bright, distinct)
        self.CRYSTAL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
            (255, 160, 80),  # Orange
            (200, 120, 255), # Purple
        ]
        
        # Isometric projection parameters
        self.TILE_WIDTH_HALF = 30
        self.TILE_HEIGHT_HALF = 15
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Initialize state variables
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.moves_left = 0
        self.sets_aligned = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particle_effects = []
        self.last_potential_matches = 0
        self.rng = np.random.default_rng()

        # The validation call expects the environment to be in a renderable state.
        # self.reset() is not called before this, so we must initialize `self.grid`
        # to a valid, empty state.
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.moves_left = self.MAX_MOVES
        self.sets_aligned = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particle_effects = []
        
        # Generate initial crystal layout
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        empty_cells = list(range(self.GRID_WIDTH * self.GRID_HEIGHT))
        self.rng.shuffle(empty_cells)
        
        for i in range(min(self.NUM_CRYSTALS_TO_SPAWN, len(empty_cells))):
            cell_idx = empty_cells[i]
            x = cell_idx % self.GRID_WIDTH
            y = cell_idx // self.GRID_WIDTH
            self.grid[y, x] = self.rng.integers(1, self.NUM_CRYSTAL_TYPES + 1)
        
        self.last_potential_matches = self._count_potential_matches()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        movement = action[0]
        reward = 0

        if movement > 0: # 0 is no-op, which does nothing in this turn-based game
            self.moves_left -= 1
            
            # 1. Calculate potential matches before move
            potential_before = self._count_potential_matches()

            # 2. Slide crystals
            self._slide_crystals(movement)

            # 3. Check for and process matches
            num_new_sets, removed_crystals = self._check_and_process_matches()
            if num_new_sets > 0:
                self.sets_aligned += num_new_sets
                reward += num_new_sets * 10
                # Sound effect placeholder: # play_match_sound()
                for x, y, color_idx in removed_crystals:
                    self._create_particles(x, y, self.CRYSTAL_COLORS[color_idx-1])

            # 4. Calculate potential matches after move
            potential_after = self._count_potential_matches()
            
            # 5. Continuous reward for creating/breaking pairs
            reward += (potential_after - potential_before) * 0.1
            self.last_potential_matches = potential_after

        # 6. Check termination conditions
        terminated = False
        if self.sets_aligned >= self.WIN_SETS:
            self.win = True
            terminated = True
            reward += 100 # Win bonus
            # Sound effect placeholder: # play_win_sound()
        elif self.moves_left <= 0:
            terminated = True
            # Sound effect placeholder: # play_lose_sound()
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _slide_crystals(self, direction):
        # direction: 1=up, 2=down, 3=left, 4=right
        if direction == 1: # Up
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if self.grid[y, x] > 0:
                        cy = y
                        while cy > 0 and self.grid[cy - 1, x] == 0:
                            cy -= 1
                        if cy != y:
                            self.grid[cy, x] = self.grid[y, x]
                            self.grid[y, x] = 0
        elif direction == 2: # Down
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT - 1, -1, -1):
                    if self.grid[y, x] > 0:
                        cy = y
                        while cy < self.GRID_HEIGHT - 1 and self.grid[cy + 1, x] == 0:
                            cy += 1
                        if cy != y:
                            self.grid[cy, x] = self.grid[y, x]
                            self.grid[y, x] = 0
        elif direction == 3: # Left
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y, x] > 0:
                        cx = x
                        while cx > 0 and self.grid[y, cx - 1] == 0:
                            cx -= 1
                        if cx != x:
                            self.grid[y, cx] = self.grid[y, x]
                            self.grid[y, x] = 0
        elif direction == 4: # Right
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH - 1, -1, -1):
                    if self.grid[y, x] > 0:
                        cx = x
                        while cx < self.GRID_WIDTH - 1 and self.grid[y, cx + 1] == 0:
                            cx += 1
                        if cx != x:
                            self.grid[y, cx] = self.grid[y, x]
                            self.grid[y, x] = 0

    def _check_and_process_matches(self):
        to_remove = set()
        visited = np.zeros_like(self.grid, dtype=bool)
        num_sets = 0
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0 and not visited[y, x]:
                    color = self.grid[y, x]
                    component = set()
                    q = deque([(x, y)])
                    visited[y, x] = True
                    
                    while q:
                        cx, cy = q.popleft()
                        component.add((cx, cy))
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                               not visited[ny, nx] and self.grid[ny, nx] == color:
                                visited[ny, nx] = True
                                q.append((nx, ny))
                    
                    if len(component) >= 3:
                        num_sets += 1
                        to_remove.update(component)

        removed_crystals_info = []
        if to_remove:
            for x, y in to_remove:
                removed_crystals_info.append((x, y, self.grid[y, x]))
                self.grid[y, x] = 0
        
        return num_sets, removed_crystals_info
    
    def _count_potential_matches(self):
        pairs = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y, x]
                if color > 0:
                    # Check right neighbor
                    if x + 1 < self.GRID_WIDTH and self.grid[y, x + 1] == color:
                        pairs += 1
                    # Check down neighbor
                    if y + 1 < self.GRID_HEIGHT and self.grid[y + 1, x] == color:
                        pairs += 1
        return pairs

    def _cart_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, grid_x, grid_y, color):
        x, y = self._cart_to_iso(grid_x, grid_y)
        
        # Make colors for faces
        top_color = color
        side_color1 = tuple(max(0, c - 40) for c in color)
        side_color2 = tuple(max(0, c - 80) for c in color)

        h = self.TILE_HEIGHT_HALF * 2

        # Points for the cube
        p = [
            (x, y - h),                                   # Top center
            (x - self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF), # Top left
            (x, y - h + self.TILE_HEIGHT_HALF * 2),       # Top bottom
            (x + self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF), # Top right
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),     # Bottom left
            (x, y + self.TILE_HEIGHT_HALF * 2),           # Bottom center
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),     # Bottom right
        ]

        # Draw faces
        # Top face
        pygame.gfxdraw.filled_polygon(surface, [p[0], p[1], p[2], p[3]], top_color)
        pygame.gfxdraw.aapolygon(surface, [p[0], p[1], p[2], p[3]], top_color)
        
        # Left face
        pygame.gfxdraw.filled_polygon(surface, [p[1], p[4], p[5], p[2]], side_color1)
        pygame.gfxdraw.aapolygon(surface, [p[1], p[4], p[5], p[2]], side_color1)
        
        # Right face
        pygame.gfxdraw.filled_polygon(surface, [p[3], p[6], p[5], p[2]], side_color2)
        pygame.gfxdraw.aapolygon(surface, [p[3], p[6], p[5], p[2]], side_color2)

    def _create_particles(self, grid_x, grid_y, color):
        iso_x, iso_y = self._cart_to_iso(grid_x, grid_y)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particle_effects.append({'pos': [iso_x, iso_y], 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particle_effects[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particle_effects.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._cart_to_iso(0, y)
            end = self._cart_to_iso(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._cart_to_iso(x, 0)
            end = self._cart_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        
        # Draw crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx > 0:
                    self._draw_iso_cube(self.screen, x, y, self.CRYSTAL_COLORS[color_idx-1])

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((180, 80), pygame.SRCALPHA)
        pygame.draw.rect(ui_panel, self.COLOR_UI_BG, ui_panel.get_rect(), border_radius=10)
        
        # Score
        score_text = self.font_ui.render(f"SETS: {self.sets_aligned}/{self.WIN_SETS}", True, self.COLOR_UI_TEXT)
        ui_panel.blit(score_text, (15, 15))

        # Moves
        if self.moves_left > self.MAX_MOVES * 0.5:
            moves_color = self.COLOR_MOVES_HIGH
        elif self.moves_left > self.MAX_MOVES * 0.2:
            moves_color = self.COLOR_MOVES_MID
        else:
            moves_color = self.COLOR_MOVES_LOW
        
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, moves_color)
        ui_panel.blit(moves_text, (15, 45))
        
        self.screen.blit(ui_panel, (10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_MOVES_HIGH if self.win else self.COLOR_MOVES_LOW
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            # Draw a shadow/outline
            shadow_surf = self.font_game_over.render(msg, True, (0,0,0))
            self.screen.blit(shadow_surf, text_rect.move(3,3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "sets_aligned": self.sets_aligned,
            "moves_left": self.moves_left,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # For manual play
    import sys
    # The main script will set the driver, but for local testing, we might need to unset it.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()