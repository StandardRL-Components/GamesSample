import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:00:36.155246
# Source Brief: brief_00820.md
# Brief Index: 820
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A single particle for visual effects, like explosions or sparks."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 6)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(20, 40)
        self.initial_life = self.life
        self.size = random.uniform(3, 7)

    def update(self):
        """Update particle state for one frame."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity effect
        self.life -= 1
        self.size = max(0, self.size * 0.97)

    def draw(self, surface):
        """Draw the particle on the given Pygame surface."""
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
            # Use additive blending for a glow effect
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Place colored hex tiles to create matching groups. Clear the board by making chains, but don't let the grid fill up!"
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to place the next tile."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_TILES = 7
        self.MAX_TILES = 20
        self.CHILD_SPAWN_CHANCE = 0.2

        # --- Hex Grid Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 9
        self.HEX_SIZE = 20
        self.HEX_OFFSET_X = 50
        self.HEX_OFFSET_Y = 50

        # --- Visuals ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (80, 255, 255),  # Cyan
        ]
        self.CURSOR_COLOR = (255, 255, 255)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.grid = None
        self.tiles_on_board = None
        self.cursor_hex_pos = None
        self.cursor_pixel_pos = None
        self.next_tile_color_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.particles = []
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[-1 for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        self.tiles_on_board = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        
        self.cursor_hex_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.cursor_pixel_pos = self._hex_to_pixel(*self.cursor_hex_pos)

        for _ in range(self.INITIAL_TILES):
            self._place_random_tile()
            
        self.next_tile_color_idx = self.np_random.integers(0, len(self.TILE_COLORS))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        self._handle_movement(movement)
        
        placement_triggered = space_held and not self.prev_space_held
        if placement_triggered:
            # sfx: tile_place.wav
            placement_reward = self._handle_placement()
            reward += placement_reward

        self.prev_space_held = space_held
        
        terminated = False
        truncated = False
        if self.tiles_on_board == 0 and self.steps > 1:
            # sfx: win_fanfare.wav
            reward += 100
            terminated = True
            self.game_over = True
        elif self.tiles_on_board >= self.MAX_TILES:
            # sfx: game_over_lose.wav
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        q, r = self.cursor_hex_pos
        if movement == 1: r -= 1 # Up
        elif movement == 2: r += 1 # Down
        elif movement == 3: q -= 1 # Left
        elif movement == 4: q += 1 # Right
        
        q = max(0, min(self.GRID_WIDTH - 1, q))
        r = max(0, min(self.GRID_HEIGHT - 1, r))
        self.cursor_hex_pos = (q, r)

    def _handle_placement(self):
        q, r = self.cursor_hex_pos
        if self.grid[q][r] == -1:
            self.grid[q][r] = self.next_tile_color_idx
            self.tiles_on_board += 1
            
            num_cleared, cleared_tiles_info = self._resolve_matches(q, r)
            
            if num_cleared > 0:
                # sfx: chain_reaction_pop.wav
                self._create_particles(cleared_tiles_info)
                for cq, cr, color_idx in cleared_tiles_info:
                    if self.np_random.random() < self.CHILD_SPAWN_CHANCE:
                        self._spawn_child_tile(color_idx, cq, cr)
            
            self.next_tile_color_idx = self.np_random.integers(0, len(self.TILE_COLORS))
            return num_cleared * 0.1
        return 0 # Invalid placement

    def _resolve_matches(self, q, r):
        color_idx = self.grid[q][r]
        if color_idx == -1: return 0, []

        component = self._find_connected_component(q, r, color_idx)
        
        if len(component) > 1:
            cleared_info = []
            for cq, cr in component:
                cleared_info.append((cq, cr, self.grid[cq][cr]))
                self.grid[cq][cr] = -1
            
            self.tiles_on_board -= len(component)
            self.score += len(component) ** 2 # Bonus for larger chains
            return len(component), cleared_info
        
        return 0, []

    def _find_connected_component(self, start_q, start_r, color_idx):
        q = [(start_q, start_r)]
        visited = set([(start_q, start_r)])
        component = []
        
        while q:
            curr_q, curr_r = q.pop(0)
            component.append((curr_q, curr_r))
            
            for nq, nr in self._get_neighbors(curr_q, curr_r):
                if (nq, nr) not in visited and self.grid[nq][nr] == color_idx:
                    visited.add((nq, nr))
                    q.append((nq, nr))
        return component

    def _get_neighbors(self, q, r):
        neighbors = []
        # Odd-q vertical layout neighbor logic
        if q % 2 == 1: # Odd columns
            potential_neighbors = [(q, r-1), (q, r+1), (q-1, r), (q+1, r), (q-1, r+1), (q+1, r+1)]
        else: # Even columns
            potential_neighbors = [(q, r-1), (q, r+1), (q-1, r-1), (q+1, r-1), (q-1, r), (q+1, r)]
        
        for nq, nr in potential_neighbors:
            if 0 <= nq < self.GRID_WIDTH and 0 <= nr < self.GRID_HEIGHT:
                neighbors.append((nq, nr))
        return neighbors

    def _place_random_tile(self):
        empty_cells = []
        for q in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[q][r] == -1:
                    empty_cells.append((q, r))
        
        if empty_cells:
            q, r = random.choice(empty_cells)
            self.grid[q][r] = self.np_random.integers(0, len(self.TILE_COLORS))
            self.tiles_on_board += 1

    def _spawn_child_tile(self, color_idx, origin_q, origin_r):
        empty_neighbors = [n for n in self._get_neighbors(origin_q, origin_r) if self.grid[n[0]][n[1]] == -1]
        if empty_neighbors:
            # sfx: child_spawn.wav
            q, r = random.choice(empty_neighbors)
            self.grid[q][r] = color_idx
            self.tiles_on_board += 1

    def _create_particles(self, cleared_tiles_info):
        for q, r, color_idx in cleared_tiles_info:
            px, py = self._hex_to_pixel(q, r)
            color = self.TILE_COLORS[color_idx]
            for _ in range(15): # Number of particles per tile
                self.particles.append(Particle(px, py, color))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._draw_grid_background()
        self._draw_tiles()
        self._update_and_draw_particles()
        self._draw_cursor()

    def _draw_grid_background(self):
        for q in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                center = self._hex_to_pixel(q, r)
                self._draw_hexagon(self.screen, self.COLOR_GRID, center, self.HEX_SIZE, border_width=1)

    def _draw_tiles(self):
        for q in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                color_idx = self.grid[q][r]
                if color_idx != -1:
                    center = self._hex_to_pixel(q, r)
                    color = self.TILE_COLORS[color_idx]
                    self._draw_hexagon(self.screen, color, center, self.HEX_SIZE, filled=True)
                    self._draw_hexagon(self.screen, tuple(c*0.7 for c in color), center, self.HEX_SIZE, border_width=2)


    def _update_and_draw_particles(self):
        for p in self.particles:
            p.update()
            p.draw(self.screen)
        self.particles = [p for p in self.particles if p.life > 0]

    def _draw_cursor(self):
        target_pixel_pos = self._hex_to_pixel(*self.cursor_hex_pos)
        # Interpolate for smooth movement
        dx = target_pixel_pos[0] - self.cursor_pixel_pos[0]
        dy = target_pixel_pos[1] - self.cursor_pixel_pos[1]
        self.cursor_pixel_pos = (self.cursor_pixel_pos[0] + dx * 0.5, self.cursor_pixel_pos[1] + dy * 0.5)
        
        # Pulsing glow effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        glow_size = self.HEX_SIZE + 3 + pulse * 3
        glow_alpha = 100 + pulse * 100
        
        self._draw_hexagon(self.screen, self.CURSOR_COLOR, self.cursor_pixel_pos, glow_size, alpha=glow_alpha, filled=False, border_width=3)
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Tile Count
        tile_count_text = self.font_ui.render(f"TILES: {self.tiles_on_board}/{self.MAX_TILES}", True, self.COLOR_UI_TEXT)
        text_rect = tile_count_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(tile_count_text, text_rect)
        
        # Next Tile Preview
        preview_pos = (self.WIDTH - 50, self.HEIGHT - 50)
        preview_color = self.TILE_COLORS[self.next_tile_color_idx]
        self._draw_hexagon(self.screen, preview_color, preview_pos, self.HEX_SIZE * 1.2, filled=True)
        self._draw_hexagon(self.screen, tuple(c*0.7 for c in preview_color), preview_pos, self.HEX_SIZE * 1.2, border_width=2)
        next_text = self.font_ui.render("NEXT", True, self.COLOR_UI_TEXT)
        next_rect = next_text.get_rect(center=(preview_pos[0], preview_pos[1] - 45))
        self.screen.blit(next_text, next_rect)
        
    def _hex_to_pixel(self, q, r):
        # Pointy-top, odd-q
        x = self.HEX_SIZE * 3/2 * q + self.HEX_OFFSET_X
        y = self.HEX_SIZE * math.sqrt(3) * (r + 0.5 * (q % 2)) + self.HEX_OFFSET_Y
        return int(x), int(y)

    def _draw_hexagon(self, surface, color, center, size, filled=False, border_width=0, alpha=255):
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + size * math.cos(angle_rad),
                           center[1] + size * math.sin(angle_rad)))
        
        points_int = [(int(p[0]), int(p[1])) for p in points]

        if alpha < 255:
            target_rect = pygame.Rect(center[0]-size, center[1]-size, size*2, size*2)
            temp_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            points_local = [(p[0]-target_rect.left, p[1]-target_rect.top) for p in points_int]
            
            if filled:
                pygame.gfxdraw.filled_polygon(temp_surf, points_local, (*color, alpha))
            if border_width > 0:
                pygame.draw.aalines(temp_surf, (*color, alpha), True, points_local, True)
            surface.blit(temp_surf, target_rect)
        else:
            if filled:
                pygame.gfxdraw.filled_polygon(surface, points_int, color)
            if border_width > 0:
                pygame.gfxdraw.aapolygon(surface, points_int, color)
                if border_width > 1: # Thicken line
                     pygame.draw.lines(surface, color, True, points_int, border_width)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tiles_on_board": self.tiles_on_board,
        }

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
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example of how to run the environment ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    # This part requires a display. If you run this headless, this will fail.
    # To run headless, just interact with the env directly without this block.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Hex Chain Reaction")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Human Controls ---
            movement = 0 # No-op
            space_held = 0
            shift_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- RESET ---")
                    obs, info = env.reset()
                    total_reward = 0

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward}")
                # In a real training loop, you would reset here.
                # For human play, we can just observe the end state.

            # --- Rendering ---
            # The observation is already a rendered frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Limit to 30 FPS for human play
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        env.validate_implementation()
        
    env.close()