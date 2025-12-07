import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:04:18.529257
# Source Brief: brief_00794.md
# Brief Index: 794
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player is a miner in a cave.
    The goal is to match colored gems to gather resources, use those resources
    to build platforms to explore, and uncover hidden musical instruments.
    The environment is designed with a focus on visual quality and game feel.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Explore a mysterious cave, match colored gems to gather resources, and "
        "build platforms to uncover hidden musical instruments."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to match gems or build a platform. "
        "Press shift to cycle through platform types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    TILE_SIZE = 20
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * TILE_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * TILE_SIZE) // 2

    # Tile Types
    T_EMPTY, T_ROCK = 0, 1
    T_GEM_RED, T_GEM_GREEN, T_GEM_BLUE, T_GEM_YELLOW = 2, 3, 4, 5
    T_PLAT_RED, T_PLAT_GREEN, T_PLAT_BLUE, T_PLAT_YELLOW = 6, 7, 8, 9

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_ROCK = (60, 50, 70)
    COLOR_ROCK_ACCENT = (75, 65, 85)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_GOLD = (255, 215, 0)
    COLOR_GOLD_DARK = (200, 160, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 35, 55, 180)

    GEM_COLORS = {
        T_GEM_RED: (255, 80, 80),
        T_GEM_GREEN: (80, 255, 80),
        T_GEM_BLUE: (80, 120, 255),
        T_GEM_YELLOW: (255, 255, 80),
    }
    PLATFORM_COLORS = {
        T_PLAT_RED: (180, 40, 40),
        T_PLAT_GREEN: (40, 180, 40),
        T_PLAT_BLUE: (40, 80, 180),
        T_PLAT_YELLOW: (180, 180, 40),
    }
    PLATFORM_GEM_MAP = {
        T_PLAT_RED: T_GEM_RED, T_PLAT_GREEN: T_GEM_GREEN,
        T_PLAT_BLUE: T_GEM_BLUE, T_PLAT_YELLOW: T_GEM_YELLOW
    }
    COLOR_ORDER = [T_PLAT_RED, T_PLAT_GREEN, T_PLAT_BLUE, T_PLAT_YELLOW]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Initialize state variables
        self.grid = None
        self.player_pos = None
        self.player_visual_pos = None
        self.instruments = None
        self.platform_resources = None
        self.selected_platform_idx = None
        self.moves_left = None
        self.biome_level = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None
        self.step_reward = 0.0

        # Use a numpy random number generator
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.biome_level = 1
        self.moves_left = 50
        self.platform_resources = defaultdict(int)
        self.selected_platform_idx = 0
        self.particles = []
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        self._generate_biome()
        self.player_visual_pos = [self.player_pos[0] * self.TILE_SIZE, self.player_pos[1] * self.TILE_SIZE]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        action_taken = movement != 0 or space_pressed or shift_pressed
        if action_taken:
            self.moves_left -= 1
        
        self._handle_movement(movement)
        self._handle_actions(space_pressed, shift_pressed)
        self._apply_gravity()
        self._update_instrument_visibility()
        
        self.steps += 1
        reward = self.step_reward
        terminated = self._check_termination()
        truncated = self.steps >= 1000
        
        if terminated and not truncated:
            if self.moves_left <= 0 and not self._all_instruments_found():
                reward -= 100 # Penalty for losing
            elif self._all_instruments_found():
                reward += 100 # Bonus for winning

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._update_visuals()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "biome_level": self.biome_level,
            "instruments_found": sum(1 for _, _, found in self.instruments if found),
            "platform_resources": dict(self.platform_resources)
        }

    # --- Game Logic ---

    def _generate_biome(self):
        # 1. Cellular Automata for cave generation
        self.grid = self.np_random.choice([self.T_EMPTY, self.T_ROCK], 
                                     size=(self.GRID_WIDTH, self.GRID_HEIGHT), 
                                     p=[0.55, 0.45])

        for _ in range(4):
            new_grid = self.grid.copy()
            for x in range(1, self.GRID_WIDTH - 1):
                for y in range(1, self.GRID_HEIGHT - 1):
                    neighbors = np.sum(self.grid[x-1:x+2, y-1:y+2] == self.T_ROCK)
                    if neighbors > 4: new_grid[x, y] = self.T_ROCK
                    elif neighbors < 4: new_grid[x, y] = self.T_EMPTY
            self.grid = new_grid
        
        # 2. Ensure borders are rock
        self.grid[0, :], self.grid[-1, :], self.grid[:, 0], self.grid[:, -1] = self.T_ROCK, self.T_ROCK, self.T_ROCK, self.T_ROCK

        # 3. Find largest cavern and place player
        empty_tiles = list(zip(*np.where(self.grid == self.T_EMPTY)))
        if not empty_tiles: 
            self.reset() # Failsafe
            return
        
        start_node = empty_tiles[self.np_random.integers(len(empty_tiles))]
        q = [start_node]
        visited = {start_node}
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == self.T_EMPTY:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        for x, y in empty_tiles:
            if (x, y) not in visited:
                self.grid[x, y] = self.T_ROCK
        
        self.player_pos = list(list(visited)[self.np_random.integers(len(visited))])

        # 4. Place instruments
        self.instruments = []
        num_instruments = 2 + (self.biome_level - 1)
        rock_tiles = list(zip(*np.where(self.grid == self.T_ROCK)))
        self.np_random.shuffle(rock_tiles)
        for x, y in rock_tiles:
            if len(self.instruments) >= num_instruments: break
            # Must be fully enclosed by rock
            if 1 < x < self.GRID_WIDTH - 2 and 1 < y < self.GRID_HEIGHT - 2:
                neighbors = self.grid[x-1:x+2, y-1:y+2]
                if np.all(neighbors == self.T_ROCK):
                    self.instruments.append([x, y, False])
                    # Carve out space around it
                    self.grid[x-1:x+2, y-1:y+2] = self.T_ROCK
                    self.grid[x, y] = self.T_EMPTY

        # 5. Place gems
        empty_tiles = list(zip(*np.where(self.grid == self.T_EMPTY)))
        self.np_random.shuffle(empty_tiles)
        for i in range(min(len(empty_tiles) // 3, 100)):
            x, y = empty_tiles[i]
            if [x, y] != self.player_pos:
                self.grid[x, y] = self.np_random.choice(list(self.GEM_COLORS.keys()))
        
        self._apply_gravity(instant=True)

    def _handle_movement(self, movement):
        if movement == 0: return
        
        self.step_reward -= 0.1
        px, py = self.player_pos
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right

        if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
            tile = self.grid[px, py]
            if tile != self.T_ROCK:
                self.player_pos = [px, py]

    def _handle_actions(self, space_pressed, shift_pressed):
        if shift_pressed:
            self.selected_platform_idx = (self.selected_platform_idx + 1) % len(self.COLOR_ORDER)
            # SFX: UI_Cycle.wav

        if space_pressed:
            # Action 1: Match gems
            matched_anything = self._attempt_match()
            if matched_anything:
                return
            
            # Action 2: Place platform
            self._attempt_place_platform()

    def _attempt_match(self):
        px, py = self.player_pos
        adj_gems = defaultdict(list)
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                tile = self.grid[nx, ny]
                if tile in self.GEM_COLORS:
                    adj_gems[tile].append((nx, ny))
        
        gems_matched_count = 0
        for gem_type, positions in adj_gems.items():
            if len(positions) >= 2:
                for x, y in positions:
                    self.grid[x, y] = self.T_EMPTY
                    self.platform_resources[gem_type] += 1
                    gems_matched_count += 1
                    self.step_reward += 1.0
                    self._add_particles(x, y, self.GEM_COLORS[gem_type], 10)
        
        if gems_matched_count > 0:
            # SFX: Gem_Match.wav
            return True
        return False

    def _attempt_place_platform(self):
        px, py = self.player_pos
        if self.grid[px, py] == self.T_EMPTY:
            plat_type = self.COLOR_ORDER[self.selected_platform_idx]
            gem_type = self.PLATFORM_GEM_MAP[plat_type]
            if self.platform_resources[gem_type] > 0:
                self.platform_resources[gem_type] -= 1
                self.grid[px, py] = plat_type
                # SFX: Place_Platform.wav

    def _apply_gravity(self, instant=False):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2, -1, -1):
                if self.grid[x, y] in self.GEM_COLORS and self.grid[x, y+1] == self.T_EMPTY:
                    # Find how far it can fall
                    fall_to = y + 1
                    while fall_to < self.GRID_HEIGHT -1 and self.grid[x, fall_to + 1] == self.T_EMPTY:
                        fall_to += 1
                    
                    # Move gem
                    self.grid[x, fall_to] = self.grid[x, y]
                    self.grid[x, y] = self.T_EMPTY
                    # SFX: Gem_Fall.wav (if not instant)

    def _update_instrument_visibility(self):
        for i, (ix, iy, found) in enumerate(self.instruments):
            if not found:
                is_excavated = True
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = ix + dx, iy + dy
                        if self.grid[nx, ny] == self.T_ROCK:
                            is_excavated = False
                            break
                    if not is_excavated: break
                
                if is_excavated:
                    self.instruments[i][2] = True
                    self.step_reward += 5.0
                    self._add_particles(ix, iy, self.COLOR_GOLD, 50, life=60)
                    # SFX: Instrument_Reveal.wav

    def _all_instruments_found(self):
        return all(found for _, _, found in self.instruments)

    def _check_termination(self):
        if self.game_over: return True
        
        if self.moves_left <= 0:
            self.game_over = True
            return True
        if self._all_instruments_found():
            self.game_over = True
            # SFX: Biome_Complete.wav
            return True
        
        return False
    
    # --- Rendering ---

    def _update_visuals(self):
        # Interpolate player position
        target_x = self.player_pos[0] * self.TILE_SIZE
        target_y = self.player_pos[1] * self.TILE_SIZE
        self.player_visual_pos[0] += (target_x - self.player_visual_pos[0]) * 0.5
        self.player_visual_pos[1] += (target_y - self.player_visual_pos[1]) * 0.5

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += p['grav']
            p['life'] -= 1

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE))
        grid_surface.fill(self.COLOR_BG)
        
        self._draw_grid(grid_surface)
        self._draw_instruments(grid_surface)
        self._draw_player(grid_surface)
        
        self.screen.blit(grid_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))
        self._update_and_draw_particles() # Draw particles on top of main screen

    def _draw_grid(self, surface):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                tile = self.grid[x, y]
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                if tile == self.T_ROCK:
                    pygame.draw.rect(surface, self.COLOR_ROCK, rect)
                    if (x + y) % 2 == 0: # Add simple texture
                         pygame.draw.rect(surface, self.COLOR_ROCK_ACCENT, rect.inflate(-self.TILE_SIZE//2, -self.TILE_SIZE//2))
                elif tile in self.GEM_COLORS:
                    color = self.GEM_COLORS[tile]
                    pygame.gfxdraw.box(surface, rect, (*color, 50))
                    pygame.gfxdraw.filled_circle(surface, rect.centerx, rect.centery, self.TILE_SIZE // 3, color)
                    pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, self.TILE_SIZE // 3, color)
                elif tile in self.PLATFORM_COLORS:
                    color = self.PLATFORM_COLORS[tile]
                    pygame.draw.rect(surface, color, rect)
                    pygame.draw.line(surface, tuple(min(255, c+40) for c in color), rect.topleft, rect.topright, 2)
                    pygame.draw.line(surface, tuple(max(0, c-40) for c in color), rect.bottomleft, rect.bottomright, 2)

    def _draw_instruments(self, surface):
        for x, y, found in self.instruments:
            pos_x, pos_y = int(x * self.TILE_SIZE + self.TILE_SIZE / 2), int(y * self.TILE_SIZE + self.TILE_SIZE / 2)
            if found:
                pygame.gfxdraw.filled_circle(surface, pos_x, pos_y, self.TILE_SIZE // 2, self.COLOR_GOLD)
                pygame.gfxdraw.aacircle(surface, pos_x, pos_y, self.TILE_SIZE // 2, self.COLOR_GOLD)
            else: # Draw outline if not found
                pygame.gfxdraw.aacircle(surface, pos_x, pos_y, self.TILE_SIZE // 2, (*self.COLOR_GOLD_DARK, 100))

    def _draw_player(self, surface):
        px, py = self.player_visual_pos
        center_x, center_y = int(px + self.TILE_SIZE/2), int(py + self.TILE_SIZE/2)
        
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.8)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player sprite
        player_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(surface, self.COLOR_PLAYER, player_rect.inflate(-4, -4))

    def _render_ui(self):
        # Moves left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Biome
        biome_text = self.font_large.render(f"Biome: {self.biome_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(biome_text, (self.SCREEN_WIDTH - biome_text.get_width() - 10, 10))

        # Resources and selected platform
        ui_box = pygame.Surface((180, 50), pygame.SRCALPHA)
        ui_box.fill(self.COLOR_UI_BG)
        
        for i, plat_type in enumerate(self.COLOR_ORDER):
            gem_type = self.PLATFORM_GEM_MAP[plat_type]
            color = self.GEM_COLORS[gem_type]
            count = self.platform_resources[gem_type]
            
            x_pos = 10 + i * 40
            pygame.draw.circle(ui_box, color, (x_pos + 10, 15), 8)
            count_text = self.font_small.render(f"{count}", True, self.COLOR_UI_TEXT)
            ui_box.blit(count_text, (x_pos, 30))

            if i == self.selected_platform_idx:
                pygame.draw.rect(ui_box, self.COLOR_PLAYER, (x_pos-2, 3, 24, 24), 2, border_radius=4)
        
        self.screen.blit(ui_box, (10, self.SCREEN_HEIGHT - 60))

    # --- Particles ---
    def _add_particles(self, x, y, color, count, life=30):
        gx, gy = x * self.TILE_SIZE + self.GRID_OFFSET_X, y * self.TILE_SIZE + self.GRID_OFFSET_Y
        for _ in range(count):
            self.particles.append({
                'x': gx + self.TILE_SIZE / 2,
                'y': gy + self.TILE_SIZE / 2,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-3, -1),
                'grav': 0.1,
                'life': self.np_random.integers(life // 2, life + 1),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 30)))
            color = (*p['color'], alpha)
            pygame.gfxdraw.box(self.screen, (int(p['x']), int(p['y']), p['size'], p['size']), color)
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Excavator")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        if terminated or truncated:
            font = pygame.font.SysFont("monospace", 50, bold=True)
            text_str = "GAME OVER" if terminated else "TRUNCATED"
            text = font.render(text_str, True, (255, 50, 50))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2))
            screen.blit(text, text_rect)
            
        pygame.display.flip()
        
        if (terminated or truncated) and keys[pygame.K_r]:
            print(f"Episode finished. Total Reward: {total_reward}")
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

        clock.tick(env.metadata["render_fps"])

    env.close()