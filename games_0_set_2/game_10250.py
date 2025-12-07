import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:10:12.651922
# Source Brief: brief_00250.md
# Brief Index: 250
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


class GameEnv(gym.Env):
    """
    Gymnasium environment for a dream-themed puzzle/action game.
    The player must match symbols on a grid to craft potions, which shrink
    the detection radius of a pursuing Nightmare creature. The goal is to
    reach the exit before being caught or running out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match symbols on a grid to craft potions, which shrink the detection radius of a pursuing Nightmare. "
        "Reach the exit before being caught."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to enter swap mode, "
        "then use arrow keys to swap adjacent symbols."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (50, 40, 90)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_PLAYER_GLOW = (255, 255, 100, 50)
    COLOR_NIGHTMARE = (180, 50, 255)
    COLOR_NIGHTMARE_GLOW = (180, 50, 255, 50)
    COLOR_EXIT = (255, 150, 50)
    COLOR_EXIT_GLOW = (255, 150, 50, 50)
    COLOR_DETECTION_RADIUS = (100, 150, 255, 60)
    COLOR_TEXT = (230, 230, 240)
    SYMBOL_COLORS = [
        (255, 80, 80),   # Red
        (80, 120, 255),  # Blue
        (80, 255, 120),  # Green
        (255, 255, 80),  # Yellow
        (255, 120, 255), # Magenta
        (80, 255, 255),  # Cyan
    ]

    # Screen and Grid Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * 40) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * 40) // 2
    TILE_SIZE = 40
    
    # Game Parameters
    MAX_STEPS = 2000
    NIGHTMARE_BASE_SPEED = 0.5
    NIGHTMARE_SPEED_INCREASE = 0.05
    NIGHTMARE_BASE_RADIUS = 200
    POTION_RADIUS_REDUCTION = 2
    MIN_NIGHTMARE_RADIUS = 30
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.nightmare_pos = [0.0, 0.0]
        self.nightmare_speed = 0.0
        self.exit_pos = [0, 0]
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.potion_strength = 0
        self.num_symbol_types = 0
        self.particles = []
        self.stars = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.exit_pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
        while self.exit_pos == self.player_pos:
             self.exit_pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]

        nightmare_start_side = self.np_random.integers(0, 4)
        if nightmare_start_side == 0: self.nightmare_pos = [self.np_random.uniform(0, self.GRID_WIDTH-1), -2.0]
        elif nightmare_start_side == 1: self.nightmare_pos = [self.np_random.uniform(0, self.GRID_WIDTH-1), self.GRID_HEIGHT+1]
        elif nightmare_start_side == 2: self.nightmare_pos = [-2.0, self.np_random.uniform(0, self.GRID_HEIGHT-1)]
        else: self.nightmare_pos = [self.GRID_WIDTH+1, self.np_random.uniform(0, self.GRID_HEIGHT-1)]
            
        self.nightmare_speed = self.NIGHTMARE_BASE_SPEED
        self.potion_strength = 0
        self.num_symbol_types = 3
        self.particles = []
        self._create_stars()

        self._create_initial_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if space_held: # SWAP MODE
            reward += self._handle_swap(movement)
        else: # MOVEMENT MODE
            self._handle_movement(movement)
        
        # Update game logic
        self._update_nightmare()
        self._update_difficulty()
        
        # Check for termination
        terminated, term_reward = self._check_termination()
        self.game_over = terminated
        reward += term_reward
        self.score += term_reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Action Handling ---
    def _handle_movement(self, movement):
        if movement == 1 and self.player_pos[1] > 0: self.player_pos[1] -= 1 # Up
        elif movement == 2 and self.player_pos[1] < self.GRID_HEIGHT - 1: self.player_pos[1] += 1 # Down
        elif movement == 3 and self.player_pos[0] > 0: self.player_pos[0] -= 1 # Left
        elif movement == 4 and self.player_pos[0] < self.GRID_WIDTH - 1: self.player_pos[0] += 1 # Right
        # Sound: Player move woosh

    def _handle_swap(self, direction):
        if direction == 0: return 0 # No-op swap
        
        px, py = self.player_pos
        ox, oy = px, py
        
        if direction == 1 and py > 0: oy -= 1 # Up
        elif direction == 2 and py < self.GRID_HEIGHT - 1: oy += 1 # Down
        elif direction == 3 and px > 0: ox -= 1 # Left
        elif direction == 4 and px < self.GRID_WIDTH - 1: ox += 1 # Right
        else: return -0.1 # Invalid swap attempt (at edge)

        # Execute swap
        self.grid[px, py], self.grid[ox, oy] = self.grid[ox, oy], self.grid[px, py]
        
        # Check for matches
        matches = self._find_matches()
        if not matches:
            # Invalid swap, revert
            self.grid[px, py], self.grid[ox, oy] = self.grid[ox, oy], self.grid[px, py]
            # Sound: Invalid swap buzz
            return -0.1
        else:
            # Valid swap
            # Sound: Symbol swap click
            match_reward = self._handle_matches_and_cascades(matches)
            return match_reward

    # --- Game Logic ---
    def _update_nightmare(self):
        player_center_x = self.player_pos[0] + 0.5
        player_center_y = self.player_pos[1] + 0.5
        
        dx = player_center_x - self.nightmare_pos[0]
        dy = player_center_y - self.nightmare_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            self.nightmare_pos[0] += (dx / dist) * self.nightmare_speed / self.TILE_SIZE
            self.nightmare_pos[1] += (dy / dist) * self.nightmare_speed / self.TILE_SIZE

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.nightmare_speed += self.NIGHTMARE_SPEED_INCREASE
        if self.steps > 0 and self.steps % 200 == 0:
            self.num_symbol_types = min(len(self.SYMBOL_COLORS), self.num_symbol_types + 1)
            
    def _check_termination(self):
        # Win condition
        if self.player_pos == self.exit_pos:
            # Sound: Win fanfare
            return True, 100.0
        
        # Loss condition: caught by nightmare
        px, py = self.player_pos[0] * self.TILE_SIZE + self.GRID_OFFSET_X + self.TILE_SIZE/2, self.player_pos[1] * self.TILE_SIZE + self.GRID_OFFSET_Y + self.TILE_SIZE/2
        nx_pixel = self.nightmare_pos[0] * self.TILE_SIZE + self.GRID_OFFSET_X + self.TILE_SIZE/2
        ny_pixel = self.nightmare_pos[1] * self.TILE_SIZE + self.GRID_OFFSET_Y + self.TILE_SIZE/2
        dist = math.hypot(px - nx_pixel, py - ny_pixel)
        
        detection_radius = max(self.MIN_NIGHTMARE_RADIUS, self.NIGHTMARE_BASE_RADIUS - self.potion_strength * self.POTION_RADIUS_REDUCTION)
        
        if dist <= detection_radius:
            # Sound: Player caught scream
            return True, -100.0
        
        return False, 0.0

    # --- Match-3 Logic ---
    def _create_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.num_symbol_types + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            if not self._find_matches():
                if self._find_possible_moves():
                    break
    
    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0: continue
                # Horizontal
                if x < self.GRID_WIDTH - 2 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.GRID_HEIGHT - 2 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                    if self._find_matches():
                        self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                        return True
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
                    if self._find_matches():
                        self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
                        return True
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
        return False

    def _handle_matches_and_cascades(self, initial_matches):
        total_reward = 0
        all_matches = set(initial_matches)
        
        while True:
            if not all_matches: break
            
            # Process current matches
            num_matched_symbols = len(all_matches)
            self.potion_strength += num_matched_symbols
            match_reward = num_matched_symbols * 0.1 + 1.0 # Potion bonus
            total_reward += match_reward
            self.score += match_reward
            
            for x, y in all_matches:
                self._create_particles(x, y, self.grid[x, y])
                self.grid[x, y] = 0
            # Sound: Symbol match pop
            
            # Apply gravity
            for x in range(self.GRID_WIDTH):
                empty_row = self.GRID_HEIGHT - 1
                for y in range(self.GRID_HEIGHT - 1, -1, -1):
                    if self.grid[x, y] != 0:
                        self.grid[x, y], self.grid[x, empty_row] = self.grid[x, empty_row], self.grid[x, y]
                        empty_row -= 1
            
            # Fill top rows
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if self.grid[x, y] == 0:
                        self.grid[x, y] = self.np_random.integers(1, self.num_symbol_types + 1)
            # Sound: Symbols falling
            
            # Check for new matches
            new_matches = self._find_matches()
            if not new_matches:
                break
            all_matches = set(new_matches)

        # Ensure board is not stuck
        if not self._find_possible_moves():
            self._create_initial_grid() # Reshuffle
            
        return total_reward

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "potion_strength": self.potion_strength}

    def _render_game(self):
        self._update_and_render_particles()
        self._render_detection_radius()
        self._render_grid()
        self._render_exit()
        self._render_symbols()
        self._render_player()
        self._render_nightmare()

    def _create_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.integers(1, 4)
            self.stars.append((x, y, size))

    def _render_background(self):
        for x, y, size in self.stars:
            y_new = (y + self.steps // (5-size)) % self.SCREEN_HEIGHT
            brightness = 50 + size * 20
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y_new), size // 2)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.TILE_SIZE, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_symbols(self):
        padding = 8
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                symbol_id = self.grid[x, y]
                if symbol_id == 0: continue
                
                color = self.SYMBOL_COLORS[(symbol_id - 1) % len(self.SYMBOL_COLORS)]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.TILE_SIZE + padding,
                    self.GRID_OFFSET_Y + y * self.TILE_SIZE + padding,
                    self.TILE_SIZE - 2 * padding,
                    self.TILE_SIZE - 2 * padding
                )
                center = rect.center
                
                # Draw unique shape for each symbol type
                if symbol_id % 6 == 1: # Circle
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], rect.width // 2, color)
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], rect.width // 2, color)
                elif symbol_id % 6 == 2: # Square
                    pygame.draw.rect(self.screen, color, rect)
                elif symbol_id % 6 == 3: # Triangle
                    points = [(center[0], rect.top), (rect.right, rect.bottom), (rect.left, rect.bottom)]
                    pygame.gfxdraw.aapolygon(self.screen, points, color)
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                elif symbol_id % 6 == 4: # Diamond
                    points = [(center[0], rect.top), (rect.right, center[1]), (center[0], rect.bottom), (rect.left, center[1])]
                    pygame.gfxdraw.aapolygon(self.screen, points, color)
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                elif symbol_id % 6 == 5: # Cross
                    pygame.draw.line(self.screen, color, rect.topleft, rect.bottomright, 5)
                    pygame.draw.line(self.screen, color, rect.topright, rect.bottomleft, 5)
                else: # Hexagon
                    radius = rect.width // 2
                    points = [(center[0] + radius * math.cos(math.pi/3 * i), center[1] + radius * math.sin(math.pi/3*i)) for i in range(6)]
                    pygame.gfxdraw.aapolygon(self.screen, points, color)
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_player(self):
        px, py = self.player_pos
        center_x = self.GRID_OFFSET_X + px * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_OFFSET_Y + py * self.TILE_SIZE + self.TILE_SIZE // 2
        
        # Player aura (visualizes potion strength)
        aura_radius = max(0, self.potion_strength / 2)
        if aura_radius > 0:
            self._draw_glow(center_x, center_y, self.TILE_SIZE // 2 + aura_radius, self.COLOR_PLAYER_GLOW)
        
        # Player selection cursor
        rect = pygame.Rect(
            self.GRID_OFFSET_X + px * self.TILE_SIZE + 1,
            self.GRID_OFFSET_Y + py * self.TILE_SIZE + 1,
            self.TILE_SIZE - 2,
            self.TILE_SIZE - 2
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, 3, border_radius=4)
        
    def _render_nightmare(self):
        center_x = int(self.GRID_OFFSET_X + self.nightmare_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        center_y = int(self.GRID_OFFSET_Y + self.nightmare_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        
        self._draw_glow(center_x, center_y, 25, self.COLOR_NIGHTMARE_GLOW)
        
        radius = 15
        num_spikes = 7
        angle_offset = (self.steps * 0.05) % (2 * math.pi)
        points = []
        for i in range(num_spikes * 2):
            r = radius if i % 2 == 0 else radius * 0.6
            angle = (i / (num_spikes * 2)) * 2 * math.pi + angle_offset
            points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_NIGHTMARE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_NIGHTMARE)

    def _render_exit(self):
        ex, ey = self.exit_pos
        center_x = self.GRID_OFFSET_X + ex * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_OFFSET_Y + ey * self.TILE_SIZE + self.TILE_SIZE // 2
        
        self._draw_glow(center_x, center_y, 25, self.COLOR_EXIT_GLOW)
        
        radius = 12 + 3 * math.sin(self.steps * 0.1)
        rect = pygame.Rect(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
        pygame.draw.ellipse(self.screen, self.COLOR_EXIT, rect)
        
    def _render_detection_radius(self):
        nx = int(self.GRID_OFFSET_X + self.nightmare_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        ny = int(self.GRID_OFFSET_Y + self.nightmare_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        radius = int(max(self.MIN_NIGHTMARE_RADIUS, self.NIGHTMARE_BASE_RADIUS - self.potion_strength * self.POTION_RADIUS_REDUCTION))
        
        if radius > 0:
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pulse_alpha = 40 + 20 * math.sin(self.steps * 0.08)
            pygame.gfxdraw.filled_circle(surf, radius, radius, radius, (*self.COLOR_DETECTION_RADIUS[:3], pulse_alpha))
            pygame.gfxdraw.aacircle(surf, radius, radius, radius, (*self.COLOR_DETECTION_RADIUS[:3], pulse_alpha + 20))
            self.screen.blit(surf, (nx - radius, ny - radius))
            
    def _draw_glow(self, x, y, radius, color):
        for i in range(int(radius // 2), 0, -2):
            alpha = color[3] * (1 - (i / (radius // 2)))**2
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(radius - i), (*color[:3], int(alpha)))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        potion_text = self.font_large.render(f"POTION: {self.potion_strength}", True, self.COLOR_TEXT)
        self.screen.blit(potion_text, (10, 40))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    # --- Particles ---
    def _create_particles(self, grid_x, grid_y, symbol_id):
        center_x = self.GRID_OFFSET_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        color = self.SYMBOL_COLORS[(symbol_id - 1) % len(self.SYMBOL_COLORS)]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append([center_x, center_y, vx, vy, lifetime, color])

    def _update_and_render_particles(self):
        remaining_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1 # lifetime -= 1
            if p[4] > 0:
                remaining_particles.append(p)
                radius = max(0, int(p[4] / 8))
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), radius)
        self.particles = remaining_particles

    # --- Housekeeping ---
    def close(self):
        pygame.quit()
        
# Example usage
if __name__ == '__main__':
    # This block is not part of the required solution and is for testing/debugging.
    # It will not be executed by the test harness.
    # To run, you may need to unset the SDL_VIDEODRIVER dummy variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Dream Escape")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    swap_mode = False
    
    while not done:
        movement_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement_action = 1
                elif event.key == pygame.K_DOWN: movement_action = 2
                elif event.key == pygame.K_LEFT: movement_action = 3
                elif event.key == pygame.K_RIGHT: movement_action = 4
                elif event.key == pygame.K_SPACE: swap_mode = not swap_mode
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    swap_mode = False
                    continue

        # Only step if an action is taken
        if movement_action > 0 or not swap_mode:
            action = [movement_action, 1 if swap_mode and movement_action > 0 else 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # If a swap was attempted, reset swap mode for next turn
            if swap_mode and movement_action > 0:
                swap_mode = False
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display swap mode status
        if swap_mode:
            font = pygame.font.SysFont("monospace", 18, bold=True)
            swap_text = font.render("SWAP MODE", True, (255, 200, 200))
            screen.blit(swap_text, (GameEnv.SCREEN_WIDTH // 2 - swap_text.get_width() // 2, 10))

        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play

    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()