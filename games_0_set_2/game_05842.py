import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the falling crystal, ↓ to drop it faster."
    )

    game_description = (
        "Maneuver falling crystals in a procedurally generated cavern to create color matches and clear the board before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 60

        # Visuals
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.TILE_WIDTH_HALF = self.TILE_WIDTH // 2
        self.TILE_HEIGHT_HALF = self.TILE_HEIGHT // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 60

        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_UI_TEXT = (236, 240, 241)
        self.COLOR_WIN = (46, 204, 113)
        self.COLOR_LOSE = (231, 76, 60)
        self.CRYSTAL_COLORS = {
            1: (231, 76, 60),   # Red
            2: (46, 204, 113),  # Green
            3: (52, 152, 219),  # Blue
        }
        self.CRYSTAL_GLOW_COLORS = {
            1: (231, 76, 60, 100),
            2: (46, 204, 113, 100),
            3: (52, 152, 219, 100),
        }

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.grid = None
        self.falling_crystal = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = 0
        self.crystals_cleared_total = 0
        self.spawn_timer = 0
        self.spawn_interval = 0
        self.move_cooldown = 0
        self.last_action_caused_match = False
        self.np_random = None

        # self.reset() # Removed from __init__ to allow seeding from external wrappers
        # self.validate_implementation() # Also removed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        # Pre-fill bottom rows for an initial challenge
        for r in range(self.GRID_HEIGHT - 5, self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.np_random.random() < 0.7:
                    self.grid[r, c] = self.np_random.integers(1, 4)
        
        # Clear any initial accidental matches
        while self._find_and_clear_matches(is_initial_clear=True) > 0:
            self._apply_gravity()

        self.falling_crystal = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.crystals_cleared_total = 0
        self.spawn_interval = 1.5 * self.FPS
        self.spawn_timer = self.spawn_interval // 2
        self.move_cooldown = 0
        self.last_action_caused_match = False
        
        self._spawn_crystal()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small cost for taking a step (time passing)
        self.last_action_caused_match = False
        landed_this_step = False

        movement = action[0]

        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # Handle player input
        if self.falling_crystal and self.move_cooldown <= 0:
            if movement == 3:  # Left
                self._move_crystal(-1, 0)
                self.move_cooldown = 4
            elif movement == 4:  # Right
                self._move_crystal(1, 0)
                self.move_cooldown = 4
        
        if self.falling_crystal and movement == 2: # Down
            self.falling_crystal['y'] += 0.5 # Accelerate fall

        # Update falling crystal
        if self.falling_crystal:
            self.falling_crystal['y'] += self.falling_crystal['fall_speed']
            landed = self._check_crystal_landing()
            if landed:
                landed_this_step = True
                self.falling_crystal = None

        # Handle match chains and gravity
        if landed_this_step:
            match_reward, cleared_count = self._handle_matches_and_gravity()
            if cleared_count > 0:
                reward += match_reward
                self.last_action_caused_match = True
        
        if self.last_action_caused_match:
            reward += 1.0

        # Update spawner
        if not self.falling_crystal and not self.game_over:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_crystal()
        
        # Update particles
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win:
                reward += 100
            elif self.time_left <= 0:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_crystals()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "crystals_cleared": self.crystals_cleared_total,
        }

    # --- Game Logic Helpers ---

    def _spawn_crystal(self):
        if self.game_over: return
        start_x = self.np_random.integers(0, self.GRID_WIDTH)
        # Check if spawn area is blocked
        if self.grid[0, start_x] != 0:
            self.game_over = True
            return

        self.falling_crystal = {
            'x': start_x,
            'y': -1.0, # Start just above the grid
            'color': self.np_random.integers(1, 4),
            'fall_speed': 1 / self.FPS * 2.0,
        }
        self.spawn_timer = self.spawn_interval

    def _move_crystal(self, dx, dy):
        if not self.falling_crystal: return
        new_x = self.falling_crystal['x'] + dx
        if 0 <= new_x < self.GRID_WIDTH:
            # Check for collision with settled blocks
            if int(self.falling_crystal['y']) >= 0 and self.grid[int(self.falling_crystal['y']), new_x] == 0:
                self.falling_crystal['x'] = new_x

    def _check_crystal_landing(self):
        if not self.falling_crystal: return False
        
        cx, cy_float = self.falling_crystal['x'], self.falling_crystal['y']
        cy_grid = int(cy_float)
        
        # Check if it will hit the floor or another crystal in the next step
        next_y_grid = int(cy_float + self.falling_crystal['fall_speed'])
        
        landed = False
        if next_y_grid >= self.GRID_HEIGHT -1:
            landed = True
        elif next_y_grid >= 0 and self.grid[next_y_grid + 1, cx] != 0:
            landed = True

        if landed:
            final_y = min(self.GRID_HEIGHT - 1, cy_grid)
            # Ensure we don't place on an existing crystal
            while final_y >= 0 and self.grid[final_y, cx] != 0:
                final_y -= 1
            
            if final_y < 0: # Column is full
                self.game_over = True
            else:
                self.grid[final_y, cx] = self.falling_crystal['color']
                # sfx: Crystal land
            return True
        return False

    def _handle_matches_and_gravity(self):
        total_reward = 0
        total_cleared = 0
        
        while True:
            cleared_count = self._find_and_clear_matches()
            if cleared_count == 0:
                break
            
            # sfx: Match clear
            total_cleared += cleared_count
            
            if cleared_count == 3: total_reward += 5
            elif cleared_count == 4: total_reward += 10
            elif cleared_count >= 5: total_reward += 20
            
            self._apply_gravity()
            # Add a small delay for chain reaction visuals if needed, but for RL it's better to be instant
        
        return total_reward, total_cleared

    def _find_and_clear_matches(self, is_initial_clear=False):
        to_clear = set()
        
        # Horizontal check
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    for i in range(3): to_clear.add((r, c+i))
                    # Check for longer matches
                    for i in range(3, self.GRID_WIDTH - c):
                        if self.grid[r, c+i] == color: to_clear.add((r, c+i))
                        else: break

        # Vertical check
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    for i in range(3): to_clear.add((r+i, c))
                    # Check for longer matches
                    for i in range(3, self.GRID_HEIGHT - r):
                        if self.grid[r+i, c] == color: to_clear.add((r+i, c))
                        else: break

        if not to_clear: return 0

        if not is_initial_clear:
            for r, c in to_clear:
                self._create_particles(c, r, self.grid[r, c])
                self.score += 10

        for r, c in to_clear:
            self.grid[r, c] = 0
        
        if not is_initial_clear:
            self.crystals_cleared_total += len(to_clear)
            # Difficulty scaling
            spawn_reduction = self.crystals_cleared_total // 10 * 0.1
            self.spawn_interval = max(1.0, 1.5 - spawn_reduction) * self.FPS
        
        return len(to_clear)

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.time_left <= 0:
            return True
        if np.all(self.grid == 0):
            self.win = True
            return True
        return False

    # --- Rendering Helpers ---

    def _world_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_grid_and_crystals(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_val = self.grid[r, c]
                tile_color = self.COLOR_GRID if color_val == 0 else self.CRYSTAL_COLORS[color_val]
                self._draw_iso_tile(c, r, tile_color, color_val != 0, color_val=color_val)

        if self.falling_crystal:
            c, r_float = self.falling_crystal['x'], self.falling_crystal['y']
            color_val = self.falling_crystal['color']
            self._draw_iso_tile(c, r_float, self.CRYSTAL_COLORS[color_val], True, is_falling=True, color_val=color_val)

    def _draw_iso_tile(self, x, y, color, has_fill, is_falling=False, color_val=0):
        sx, sy = self._world_to_screen(x, y)
        
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
        ]
        
        if has_fill:
            # Glow effect
            if not is_falling and color_val != 0:
                glow_points = [
                    (sx, sy - 2),
                    (sx + self.TILE_WIDTH_HALF + 2, sy + self.TILE_HEIGHT_HALF),
                    (sx, sy + self.TILE_HEIGHT + 2),
                    (sx - self.TILE_WIDTH_HALF - 2, sy + self.TILE_HEIGHT_HALF),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.CRYSTAL_GLOW_COLORS[color_val])

            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, tuple(c*0.8 for c in color[:3]))
        else:
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _create_particles(self, x, y, color_val):
        sx, sy = self._world_to_screen(x, y)
        sy += self.TILE_HEIGHT_HALF
        base_color = self.CRYSTAL_COLORS[color_val]
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': sx, 'y': sy,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': base_color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = int(max(0, p['lifespan'] / 5))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['x'] - size/2, p['y'] - size/2, size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_str = f"Time: {max(0, self.time_left / self.FPS):.1f}"
        timer_text = self.font_medium.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 20, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "TIME'S UP!" if self.time_left <= 0 else "GAME OVER"
                color = self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- This block is for human play and visualization ---
    # It is not part of the required Gymnasium interface
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # --- Pygame setup for human play ---
    # This is separate from the headless rendering in the environment
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cavern Crystals")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action.fill(0)
        # action[0] = 0 is no-op for movement
        if keys[pygame.K_DOWN]: action[0] = 2
        if keys[pygame.K_LEFT]: action[0] = 3
        if keys[pygame.K_RIGHT]: action[0] = 4
        # action[1] and action[2] are not used in this human control scheme
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

        if done:
            print(f"Game Over. Final Info: {info}")
            pygame.time.wait(2000) # Wait 2 seconds before closing
    
    env.close()