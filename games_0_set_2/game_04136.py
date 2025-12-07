import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


# Set the SDL video driver to dummy to run headless, required for server-side execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. The last direction moved sets your push direction. "
        "Press space to push the selected crystal. Clear groups of 3 or more."
    )

    game_description = (
        "An isometric puzzle game. Push crystals to create lines of 3 or more of the same color. "
        "Plan your moves carefully to create chain reactions and clear the board before you run out of pushes."
    )

    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    WIN_CONDITION_CRYSTALS = 20
    MAX_MOVES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 20, 40)
    COLOR_GRID = (50, 40, 70)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
    ]
    CRYSTAL_BORDERS = [
        (255, 150, 150),
        (150, 255, 150),
        (150, 180, 255),
    ]
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (255, 255, 220)

    # Rendering
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    ISO_TILE_WIDTH = 48
    ISO_TILE_HEIGHT = 24
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.ISO_TILE_HEIGHT) // 2 + 30

        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.previous_space_held = False
        self.moves_left = 0
        self.crystals_cleared_total = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.rng = np.random.default_rng()
        
        # NOTE: self.reset() is not called in __init__ to avoid long initialization times.
        # The user is expected to call reset() before the first step.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.crystals_cleared_total = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_dir = (0, -1) # Default up
        self.previous_space_held = True # Prevent action on first frame
        self.particles = []
        
        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = [[None for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self.grid[x][y] = self.rng.integers(0, len(self.CRYSTAL_COLORS))
        
        # Ensure no initial matches
        while self._find_matches():
            matches = self._find_matches()
            for x, y in matches:
                # Re-randomize to break the match, which is more robust than incrementing.
                self.grid[x][y] = self.rng.integers(0, len(self.CRYSTAL_COLORS))

    def step(self, action):
        if self.game_over:
            # Per Gymnasium API, behavior is undefined after an episode ends.
            # We return the last observation and signal termination.
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        # Particles are cleared and recreated each step for simplicity
        self.particles.clear()

        # 1. Handle cursor movement and direction setting
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right in grid space
        if movement in move_map:
            move_dir = move_map[movement]
            self.last_move_dir = move_dir
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + move_dir[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + move_dir[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle push action (on space press)
        space_pressed = space_held and not self.previous_space_held
        if space_pressed and self.moves_left > 0:
            self.moves_left -= 1
            cx, cy = self.cursor_pos
            if self.grid[cx][cy] is not None:
                self._push_line(cx, cy, self.last_move_dir)
            
            cleared_this_turn, cascade_reward = self._handle_cascades()
            reward += cascade_reward
            self.crystals_cleared_total += cleared_this_turn
            self.score += cleared_this_turn * 10
            if cleared_this_turn >= 4:
                self.score += cleared_this_turn * 10 # Bonus score for combos
        
        self.previous_space_held = space_held
        self.steps += 1
        
        # Check for terminal states (win/loss)
        terminated = (self.crystals_cleared_total >= self.WIN_CONDITION_CRYSTALS) or (self.moves_left <= 0)
        
        # Check for truncation (time limit)
        truncated = self.steps >= self.MAX_STEPS

        # Apply win bonus only on the step the win occurs
        if terminated and not self.game_over and self.crystals_cleared_total >= self.WIN_CONDITION_CRYSTALS:
            reward += 50
            self.score += 500

        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _push_line(self, x, y, direction):
        dx, dy = direction
        line = []
        cx, cy = x, y
        while 0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT:
            if self.grid[cx][cy] is not None:
                line.append((cx, cy))
            cx += dx
            cy += dy
        
        if not line: return

        # Find where the last crystal in the line will land
        end_x, end_y = line[-1]
        target_x, target_y = end_x + dx, end_y + dy

        # If target is off-grid, the last crystal is removed
        if not (0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT):
            self.grid[end_x][end_y] = None
            line.pop()

        # Move all other crystals in reverse order
        for i in range(len(line) - 1, -1, -1):
            px, py = line[i]
            self.grid[px + dx][py + dy] = self.grid[px][py]
            self.grid[px][py] = None

    def _handle_cascades(self):
        total_cleared = 0
        total_reward = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            num_cleared_this_pass = len(matches)
            total_cleared += num_cleared_this_pass
            total_reward += num_cleared_this_pass
            if num_cleared_this_pass >= 4:
                total_reward += 5 # Bonus reward for clearing 4+

            for x, y in matches:
                self._create_particles(x, y)
                self.grid[x][y] = None
            
            self._apply_gravity()

        return total_cleared, total_reward

    def _find_matches(self):
        to_clear = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x][y] is None: continue
                color = self.grid[x][y]
                
                # Horizontal check
                h_match = [(x, y)]
                for i in range(1, self.GRID_WIDTH):
                    if x + i < self.GRID_WIDTH and self.grid[x+i][y] == color:
                        h_match.append((x+i, y))
                    else: break
                if len(h_match) >= 3:
                    to_clear.update(h_match)

                # Vertical check
                v_match = [(x, y)]
                for i in range(1, self.GRID_HEIGHT):
                    if y + i < self.GRID_HEIGHT and self.grid[x][y+i] == color:
                        v_match.append((x, y+i))
                    else: break
                if len(v_match) >= 3:
                    to_clear.update(v_match)
        return list(to_clear)

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x][y] is not None:
                    if y != empty_y:
                        self.grid[x][empty_y] = self.grid[x][y]
                        self.grid[x][y] = None
                    empty_y -= 1
        
    def _get_observation(self):
        # If reset hasn't been called, grid is None. Render a blank screen.
        if self.grid is None:
            self.screen.fill(self.COLOR_BG)
        else:
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
            "crystals_cleared": self.crystals_cleared_total,
        }

    def _project_iso(self, x, y):
        screen_x = self.grid_origin_x + (x - y) * (self.ISO_TILE_WIDTH / 2)
        screen_y = self.grid_origin_y + (x + y) * (self.ISO_TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._project_iso(x, y)
                points = [
                    (sx, sy),
                    (sx + self.ISO_TILE_WIDTH / 2, sy + self.ISO_TILE_HEIGHT / 2),
                    (sx, sy + self.ISO_TILE_HEIGHT),
                    (sx - self.ISO_TILE_WIDTH / 2, sy + self.ISO_TILE_HEIGHT / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x][y] is not None:
                    self._draw_crystal(x, y, self.grid[x][y])
        
        # Draw particles
        for p in self.particles:
            pygame.draw.line(self.screen, self.COLOR_PARTICLE, p[0], p[1], 1)

        # Draw cursor
        cx, cy = self.cursor_pos
        sx, sy = self._project_iso(cx, cy)
        points = [
            (sx, sy - 2),
            (sx + self.ISO_TILE_WIDTH / 2 + 2, sy + self.ISO_TILE_HEIGHT / 2),
            (sx, sy + self.ISO_TILE_HEIGHT + 2),
            (sx - self.ISO_TILE_WIDTH / 2 - 2, sy + self.ISO_TILE_HEIGHT / 2)
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
    def _draw_crystal(self, x, y, color_index):
        sx, sy = self._project_iso(x, y)
        base_color = self.CRYSTAL_COLORS[color_index]
        border_color = self.CRYSTAL_BORDERS[color_index]
        
        # Pulsing glow for selected crystal
        if x == self.cursor_pos[0] and y == self.cursor_pos[1]:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
            glow_size = int(self.ISO_TILE_WIDTH * (0.6 + pulse * 0.2))
            glow_color = base_color
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, 60), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (sx - glow_size, sy - glow_size + self.ISO_TILE_HEIGHT/2), special_flags=pygame.BLEND_RGBA_ADD)

        # Crystal body
        points = [
            (sx, sy + 4),
            (sx + self.ISO_TILE_WIDTH / 2 - 4, sy + self.ISO_TILE_HEIGHT / 2 + 2),
            (sx, sy + self.ISO_TILE_HEIGHT - 4),
            (sx - self.ISO_TILE_WIDTH / 2 + 4, sy + self.ISO_TILE_HEIGHT / 2 + 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, base_color)
        pygame.gfxdraw.aapolygon(self.screen, points, border_color)

    def _create_particles(self, x, y):
        sx, sy = self._project_iso(x, y)
        center_x = sx
        center_y = sy + self.ISO_TILE_HEIGHT / 2
        for _ in range(12):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(10, 25)
            end_x = center_x + math.cos(angle) * speed
            end_y = center_y + math.sin(angle) * speed
            self.particles.append(((center_x, center_y), (end_x, end_y)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves left
        moves_text = self.font_large.render(f"Pushes: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Crystals cleared
        cleared_text = self.font_small.render(f"Cleared: {self.crystals_cleared_total} / {self.WIN_CONDITION_CRYSTALS}", True, self.COLOR_TEXT)
        self.screen.blit(cleared_text, (10, 45))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.crystals_cleared_total >= self.WIN_CONDITION_CRYSTALS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()