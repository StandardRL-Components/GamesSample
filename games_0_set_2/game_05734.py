
# Generated: 2025-08-28T05:56:00.315764
# Source Brief: brief_05734.md
# Brief Index: 5734

        
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


class Particle:
    """A simple class for a single particle effect."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = random.randint(15, 30)
        self.radius = random.uniform(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.2

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), self.color)


class Tile:
    """Represents a single tile on the grid with animation properties."""
    def __init__(self, color_idx, grid_x, grid_y, start_y_offset=0):
        self.color_idx = color_idx
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_y_offset = start_y_offset
        self.size_scale = 0.0
        self.is_selected = False
        self.is_clearing = False
        self.is_hint = False

    def update(self):
        # Animate size for spawning and clearing
        if self.is_clearing:
            self.size_scale = max(0, self.size_scale - 0.15)
        else:
            self.size_scale = min(1.0, self.size_scale + 0.1)

        # Animate falling
        if self.pixel_y_offset > 0:
            self.pixel_y_offset = max(0, self.pixel_y_offset - 8)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select/deselect tiles. "
        "Match adjacent tiles of the same color to score points."
    )

    game_description = (
        "Fast-paced puzzle game. Match colored tiles on a grid to score points "
        "and clear stages against the clock."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visual & Game Constants ---
        self.GRID_SIZE = 10
        self.TILE_SIZE = 36
        self.TILE_PADDING = 4
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_WIDTH) // 2 + 20

        # Colors
        self.COLOR_BG = {1: (20, 30, 40), 2: (30, 20, 40), 3: (40, 30, 20)}
        self.COLOR_GRID = (60, 70, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.TILE_COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 150, 255),   # Blue
            (255, 255, 87),   # Yellow
            (255, 87, 255),   # Magenta
        ]

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)

        # Game constants
        self.FPS = 30
        self.STAGE_TIME_LIMIT = 60
        self.STAGE_SCORE_TARGETS = {1: 150, 2: 250, 3: 350}
        self.MAX_STEPS = self.STAGE_TIME_LIMIT * self.FPS * 3 + 100 # Approx max steps for 3 stages

        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.selected_tile_pos = None
        self.score = None
        self.stage = None
        self.time_remaining = None
        self.game_over = None
        self.steps = None
        self.prev_space_held = None
        self.particles = None
        self.game_state = "playing" # playing, stage_clear, win, lose
        self.message_timer = 0
        self.message_text = ""
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.prev_space_held = False
        self.particles = []

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.visual_cursor_pos = [float(self.cursor_pos[0]), float(self.cursor_pos[1])]
        
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.time_remaining = self.STAGE_TIME_LIMIT
        self.selected_tile_pos = None
        self.game_state = "playing"
        self._show_message(f"STAGE {self.stage}", 60)
        self._generate_grid()

    def _generate_grid(self):
        while True:
            self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), None, dtype=object)
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    color_idx = self._get_random_color_for_stage()
                    self.grid[r, c] = Tile(color_idx, c, r, start_y_offset=-self.TILE_SIZE * r)
            
            if not self._find_and_remove_initial_matches() and self._count_valid_moves() >= 5:
                break

    def _get_random_color_for_stage(self):
        num_colors = len(self.TILE_COLORS)
        if self.stage == 1:
            return self.rng.integers(0, num_colors)
        elif self.stage == 2:
            # Reduce probability of 2 colors
            weights = [1.0] * num_colors
            weights[0] = 0.2
            weights[1] = 0.2
            weights = np.array(weights) / sum(weights)
            return self.rng.choice(num_colors, p=weights)
        else: # Stage 3
            # Reduce probability of 3 colors
            weights = [1.0] * num_colors
            weights[0] = 0.1
            weights[1] = 0.1
            weights[2] = 0.1
            weights = np.array(weights) / sum(weights)
            return self.rng.choice(num_colors, p=weights)

    def _find_and_remove_initial_matches(self):
        matches_found = False
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if c < self.GRID_SIZE - 2 and self.grid[r, c].color_idx == self.grid[r, c+1].color_idx == self.grid[r, c+2].color_idx:
                    matches_found = True
                if r < self.GRID_SIZE - 2 and self.grid[r, c].color_idx == self.grid[r+1, c].color_idx == self.grid[r+2, c].color_idx:
                    matches_found = True
        return matches_found

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if self.game_state != "playing":
            self.message_timer -= 1
            if self.message_timer <= 0:
                if self.game_state == "stage_clear":
                    self.stage += 1
                    self.score = 0
                    self._setup_stage()
                elif self.game_state in ["win", "lose"]:
                    terminated = True

        if self.game_state == "playing":
            self.time_remaining -= 1 / self.FPS
            self._handle_input(movement, space_held, reward_ref=[reward])

        self._update_animations()
        board_was_stable = self._is_board_stable()
        if self._handle_gravity_and_refill() and board_was_stable:
            # sfx: tiles_land
            if self._check_and_resolve_softlock():
                # sfx: reshuffle
                self._show_message("Reshuffling...", 45)

        # Check for win/loss conditions
        if self.game_state == "playing":
            if self.score >= self.STAGE_SCORE_TARGETS[self.stage]:
                reward += 50.0
                if self.stage == 3:
                    self.game_state = "win"
                    self._show_message("YOU WIN!", 120)
                    reward += 100.0
                else:
                    self.game_state = "stage_clear"
                    self._show_message("STAGE CLEAR!", 90)
            elif self.time_remaining <= 0:
                self.game_state = "lose"
                self._show_message("TIME'S UP!", 120)
                reward -= 50.0
        
        self.game_over = terminated
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, reward_ref):
        if not self._is_board_stable():
            return

        # Cursor Movement
        cx, cy = self.cursor_pos
        if movement == 1 and cy > 0: self.cursor_pos[1] -= 1  # Up
        elif movement == 2 and cy < self.GRID_SIZE - 1: self.cursor_pos[1] += 1  # Down
        elif movement == 3 and cx > 0: self.cursor_pos[0] -= 1  # Left
        elif movement == 4 and cx < self.GRID_SIZE - 1: self.cursor_pos[0] += 1  # Right
        if movement > 0: # sfx: cursor_move
            pass

        # Selection Logic
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # sfx: select_tile
            cx, cy = self.cursor_pos
            if self.selected_tile_pos is None:
                self.selected_tile_pos = (cx, cy)
            else:
                px, py = self.selected_tile_pos
                dist = abs(cx - px) + abs(cy - py)
                if (cx, cy) == (px, py): # Deselect
                    self.selected_tile_pos = None
                elif dist == 1 and self.grid[cy, cx].color_idx == self.grid[py, px].color_idx:
                    self._process_match(cx, cy, px, py, reward_ref)
                else: # Invalid match
                    # sfx: invalid_match
                    reward_ref[0] -= 0.2
                    self.selected_tile_pos = None
        self.prev_space_held = space_held

    def _process_match(self, x1, y1, x2, y2, reward_ref):
        start_tile = self.grid[y1, x1]
        color_to_match = start_tile.color_idx
        
        q = [(x1, y1), (x2, y2)]
        visited = set(q)
        to_clear = list(q)

        while q:
            cx, cy = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    if self.grid[ny, nx].color_idx == color_to_match:
                        visited.add((nx, ny))
                        q.append((nx, ny))
                        to_clear.append((nx, ny))
        
        if len(to_clear) > 2: # Found a chain
            # sfx: match_success_chain
            if len(to_clear) > 3:
                reward_ref[0] += 5.0
        else: # sfx: match_success
            pass

        for x, y in to_clear:
            self.grid[y, x].is_clearing = True
            reward_ref[0] += 1.0
            self.score += 1
            self._create_particles(x, y, self.grid[y, x].color_idx)
        
        self.selected_tile_pos = None

    def _update_animations(self):
        # Update tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c]:
                    self.grid[r, c].update()
        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
        # Update cursor
        lerp_rate = 0.4
        self.visual_cursor_pos[0] += (self.cursor_pos[0] - self.visual_cursor_pos[0]) * lerp_rate
        self.visual_cursor_pos[1] += (self.cursor_pos[1] - self.visual_cursor_pos[1]) * lerp_rate

    def _handle_gravity_and_refill(self):
        board_changed = False
        # Remove cleared tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] and self.grid[r, c].size_scale == 0 and self.grid[r, c].is_clearing:
                    self.grid[r, c] = None
                    board_changed = True

        # Gravity
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] is not None:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[empty_row, c].grid_y = empty_row
                        self.grid[empty_row, c].pixel_y_offset = (empty_row - r) * self.TILE_SIZE
                        self.grid[r, c] = None
                        board_changed = True
                    empty_row -= 1
        
        # Refill
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE):
                if self.grid[r, c] is None:
                    color = self._get_random_color_for_stage()
                    self.grid[r, c] = Tile(color, c, r, start_y_offset=-self.TILE_SIZE)
                    board_changed = True
        return board_changed

    def _is_board_stable(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile = self.grid[r, c]
                if not tile or tile.is_clearing or tile.pixel_y_offset > 0:
                    return False
        return True

    def _count_valid_moves(self):
        moves = 0
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check right
                if c < self.GRID_SIZE - 1 and self.grid[r, c].color_idx == self.grid[r, c+1].color_idx:
                    moves += 1
                # Check down
                if r < self.GRID_SIZE - 1 and self.grid[r, c].color_idx == self.grid[r+1, c].color_idx:
                    moves += 1
        return moves

    def _check_and_resolve_softlock(self):
        if self._count_valid_moves() == 0:
            self.rng.shuffle(self.grid.flatten())
            for i, tile in enumerate(self.grid.flatten()):
                r, c = divmod(i, self.GRID_SIZE)
                tile.grid_x, tile.grid_y = c, r
                tile.pixel_y_offset = -self.TILE_SIZE * 2
            return True
        return False
        
    def _create_particles(self, grid_x, grid_y, color_idx):
        px = self.GRID_ORIGIN_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_ORIGIN_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(15):
            self.particles.append(Particle(px, py, self.TILE_COLORS[color_idx]))

    def _show_message(self, text, duration):
        self.message_text = text
        self.message_timer = duration

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG[self.stage])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y, self.GRID_WIDTH, self.GRID_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile = self.grid[r, c]
                if tile:
                    self._draw_tile(tile)
        
        # Draw cursor
        if self._is_board_stable() and self.game_state == "playing":
            cursor_x = self.GRID_ORIGIN_X + self.visual_cursor_pos[0] * self.TILE_SIZE
            cursor_y = self.GRID_ORIGIN_Y + self.visual_cursor_pos[1] * self.TILE_SIZE
            cursor_rect = pygame.Rect(cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_tile(self, tile):
        color = self.TILE_COLORS[tile.color_idx]
        ts = self.TILE_SIZE - self.TILE_PADDING
        scaled_ts = int(ts * tile.size_scale)
        if scaled_ts <= 0: return

        px = self.GRID_ORIGIN_X + tile.grid_x * self.TILE_SIZE + (self.TILE_SIZE - scaled_ts) // 2
        py = self.GRID_ORIGIN_Y + tile.grid_y * self.TILE_SIZE + (self.TILE_SIZE - scaled_ts) // 2 - tile.pixel_y_offset
        
        tile_rect = pygame.Rect(px, py, scaled_ts, scaled_ts)
        
        # Shadow
        shadow_rect = tile_rect.copy()
        shadow_rect.move_ip(2, 2)
        pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=6)
        
        # Main tile
        pygame.draw.rect(self.screen, color, tile_rect, border_radius=6)

        # Highlight for selected tile
        if self.selected_tile_pos == (tile.grid_x, tile.grid_y):
            pygame.draw.rect(self.screen, (255, 255, 255), tile_rect, 3, border_radius=7)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 15))

        # Stage
        target_score = self.STAGE_SCORE_TARGETS[self.stage]
        stage_surf = self.font_small.render(f"STAGE: {self.stage} (TARGET: {target_score})", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_surf, (20, 50))
        
        # Timer
        time_int = max(0, int(self.time_remaining))
        time_color = (255, 100, 100) if time_int <= 10 else self.COLOR_UI_TEXT
        time_surf = self.font_main.render(f"TIME: {time_int}", True, time_color)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_surf, time_rect)

        # Messages
        if self.message_timer > 0:
            msg_surf = self.font_title.render(self.message_text, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Message background
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    obs, info = env.reset()
    terminated = False
    
    # Use a different screen for display to not interfere with the environment's screen
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tile Matcher")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Shift (unused)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Draw the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")