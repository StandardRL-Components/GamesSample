
# Generated: 2025-08-27T22:17:06.440180
# Source Brief: brief_03070.md
# Brief Index: 3070

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move cursor. Space to select a tile group. Clear the board before time runs out!"
    )

    game_description = (
        "A fast-paced puzzle game. Match 3 or more adjacent colored tiles to clear them. "
        "Create chain reactions and clear the board to maximize your score."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    TILE_SIZE = 40
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINES = (40, 60, 80)
    COLOR_EMPTY = (30, 45, 60)
    TILE_COLORS = {
        1: (220, 50, 50),   # Red
        2: (50, 220, 50),   # Green
        3: (50, 100, 220),  # Blue
        4: (220, 220, 50)   # Yellow
    }
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    NUM_TILE_TYPES = len(TILE_COLORS)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.grid = None
        self.cursor_pos = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.board_cleared = None
        self.space_was_held = None
        self.move_cooldown = None
        self.particles = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.board_cleared = False
        self.timer = self.GAME_DURATION_SECONDS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.space_was_held = True # Prevent action on first frame
        self.move_cooldown = 0
        self.particles = []
        
        while True:
            self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_COLS, self.GRID_ROWS))
            if self._is_board_solvable(self.grid):
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self.timer = max(0, self.timer - 1.0 / self.FPS)
            reward -= 0.1 # Per-step time penalty

            # --- Handle Input ---
            self.move_cooldown = max(0, self.move_cooldown - 1)
            if self.move_cooldown == 0:
                moved = self._handle_movement(movement)
                if moved:
                    self.move_cooldown = 5 # 5 frames cooldown

            space_pressed = space_held and not self.space_was_held
            if space_pressed:
                match_reward = self._handle_selection()
                reward += match_reward

            self.space_was_held = space_held

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.board_cleared:
                reward += 100 # Goal reward
                # sound: win_sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        moved = True
        x, y = self.cursor_pos
        if movement == 1: # Up
            self.cursor_pos[1] = (y - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[1] = (y + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[0] = (x - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[0] = (x + 1) % self.GRID_COLS
        else:
            moved = False
        
        if moved:
            # sound: cursor_move_sound
            pass
        return moved

    def _handle_selection(self):
        x, y = self.cursor_pos
        if self.grid[x, y] == 0: # Cannot select empty tile
            return 0

        matches = self._find_matches(x, y)
        
        if len(matches) < 3:
            # sound: fail_sound
            return 0

        # sound: match_sound
        reward = len(matches) # +1 per tile
        self.score += len(matches)
        
        # Check for cleared rows/columns before gravity
        rows_affected = {r for c, r in matches}
        cols_affected = {c for c, r in matches}

        for c, r in matches:
            self._create_particles(c, r, self.grid[c, r])
            self.grid[c, r] = 0

        for r_idx in rows_affected:
            if all(self.grid[c_idx, r_idx] == 0 for c_idx in range(self.GRID_COLS)):
                reward += 5
                self.score += 5
        
        for c_idx in cols_affected:
            if all(self.grid[c_idx, r_idx] == 0 for r_idx in range(self.GRID_ROWS)):
                reward += 5
                self.score += 5

        self._apply_gravity_and_refill()

        # Check for board clear
        if np.all(self.grid == 0):
            self.board_cleared = True
            
        # Check if the new board is solvable, reshuffle if not
        if not self.board_cleared and not self._is_board_solvable(self.grid):
            # This is a rare case, but prevents softlocks
            while not self._is_board_solvable(self.grid):
                self._apply_gravity_and_refill(force_refill=True)

        return reward

    def _find_matches(self, start_c, start_r):
        if self.grid[start_c, start_r] == 0:
            return set()
            
        target_color = self.grid[start_c, start_r]
        q = deque([(start_c, start_r)])
        visited = {(start_c, start_r)}
        
        while q:
            c, r = q.popleft()
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.GRID_COLS and 0 <= nr < self.GRID_ROWS and (nc, nr) not in visited:
                    if self.grid[nc, nr] == target_color:
                        visited.add((nc, nr))
                        q.append((nc, nr))
        return visited

    def _apply_gravity_and_refill(self, force_refill=False):
        for c in range(self.GRID_COLS):
            empty_r = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[c, r] != 0:
                    if r != empty_r:
                        self.grid[c, empty_r] = self.grid[c, r]
                        self.grid[c, r] = 0
                    empty_r -= 1
        
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[c, r] == 0 or force_refill:
                    self.grid[c, r] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

    def _is_board_solvable(self, grid):
        if np.all(grid == 0):
            return True
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if grid[c,r] != 0 and len(self._find_matches_on_grid(grid, c, r)) >= 3:
                    return True
        return False

    def _find_matches_on_grid(self, grid, start_c, start_r):
        target_color = grid[start_c, start_r]
        q = deque([(start_c, start_r)])
        visited = {(start_c, start_r)}
        while q:
            c, r = q.popleft()
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.GRID_COLS and 0 <= nr < self.GRID_ROWS and (nc, nr) not in visited:
                    if grid[nc, nr] == target_color:
                        visited.add((nc, nr))
                        q.append((nc, nr))
        return visited

    def _check_termination(self):
        return self.timer <= 0 or self.board_cleared or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.timer}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_tiles()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.grid[c, r]
                tile_rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.TILE_SIZE,
                    self.GRID_Y_OFFSET + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                color = self.COLOR_EMPTY if tile_val == 0 else self.TILE_COLORS[tile_val]
                
                # Use gfxdraw for antialiasing
                pygame.gfxdraw.box(self.screen, tile_rect.inflate(-4, -4), color)
                darker_color = tuple(max(0, x - 40) for x in color)
                pygame.gfxdraw.rectangle(self.screen, tile_rect.inflate(-4, -4), darker_color)

    def _render_cursor(self):
        if self.game_over: return
        
        c, r = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + c * self.TILE_SIZE,
            self.GRID_Y_OFFSET + r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        
        # Flashing effect
        flash_alpha = 128 + 127 * math.sin(self.steps * 0.4)
        color = (*self.COLOR_CURSOR, flash_alpha)
        
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect(), width=3, border_radius=3)
        self.screen.blit(s, cursor_rect.topleft)
        
    def _create_particles(self, c, r, tile_type):
        px = self.GRID_X_OFFSET + c * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_Y_OFFSET + r * self.TILE_SIZE + self.TILE_SIZE // 2
        base_color = self.TILE_COLORS[tile_type]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [px, py],
                "vel": vel,
                "radius": random.uniform(3, 7),
                "lifespan": random.randint(15, 25),
                "color": base_color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {math.ceil(self.timer):02d}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.board_cleared:
            msg = "BOARD CLEARED!"
        elif self.timer <= 0:
            msg = "TIME'S UP!"
        else:
            msg = "GAME OVER"
        
        text = self.font_game_over.render(msg, True, (255, 220, 100))
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Tile Match Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # In a real training loop, you would just call env.reset()
            # For human play, we can add a delay or wait for a key press
            pass

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()