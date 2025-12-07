import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade puzzle game.
    The player defends an arcade cabinet by matching tiles on a grid.
    Matching tiles charges a power-up that can be used to destroy glitch enemies
    advancing towards the cabinet.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your arcade cabinet by matching tiles. "
        "Matching tiles charges a bomb power-up to destroy incoming glitch enemies."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to swap tiles and shift to use the bomb power-up when charged."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 8
    TILE_SIZE = 40
    GRID_X = (SCREEN_WIDTH - GRID_COLS * TILE_SIZE) // 2 + 80
    GRID_Y = (SCREEN_HEIGHT - GRID_ROWS * TILE_SIZE) // 2
    NUM_TILE_TYPES = 6
    MAX_STEPS = 1000
    MAX_WAVES = 10
    CABINET_MAX_HEALTH = 100

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_GRID_BG = (25, 20, 40)
    COLOR_CABINET = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_SCORE = (255, 255, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 50)
    COLOR_POWER_BAR_BG = (80, 80, 20)
    COLOR_POWER_BAR_FG = (255, 220, 0)
    COLOR_CURSOR = (255, 255, 255)
    TILE_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (255, 50, 255),  # Magenta
        (50, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.enemies = None
        self.particles = None
        self.floating_texts = None
        self.cabinet_health = None
        self.power_up_charge = None
        self.power_up_cost = None
        self.current_wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.shake_magnitude = 0
        self.swap_animation = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cabinet_health = self.CABINET_MAX_HEALTH
        self.power_up_charge = 0
        self.power_up_cost = 15 # Number of tiles to match for power-up
        self.current_wave = 0
        self.enemies = []
        self.particles = []
        self.floating_texts = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_move_dir = (1, 0) # Default to right
        self.shake_magnitude = 0
        self.swap_animation = None # {'from_pos', 'to_pos', 'progress'}

        self._create_grid()
        self._ensure_matches_exist()
        self._spawn_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle Player Input ---
        # 1. Cursor Movement
        self._handle_cursor_movement(movement)

        # 2. Power-up Activation
        if shift_pressed and self.power_up_charge >= self.power_up_cost:
            reward += self._activate_power_up()

        # 3. Tile Swap
        if space_pressed:
            swap_reward = self._attempt_swap()
            reward += swap_reward

        # --- Update Game State ---
        # Update enemies
        damage_taken, enemies_defeated_by_contact = self._update_enemies()
        if damage_taken > 0:
            reward -= damage_taken * 0.5 # Higher penalty for damage
            self.score -= damage_taken * 10
            self.shake_magnitude = 15

        self.score += enemies_defeated_by_contact * 5 # Small bonus for incidental defeats

        # Update effects
        self._update_particles()
        self._update_floating_texts()
        self._update_shake()

        # Check for wave clear
        if not self.enemies and not self.game_over:
            reward += 5
            self.score += 100 * self.current_wave
            self._spawn_next_wave()
            self._create_floating_text(f"WAVE {self.current_wave}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.TILE_COLORS[3], 60)

        # --- Check Termination Conditions ---
        if self.cabinet_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100
            self._create_floating_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.TILE_COLORS[0], 90)
        elif self.current_wave > self.MAX_WAVES:
            terminated = True
            self.game_over = True
            reward = 100
            self.score += 1000
            self._create_floating_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.TILE_COLORS[1], 90)
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            reward = -50 # Penalty for timeout

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_cursor_movement(self, movement):
        moved = False
        prev_pos = list(self.cursor_pos)
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
            self.last_move_dir = (0, -1)
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
            self.last_move_dir = (0, 1)
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
            self.last_move_dir = (-1, 0)
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
            self.last_move_dir = (1, 0)

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        return prev_pos != self.cursor_pos

    def _attempt_swap(self):
        cx, cy = self.cursor_pos
        dx, dy = self.last_move_dir
        nx, ny = cx + dx, cy + dy

        if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
            # Perform swap
            self.grid[cy][cx], self.grid[ny][nx] = self.grid[ny][nx], self.grid[cy][cx]

            matches_found = self._process_matches()
            if matches_found > 0:
                self.shake_magnitude = 5
                return matches_found * 0.1
            else:
                # No match, swap back
                self.grid[cy][cx], self.grid[ny][nx] = self.grid[ny][nx], self.grid[cy][cx]
                self._create_floating_text("INVALID", self._grid_to_screen(cx, cy), (255,100,100), 20)
        return 0

    def _activate_power_up(self):
        self.power_up_charge = 0
        reward = 1.0
        enemies_destroyed = 0
        for enemy in self.enemies:
            self._create_explosion(enemy['pos'], 20, 50)
            enemies_destroyed += 1

        self.enemies.clear()
        self.score += enemies_destroyed * 50
        self.shake_magnitude = 20
        self._create_floating_text("BOMB!", (self.SCREEN_WIDTH // 2, 200), self.TILE_COLORS[0], 60)
        return reward

    def _update_enemies(self):
        damage_taken = 0
        enemies_defeated = 0

        target_pos = (self.GRID_X // 2, self.SCREEN_HEIGHT // 2) # Center of the cabinet

        for enemy in self.enemies[:]:
            # Move enemy
            direction = (target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1])
            dist = math.hypot(*direction)
            if dist > 0:
                enemy['pos'] = (enemy['pos'][0] + direction[0] / dist * enemy['speed'],
                                enemy['pos'][1] + direction[1] / dist * enemy['speed'])

            # Check collision with cabinet
            if enemy['pos'][0] > self.GRID_X - self.TILE_SIZE:
                damage_taken += enemy['damage']
                self.cabinet_health -= enemy['damage']
                self.enemies.remove(enemy)
                self._create_explosion((self.GRID_X - self.TILE_SIZE, enemy['pos'][1]), 15, 30)

        return damage_taken, enemies_defeated

    def _spawn_next_wave(self):
        if self.current_wave >= self.MAX_WAVES:
            return
        self.current_wave += 1

        num_enemies = min(1 + self.current_wave, 5)
        enemy_speed = 1.0 + (self.current_wave // 5) * 0.5
        enemy_health = 10 * self.current_wave
        enemy_damage = 5 + self.current_wave

        for _ in range(num_enemies):
            spawn_y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            self.enemies.append({
                'pos': (-20, spawn_y),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed * self.np_random.uniform(0.8, 1.2),
                'damage': enemy_damage,
                'size': 12,
            })

    def _process_matches(self):
        total_tiles_matched = 0
        chain_multiplier = 1

        while True:
            matches = self._find_all_matches()
            if not matches:
                break

            tiles_in_this_pass = len(matches)
            total_tiles_matched += tiles_in_this_pass

            self.score += tiles_in_this_pass * 10 * chain_multiplier
            self.power_up_charge = min(self.power_up_cost, self.power_up_charge + tiles_in_this_pass)

            for x, y in matches:
                if self.grid[y][x] != -1:
                    screen_pos = self._grid_to_screen(x, y, center=True)
                    self._create_explosion(screen_pos, 10, 20, self.TILE_COLORS[self.grid[y][x]])
                    self.grid[y][x] = -1 # Mark for removal

            self._apply_gravity()
            self._fill_top_rows()

            chain_multiplier += 1
            if chain_multiplier > 2:
                self._create_floating_text(f"CHAIN x{chain_multiplier-1}", self._grid_to_screen(self.GRID_COLS//2, -1), self.TILE_COLORS[3])

        return total_tiles_matched

    # --- Grid Logic ---
    def _create_grid(self):
        self.grid = [[self.np_random.integers(0, self.NUM_TILE_TYPES) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if c < self.GRID_COLS - 2:
                    if self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2] != -1:
                        matches.update([(c, r), (c+1, r), (c+2, r)])
                if r < self.GRID_ROWS - 2:
                    if self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c] != -1:
                        matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    empty_row -= 1

    def _fill_top_rows(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == -1:
                    self.grid[r][c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _ensure_matches_exist(self):
        # Create a temporary grid to avoid modifying the main one
        original_grid = [row[:] for row in self.grid]
        if not self._find_possible_swaps(original_grid):
            self._create_grid()
            self._ensure_matches_exist() # Recurse until a valid grid is made

    def _find_possible_swaps(self, grid):
        swaps = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Swap right
                if c < self.GRID_COLS - 1:
                    grid[r][c], grid[r][c+1] = grid[r][c+1], grid[r][c]
                    if self._find_all_matches_on_grid(grid):
                        swaps.append(((c,r), (c+1,r)))
                    grid[r][c], grid[r][c+1] = grid[r][c+1], grid[r][c] # Swap back
                # Swap down
                if r < self.GRID_ROWS - 1:
                    grid[r][c], grid[r+1][c] = grid[r+1][c], grid[r][c]
                    if self._find_all_matches_on_grid(grid):
                        swaps.append(((c,r), (c,r+1)))
                    grid[r][c], grid[r+1][c] = grid[r+1][c], grid[r][c] # Swap back
        return swaps

    def _find_all_matches_on_grid(self, grid):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if c < self.GRID_COLS - 2 and grid[r][c] == grid[r][c+1] == grid[r][c+2] != -1:
                    return True
                if r < self.GRID_ROWS - 2 and grid[r][c] == grid[r+1][c] == grid[r+2][c] != -1:
                    return True
        return False

    # --- Effects Logic ---
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _update_floating_texts(self):
        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['pos'] = (t['pos'][0], t['pos'][1] - 0.5)
            t['life'] -= 1

    def _update_shake(self):
        if self.shake_magnitude > 0:
            self.shake_magnitude *= 0.9
            if self.shake_magnitude < 0.5:
                self.shake_magnitude = 0

    def _create_explosion(self, pos, num_particles, radius, color=None):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            p_color = color if color else self.TILE_COLORS[self.np_random.integers(0, self.NUM_TILE_TYPES)]
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(2, 5),
                'color': p_color
            })

    def _create_floating_text(self, text, pos, color, lifetime=30, size=30):
        font = pygame.font.Font(None, size)
        self.floating_texts.append({
            'text': text,
            'pos': pos,
            'color': color,
            'life': lifetime,
            'font': font
        })

    # --- Rendering ---
    def _get_observation(self):
        # Create a temporary surface to apply screen shake
        render_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # Draw background and static elements
        render_surface.fill(self.COLOR_BG)
        self._render_cabinet(render_surface)

        # Draw game elements
        self._render_grid(render_surface)
        self._render_enemies(render_surface)
        self._render_particles(render_surface)
        self._render_cursor(render_surface)
        self._render_floating_texts(render_surface)

        # Draw UI on top of everything
        self._render_ui(render_surface)

        # Apply screen shake by blitting the rendered surface with an offset
        shake_offset = (0, 0)
        if self.shake_magnitude > 0:
            shake_offset = (self.np_random.uniform(-self.shake_magnitude, self.shake_magnitude),
                            self.np_random.uniform(-self.shake_magnitude, self.shake_magnitude))

        self.screen.fill(self.COLOR_BG)
        self.screen.blit(render_surface, shake_offset)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_cabinet(self, surface):
        cabinet_rect = pygame.Rect(20, 20, self.GRID_X - 40, self.SCREEN_HEIGHT - 40)
        pygame.draw.rect(surface, self.COLOR_CABINET, cabinet_rect, border_radius=15)
        pygame.draw.rect(surface, (60,60,70), cabinet_rect.inflate(-10, -10), border_radius=10)

        # Decorative lines
        pygame.draw.line(surface, (255,100,255), (cabinet_rect.left+15, cabinet_rect.top), (cabinet_rect.left+15, cabinet_rect.bottom), 2)
        pygame.draw.line(surface, (100,255,255), (cabinet_rect.right-15, cabinet_rect.top), (cabinet_rect.right-15, cabinet_rect.bottom), 2)

    def _render_grid(self, surface):
        grid_area = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_COLS * self.TILE_SIZE, self.GRID_ROWS * self.TILE_SIZE)
        pygame.draw.rect(surface, self.COLOR_GRID_BG, grid_area)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r][c]
                if tile_type != -1:
                    rect = self._get_tile_rect(c, r)
                    color = self.TILE_COLORS[tile_type]

                    # Draw tile with a slight 3D effect
                    shadow_color = tuple(max(0, val - 50) for val in color)
                    pygame.draw.rect(surface, shadow_color, rect, border_radius=8)
                    inner_rect = pygame.Rect(rect.left, rect.top, rect.width-2, rect.height-2)
                    pygame.draw.rect(surface, color, inner_rect, border_radius=8)

    def _render_cursor(self, surface):
        cx, cy = self.cursor_pos
        rect = self._get_tile_rect(cx, cy)
        pygame.draw.rect(surface, self.COLOR_CURSOR, rect, 3, border_radius=10)

    def _render_enemies(self, surface):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = int(enemy['size'])

            # Glitch effect
            for _ in range(3):
                offset_x = self.np_random.uniform(-size/2, size/2)
                offset_y = self.np_random.uniform(-size/2, size/2)
                glitch_pos = (pos[0] + offset_x, pos[1] + offset_y)
                glitch_color = self.TILE_COLORS[self.np_random.integers(0, self.NUM_TILE_TYPES)]
                glitch_size = self.np_random.uniform(size/4, size/2)
                pygame.draw.rect(surface, glitch_color, (glitch_pos[0]-glitch_size/2, glitch_pos[1]-glitch_size/2, glitch_size, glitch_size))

            # Main body
            pygame.draw.rect(surface, self.TILE_COLORS[0], (pos[0]-size/2, pos[1]-size/2, size, size))

    def _render_particles(self, surface):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            if size > 0:
                alpha = int(255 * (p['life'] / 30.0))
                try:
                    color_with_alpha = p['color'] + (alpha,)
                    pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], size, color_with_alpha)
                except (ValueError, TypeError): # Handle potential color format issues
                    pygame.draw.circle(surface, p['color'], pos, size)


    def _render_floating_texts(self, surface):
        for t in self.floating_texts:
            alpha = int(255 * (t['life'] / 30.0))
            if alpha > 0:
                text_surf = t['font'].render(t['text'], True, t['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=t['pos'])
                surface.blit(text_surf, text_rect)

    def _render_ui(self, surface):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        surface.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        surface.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 20, 60))

        # Cabinet Health
        health_bar_rect = pygame.Rect(40, self.SCREEN_HEIGHT - 50, 150, 20)
        pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_BG, health_bar_rect, border_radius=5)
        health_ratio = max(0, self.cabinet_health / self.CABINET_MAX_HEALTH)
        fg_rect = pygame.Rect(health_bar_rect.left, health_bar_rect.top, health_bar_rect.width * health_ratio, health_bar_rect.height)
        pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_FG, fg_rect, border_radius=5)
        health_text = self.font_small.render("CABINET HP", True, self.COLOR_UI_TEXT)
        surface.blit(health_text, (health_bar_rect.centerx - health_text.get_width()//2, health_bar_rect.top - 20))

        # Power-up Meter
        power_bar_rect = pygame.Rect(40, 60, 150, 20)
        pygame.draw.rect(surface, self.COLOR_POWER_BAR_BG, power_bar_rect, border_radius=5)
        power_ratio = min(1.0, self.power_up_charge / self.power_up_cost)
        fg_rect_power = pygame.Rect(power_bar_rect.left, power_bar_rect.top, power_bar_rect.width * power_ratio, power_bar_rect.height)
        pygame.draw.rect(surface, self.COLOR_POWER_BAR_FG, fg_rect_power, border_radius=5)
        power_text = self.font_small.render("BOMB METER", True, self.COLOR_UI_TEXT)
        surface.blit(power_text, (power_bar_rect.centerx - power_text.get_width()//2, power_bar_rect.top - 20))
        if power_ratio >= 1.0:
            ready_text = self.font_small.render("READY! (SHIFT)", True, (255, 255, 255))
            surface.blit(ready_text, (power_bar_rect.centerx - ready_text.get_width()//2, power_bar_rect.bottom + 5))

    def _grid_to_screen(self, c, r, center=False):
        x = self.GRID_X + c * self.TILE_SIZE
        y = self.GRID_Y + r * self.TILE_SIZE
        if center:
            x += self.TILE_SIZE // 2
            y += self.TILE_SIZE // 2
        return x, y

    def _get_tile_rect(self, c, r):
        x, y = self._grid_to_screen(c, r)
        return pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "health": self.cabinet_health}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with video driver

    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Defender")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False

    print(GameEnv.user_guide)

    while not terminated and not truncated:
        # --- Human Input ---
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0
        
        # Poll for events
        events = pygame.event.get()
        
        # Check for quit event
        should_quit = any(event.type == pygame.QUIT for event in events)
        if should_quit:
            break

        # Check for key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # Check for single-press events
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated, truncated = False, False
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}. Press 'R' to restart.")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()