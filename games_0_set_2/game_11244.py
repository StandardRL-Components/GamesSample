import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    CyberGrid Overload: A Gymnasium environment where the agent must survive a
    virtual reality hacking attempt by matching color-coded code segments.
    The agent controls a cursor to select and rotate segments, triggering
    chain reactions to clear the grid. It must manage a depleting timer and
    a rising corruption level, using a limited time-slowing ability to survive.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive a virtual reality hacking attempt by matching color-coded code segments. "
        "Rotate segments to create matches, clear the grid, and manage a depleting timer and rising corruption."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to rotate a segment. "
        "Press shift to activate time warp and slow down the game."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 9, 6
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Colors (Cyberpunk Neon)
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (30, 60, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_TIME_BAR = (0, 200, 255)
    COLOR_TIME_BAR_BG = (30, 60, 80)
    
    SEGMENT_COLORS = [
        (50, 255, 50),   # Green
        (50, 50, 255),   # Blue
        (150, 50, 255),  # Purple
        (255, 255, 50),  # Yellow
    ]
    CORRUPTION_COLORS = {
        'low': (255, 165, 0), # Orange
        'high': (255, 60, 0) # Red
    }

    # Game parameters
    FPS = 60 # Pygame rendering FPS
    STEPS_PER_SECOND = 100 # Gym step rate
    MAX_EPISODE_STEPS = 6000 # 60 seconds * 100 steps/sec
    INITIAL_TIME = 60.0
    INITIAL_SEGMENTS = 20
    MAX_SEGMENTS = 50
    INITIAL_CORRUPTION_RATE = 0.05 / STEPS_PER_SECOND # % per step
    CORRUPTION_RATE_INCREASE = (0.02 / STEPS_PER_SECOND) / 500 # per step, per 500 steps
    TIME_SLOWDOWN_USES = 5
    TIME_SLOWDOWN_DURATION = 1.0 # seconds
    TIME_SLOWDOWN_FACTOR = 0.5
    
    ROTATION_SPEED = 18 # degrees per frame, for visual interpolation
    MOVE_COOLDOWN = 5 # steps
    ROTATE_COOLDOWN = 10 # steps

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.cursor_pos = (0, 0)
        self.time_remaining = 0.0
        self.corruption = 0.0
        self.time_slowdown_uses = 0
        self.time_slowdown_timer = 0.0
        self.particles = []
        self.light_trails = []
        self.move_cooldown_timer = 0
        self.rotate_cooldown_timer = 0
        self.prev_shift_held = False
        self.previous_corruption = 0.0
        self.all_segments_cleared = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.INITIAL_TIME
        self.corruption = 0.0
        self.previous_corruption = 0.0
        self.time_slowdown_uses = self.TIME_SLOWDOWN_USES
        self.time_slowdown_timer = 0.0
        self.particles = []
        self.light_trails = []
        self.move_cooldown_timer = 0
        self.rotate_cooldown_timer = 0
        self.prev_shift_held = False
        self.all_segments_cleared = False
        self.reward_this_step = 0

        self._generate_level()
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- Handle Timers and Cooldowns ---
        dt = 1.0 / self.STEPS_PER_SECOND
        self.time_remaining -= dt
        if self.time_slowdown_timer > 0:
            self.time_slowdown_timer -= dt
        
        self.move_cooldown_timer = max(0, self.move_cooldown_timer - 1)
        self.rotate_cooldown_timer = max(0, self.rotate_cooldown_timer - 1)
        
        # --- Handle Input ---
        self._handle_input(action)
        
        # --- Update Game State ---
        current_corruption_rate = self.INITIAL_CORRUPTION_RATE + (self.steps // 500) * self.CORRUPTION_RATE_INCREASE
        if self.time_slowdown_timer > 0:
            current_corruption_rate *= self.TIME_SLOWDOWN_FACTOR
        
        self.corruption = min(100.0, self.corruption + current_corruption_rate * 100)
        
        if self.steps > 0 and self.steps % 1000 == 0:
            num_segments = sum(1 for row in self.grid for cell in row if cell)
            if num_segments < self.MAX_SEGMENTS:
                self._add_segments(1)

        for row in self.grid:
            for segment in row:
                if segment:
                    segment.update()

        self._update_effects()

        # --- Calculate Reward ---
        corruption_increase = self.corruption - self.previous_corruption
        if corruption_increase > 0:
            reward -= (corruption_increase / 1.0) * 0.01
        self.previous_corruption = self.corruption
        
        reward += self.reward_this_step
        self.reward_this_step = 0

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False
        if terminated and not self.game_over:
            self.game_over = True
            if self.all_segments_cleared:
                reward += 100
                self.score += 5
            else:
                reward -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if self.move_cooldown_timer == 0 and movement != 0:
            x, y = self.cursor_pos
            if movement == 1: y -= 1 # Up
            elif movement == 2: y += 1 # Down
            elif movement == 3: x -= 1 # Left
            elif movement == 4: x += 1 # Right
            self.cursor_pos = (max(0, min(self.GRID_COLS - 1, x)), max(0, min(self.GRID_ROWS - 1, y)))
            self.move_cooldown_timer = self.MOVE_COOLDOWN

        # Rotation
        if space_held and self.rotate_cooldown_timer == 0:
            segment = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
            if segment and not segment.is_rotating and not segment.is_cleared:
                # sfx: rotate_sound
                segment.rotate()
                self.rotate_cooldown_timer = self.ROTATE_COOLDOWN
                cleared_count, cleared_positions = self._check_and_process_matches(self.cursor_pos[0], self.cursor_pos[1])
                if cleared_count > 0:
                    # sfx: match_success_sound
                    self.reward_this_step += cleared_count * 0.1
                    self.score += cleared_count
                    if cleared_count >= 3:
                        self.reward_this_step += 1.0
                    self._create_chain_effect(cleared_positions)

        # Time Slowdown
        if shift_held and not self.prev_shift_held and self.time_slowdown_uses > 0 and self.time_slowdown_timer <= 0:
            # sfx: time_slowdown_activate
            self.time_slowdown_uses -= 1
            self.time_slowdown_timer = self.TIME_SLOWDOWN_DURATION
        self.prev_shift_held = shift_held

    def _check_and_process_matches(self, start_x, start_y):
        start_node = self.grid[start_y][start_x]
        if not start_node:
            return 0, []

        to_visit = [(start_x, start_y)]
        visited = set()
        matches = []

        while to_visit:
            cx, cy = to_visit.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            current_segment = self.grid[cy][cx]
            if not current_segment or current_segment.is_cleared:
                continue

            is_match_found = False
            # Check neighbors: 0=top, 1=right, 2=bottom, 3=left
            neighbors = [(0, -1, 0, 2), (1, 0, 1, 3), (0, 1, 2, 0), (-1, 0, 3, 1)]
            for dx, dy, self_side, neighbor_side in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    neighbor_segment = self.grid[ny][nx]
                    if neighbor_segment and not neighbor_segment.is_cleared:
                        if current_segment.colors[self_side] == neighbor_segment.colors[neighbor_side]:
                            is_match_found = True
                            if (nx, ny) not in visited:
                                to_visit.append((nx, ny))
            
            if is_match_found:
                matches.append((cx, cy))

        if len(matches) > 1: # A match requires at least two segments
            cleared_positions = []
            for x, y in matches:
                segment = self.grid[y][x]
                if segment:
                    segment.clear()
                    for _ in range(20):
                        self.particles.append(Particle(segment.screen_pos[0], segment.screen_pos[1], random.choice(segment.colors)))
                    cleared_positions.append(segment.screen_pos)
            return len(matches), cleared_positions
        return 0, []

    def _check_termination(self):
        num_active_segments = sum(1 for row in self.grid for s in row if s and not s.is_cleared)
        if num_active_segments == 0 and self.steps > 1:
            self.all_segments_cleared = True
            return True
        if self.time_remaining <= 0: return True
        if self.corruption >= 100: return True
        if self.steps >= self.MAX_EPISODE_STEPS: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "corruption": self.corruption, "time_left": self.time_remaining}

    def _render_game(self):
        self._render_grid()
        self._render_effects()
        
        for r, row in enumerate(self.grid):
            for c, segment in enumerate(row):
                if segment:
                    is_selected = (c, r) == self.cursor_pos
                    segment.draw(self.screen, is_selected)
        
        self._render_cursor()
        if self.corruption > 0:
            self._render_corruption_effect()

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        corr_perc = int(self.corruption)
        corr_color = self.CORRUPTION_COLORS['high'] if corr_perc > 75 else self.CORRUPTION_COLORS['low']
        corr_text = self.font_large.render(f"CORRUPTION: {corr_perc}%", True, corr_color)
        self.screen.blit(corr_text, (self.SCREEN_WIDTH - corr_text.get_width() - 10, 10))

        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 15
        time_perc = max(0, self.time_remaining / self.INITIAL_TIME)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width * time_perc, bar_height))

        slowdown_text = self.font_small.render(f"TIME WARP [SHIFT]: {'● ' * self.time_slowdown_uses}{'○ ' * (self.TIME_SLOWDOWN_USES - self.time_slowdown_uses)}", True, self.COLOR_TIME_BAR if self.time_slowdown_timer > 0 else self.COLOR_UI_TEXT)
        self.screen.blit(slowdown_text, (10, self.SCREEN_HEIGHT - bar_height - 35))

    def _generate_level(self):
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self._add_segments(self.INITIAL_SEGMENTS, scramble=True)

    def _add_segments(self, count, scramble=False):
        empty_cells = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] is None:
                    empty_cells.append((c, r))
        
        random.shuffle(empty_cells)
        
        for i in range(min(count, len(empty_cells))):
            c, r = empty_cells[i]
            colors = [random.choice(self.SEGMENT_COLORS) for _ in range(4)]
            # Match with neighbors if they exist to create a solvable base
            if r > 0 and self.grid[r-1][c]: colors[0] = self.grid[r-1][c].colors[2]
            if c > 0 and self.grid[r][c-1]: colors[3] = self.grid[r][c-1].colors[1]
            
            segment = Segment((c, r), colors, self.CELL_SIZE, self.GRID_X, self.GRID_Y, self.ROTATION_SPEED)
            if scramble:
                for _ in range(self.np_random.integers(1, 4)):
                    segment.colors.insert(0, segment.colors.pop())
            self.grid[r][c] = segment

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.gfxdraw.hline(self.screen, self.GRID_X, self.GRID_X + self.GRID_WIDTH, y, self.COLOR_GRID)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.gfxdraw.vline(self.screen, x, self.GRID_Y, self.GRID_Y + self.GRID_HEIGHT, self.COLOR_GRID)

    def _render_cursor(self):
        if self.game_over: return
        x = self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE
        y = self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE
        rect = (x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        alpha = int(100 + pulse * 155)
        
        for i in range(4):
            glow_rect = pygame.Rect(rect).inflate(i*2, i*2)
            color = (*self.COLOR_CURSOR, alpha // (i+1))
            pygame.gfxdraw.rectangle(self.screen, glow_rect, color)

    def _render_corruption_effect(self):
        if self.corruption <= 0: return
        corruption_surface = self.screen.copy()
        corruption_surface.set_colorkey((0,0,0))
        
        shift_amount = int(self.corruption / 100 * 15)
        if shift_amount > 0:
            for i in range(int(self.corruption / 100 * self.SCREEN_HEIGHT / 4)):
                y = self.np_random.integers(0, self.SCREEN_HEIGHT)
                shift = self.np_random.integers(-shift_amount, shift_amount)
                line_rect = pygame.Rect(0, y, self.SCREEN_WIDTH, 1)
                sub = self.screen.subsurface(line_rect)
                corruption_surface.blit(sub, (shift, y))
        
        corruption_surface.set_alpha(int(min(255, self.corruption * 1.5)))
        self.screen.blit(corruption_surface, (0, 0))

    def _create_chain_effect(self, positions):
        if len(positions) < 2: return
        # Create trails between consecutive cleared segments
        for i in range(len(positions) - 1):
            for j in range(i + 1, len(positions)):
                dist = math.hypot(positions[i][0]-positions[j][0], positions[i][1]-positions[j][1])
                if dist < self.CELL_SIZE * 1.5: # Only connect adjacent
                    self.light_trails.append(LightTrail(positions[i], positions[j]))

    def _update_effects(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update(1/self.FPS)

        self.light_trails = [t for t in self.light_trails if t.is_alive()]
        for t in self.light_trails:
            t.update(1/self.FPS)

    def _render_effects(self):
        for t in self.light_trails:
            t.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)

class Segment:
    def __init__(self, grid_pos, colors, size, grid_x, grid_y, rot_speed):
        self.grid_pos = grid_pos
        self.colors = list(colors) # [top, right, bottom, left]
        self.size = size
        self.screen_pos = (grid_x + grid_pos[0] * size + size // 2, grid_y + grid_pos[1] * size + size // 2)
        
        self.angle = 0
        self.target_angle = 0
        self.rotation_speed = rot_speed
        self.is_rotating = False
        
        self.is_cleared = False
        self.clear_alpha = 255

    def rotate(self):
        if self.is_rotating: return
        self.is_rotating = True
        self.target_angle -= 90
        
    def update(self):
        if self.angle != self.target_angle:
            diff = (self.target_angle - self.angle + 180) % 360 - 180
            if abs(diff) < self.rotation_speed:
                self.angle = self.target_angle
                self.is_rotating = False
                self.colors.insert(0, self.colors.pop())
                self.angle %= 360
                self.target_angle %= 360
            else:
                self.angle -= self.rotation_speed
        
        if self.is_cleared:
            self.clear_alpha = max(0, self.clear_alpha - 15)

    def clear(self):
        self.is_cleared = True

    def draw(self, surface, is_selected):
        if self.clear_alpha == 0: return

        cx, cy = self.screen_pos
        rot_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        
        body_color = (40, 80, 110)
        pygame.draw.rect(rot_surface, body_color, (self.size*0.1, self.size*0.1, self.size*0.8, self.size*0.8), border_radius=3)
        
        side_width = self.size * 0.15
        points = [
            [(side_width, side_width), (self.size-side_width, side_width), (self.size*0.3, self.size*0.3), (self.size*0.7, self.size*0.3)],
            [(self.size-side_width, side_width), (self.size-side_width, self.size-side_width), (self.size*0.7, self.size*0.7), (self.size*0.7, self.size*0.3)],
            [(self.size-side_width, self.size-side_width), (side_width, self.size-side_width), (self.size*0.7, self.size*0.7), (self.size*0.3, self.size*0.7)],
            [(side_width, self.size-side_width), (side_width, side_width), (self.size*0.3, self.size*0.3), (self.size*0.3, self.size*0.7)],
        ]
        for i, color in enumerate(self.colors):
            pygame.gfxdraw.aapolygon(rot_surface, points[i], color)
            pygame.gfxdraw.filled_polygon(rot_surface, points[i], color)

        rotated_surface = pygame.transform.rotate(rot_surface, self.angle)
        new_rect = rotated_surface.get_rect(center=(cx, cy))
        
        if self.is_cleared:
            rotated_surface.set_alpha(self.clear_alpha)
        
        surface.blit(rotated_surface, new_rect.topleft)

        if is_selected and not self.is_cleared:
            pulse = (math.sin(pygame.time.get_ticks() * 0.008) + 1) / 2
            color = (255, 255, 255, int(150 + pulse * 105))
            pygame.gfxdraw.rectangle(surface, new_rect.inflate(4, 4), color)

class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(angle) * speed, math.sin(angle) * speed
        self.lifetime = random.uniform(0.4, 1.0)
        self.initial_lifetime = self.lifetime
        self.color = color
        self.radius = random.uniform(2, 5)

    def is_alive(self): return self.lifetime > 0

    def update(self, dt):
        self.lifetime -= dt
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.radius < 1: return
        progress = self.lifetime / self.initial_lifetime
        alpha = int(255 * progress)
        radius = int(self.radius * progress)
        if alpha > 0 and radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))

class LightTrail:
    def __init__(self, start_pos, end_pos, color=(200, 255, 255), duration=0.3):
        self.start_pos, self.end_pos = start_pos, end_pos
        self.color, self.duration, self.lifetime = color, duration, duration

    def is_alive(self): return self.lifetime > 0

    def update(self, dt): self.lifetime -= dt

    def draw(self, surface):
        progress = self.lifetime / self.duration
        alpha = int(255 * progress)
        width = int(max(1, 8 * progress))
        
        if alpha > 0 and width > 0:
            temp_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.line(temp_surf, (*self.color, alpha), self.start_pos, self.end_pos, width)
            surface.blit(temp_surf, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

if __name__ == '__main__':
    # Set the video driver to a real one for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    running = True
    obs, info = env.reset()
    total_reward = 0
    
    key_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4,
    }

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("CyberGrid Overload")
    clock = pygame.time.Clock()
    
    while running:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.STEPS_PER_SECOND)

    pygame.quit()