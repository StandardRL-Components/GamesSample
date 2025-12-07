import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a puzzle game called "Color Sync".
    The player moves a cursor on a grid and places colored prediction markers
    to match falling blocks. The goal is to score points by correctly
    predicting where blocks will land and to get bonus points for
    synchronizing multiple matches of the same color.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place colored markers on a grid to match falling blocks. "
        "Score points for correct predictions and get bonuses for synchronizing multiple matches."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a prediction marker and shift to remove one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Grid Configuration ---
        self.GRID_ROWS, self.GRID_COLS = 3, 4
        self.GRID_MARGIN_X = 120
        self.GRID_MARGIN_Y = 40
        self.GRID_WIDTH = self.WIDTH - 2 * self.GRID_MARGIN_X
        self.GRID_HEIGHT = self.HEIGHT - 2 * self.GRID_MARGIN_Y
        self.CELL_WIDTH = self.GRID_WIDTH / self.GRID_COLS
        self.CELL_HEIGHT = self.GRID_HEIGHT / self.GRID_ROWS
        self.GRID_TOP = self.GRID_MARGIN_Y
        self.GRID_BOTTOM = self.HEIGHT - self.GRID_MARGIN_Y

        # --- Visuals ---
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_GRID = (50, 50, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.PALETTE = [
            (255, 70, 70),   # Red
            (70, 255, 70),   # Green
            (70, 130, 255),  # Blue
            (255, 220, 70),  # Yellow
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.font_tiny = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.player_cursor = [0, 0]  # [col, row]
        self.falling_objects = []
        self.prediction_markers = {} # (col, row) -> color_index
        self.particles = []
        self.prev_action = [0, 0, 0]
        self.spawn_timer = 0
        self.spawn_interval = self.FPS  # 1 per second
        self.upcoming_colors = []

    def _get_grid_pixel_pos(self, col, row, center=True):
        """Converts grid coordinates to pixel coordinates."""
        x = self.GRID_MARGIN_X + col * self.CELL_WIDTH
        y = self.GRID_MARGIN_Y + row * self.CELL_HEIGHT
        if center:
            x += self.CELL_WIDTH / 2
            y += self.CELL_HEIGHT / 2
        return int(x), int(y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.player_cursor = [0, 0]
        self.falling_objects = []
        self.prediction_markers = {}
        self.particles = []
        self.prev_action = [0, 0, 0]
        self.spawn_timer = 0
        self.spawn_interval = self.FPS
        self.upcoming_colors = [self.np_random.integers(0, len(self.PALETTE)) for _ in range(5)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        # 1. Handle player input
        self._handle_input(action)

        # 2. Update game state
        self._update_difficulty()
        self._spawn_objects()
        landed_objects = self._update_objects()

        # 3. Process landings and calculate reward
        reward = self._process_landings(landed_objects)
        self.score += reward

        # 4. Update visual effects
        self._update_particles()

        # 5. Check for termination
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        truncated = False

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_just_pressed = space_held and not (self.prev_action[1] == 1)
        shift_just_pressed = shift_held and not (self.prev_action[2] == 1)

        # --- Cursor Movement (with wrap-around) ---
        if movement == 1:  # Up
            self.player_cursor[1] = (self.player_cursor[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2:  # Down
            self.player_cursor[1] = (self.player_cursor[1] + 1) % self.GRID_ROWS
        elif movement == 3:  # Left
            self.player_cursor[0] = (self.player_cursor[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4:  # Right
            self.player_cursor[0] = (self.player_cursor[0] + 1) % self.GRID_COLS

        # --- Place Marker ---
        if space_just_pressed:
            cursor_pos = tuple(self.player_cursor)
            marker_color_idx = self.upcoming_colors[0]
            self.prediction_markers[cursor_pos] = marker_color_idx

        # --- Remove Marker ---
        if shift_just_pressed:
            cursor_pos = tuple(self.player_cursor)
            if cursor_pos in self.prediction_markers:
                del self.prediction_markers[cursor_pos]

        self.prev_action = action

    def _update_difficulty(self):
        # Increase spawn frequency every 10 seconds
        ten_second_intervals = self.steps // (10 * self.FPS)
        # 0.1 increase per 10s -> 1/0.9, 1/0.8 etc. spawn rate
        base_rate = 1.0
        new_rate = base_rate + ten_second_intervals * 0.1
        self.spawn_interval = int(self.FPS / new_rate)

    def _spawn_objects(self):
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            target_col = self.np_random.integers(0, self.GRID_COLS)
            color_idx = self.upcoming_colors.pop(0)
            self.upcoming_colors.append(self.np_random.integers(0, len(self.PALETTE)))

            x_pos = self._get_grid_pixel_pos(target_col, 0, center=True)[0]
            
            new_object = {
                "pos": pygame.math.Vector2(x_pos, -self.CELL_HEIGHT / 2),
                "speed": 2.0,
                "color_idx": color_idx,
                "target_col": target_col
            }
            self.falling_objects.append(new_object)

    def _update_objects(self):
        landed_objects = []
        remaining_objects = []
        for obj in self.falling_objects:
            obj["pos"].y += obj["speed"]
            if obj["pos"].y >= self.GRID_BOTTOM - self.CELL_HEIGHT / 2:
                landed_objects.append(obj)
            else:
                remaining_objects.append(obj)
        self.falling_objects = remaining_objects
        return landed_objects

    def _process_landings(self, landed_objects):
        if not landed_objects:
            return 0

        reward = 0
        matches_by_color = {i: [] for i in range(len(self.PALETTE))}

        for obj in landed_objects:
            # Determine landing row (always the last one)
            landing_pos = (obj["target_col"], self.GRID_ROWS - 1)
            
            if landing_pos in self.prediction_markers and self.prediction_markers[landing_pos] == obj["color_idx"]:
                matches_by_color[obj["color_idx"]].append(landing_pos)

        for color_idx, positions in matches_by_color.items():
            count = len(positions)
            if count > 0:
                reward += count  # +1 for each individual match
                if count >= 3:
                    reward += 5  # +5 bonus for sync of 3 or more
                
                # Create visual effects and remove used markers
                for pos in positions:
                    pixel_pos = self._get_grid_pixel_pos(pos[0], pos[1])
                    self._create_particles(pixel_pos, self.PALETTE[color_idx], 30)
                    if pos in self.prediction_markers:
                        del self.prediction_markers[pos]
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": vel,
                "color": color,
                "lifetime": self.np_random.integers(20, 40),
                "radius": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Drag
            p["lifetime"] -= 1
            p["radius"] -= 0.1
            if p["lifetime"] > 0 and p["radius"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_markers()
        self._render_cursor()
        self._render_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        # Vertical lines
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_MARGIN_X + c * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_TOP), (x, self.GRID_BOTTOM), 2)
        # Horizontal lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_MARGIN_Y + r * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, y), (self.WIDTH - self.GRID_MARGIN_X, y), 2)

    def _render_markers(self):
        for pos, color_idx in self.prediction_markers.items():
            px, py = self._get_grid_pixel_pos(pos[0], pos[1])
            color = self.PALETTE[color_idx]
            radius = int(self.CELL_WIDTH * 0.3)
            # Draw semi-transparent filled circle
            target_rect = pygame.Rect(px - radius, py - radius, 2 * radius, 2 * radius)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(shape_surf, color + (100,), (radius, radius), radius)
            self.screen.blit(shape_surf, target_rect)
            # Draw solid outline
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

    def _render_cursor(self):
        col, row = self.player_cursor
        x, y = self._get_grid_pixel_pos(col, row, center=False)
        rect = pygame.Rect(x, y, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        # Pulsating glow effect
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # 0 to 1
        glow_size = int(pulse * 8 + 4)
        glow_alpha = int(pulse * 100 + 50)
        
        glow_surf = pygame.Surface((self.CELL_WIDTH + glow_size * 2, self.CELL_HEIGHT + glow_size * 2), pygame.SRCALPHA)
        glow_rect = glow_surf.get_rect()
        pygame.draw.rect(glow_surf, self.COLOR_CURSOR + (glow_alpha,), glow_rect, border_radius=10)
        self.screen.blit(glow_surf, (x - glow_size, y - glow_size))
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=8)

    def _render_objects(self):
        size = int(self.CELL_WIDTH * 0.6)
        for obj in self.falling_objects:
            color = self.PALETTE[obj["color_idx"]]
            x, y = int(obj["pos"].x), int(obj["pos"].y)
            rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 2, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.draw.circle(self.screen, p["color"], pos, int(p["radius"]))

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # --- Time ---
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # --- Upcoming Colors Queue ---
        queue_x = self.WIDTH - self.GRID_MARGIN_X / 2
        queue_y_start = self.GRID_TOP + 30
        
        next_text = self.font_tiny.render("NEXT", True, self.COLOR_UI_TEXT)
        next_rect = next_text.get_rect(centerx=queue_x, bottom=queue_y_start - 5)
        self.screen.blit(next_text, next_rect)

        for i, color_idx in enumerate(self.upcoming_colors):
            color = self.PALETTE[color_idx]
            size = int(self.CELL_WIDTH * (0.6 - i * 0.05))
            y_offset = sum(int(self.CELL_WIDTH * (0.6 - j * 0.05)) + 10 for j in range(i))
            y = queue_y_start + y_offset + size // 2
            
            rect = pygame.Rect(queue_x - size//2, y - size//2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 2, border_radius=5)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to unset the dummy video driver to see the window
    os.environ.pop("SDL_VIDEODRIVER", None)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Color Sync")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        # --- Movement Handling ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        else:
            movement = 0
        
        # --- Step Environment ---
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already the rendered screen, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()