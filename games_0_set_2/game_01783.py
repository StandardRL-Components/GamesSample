
# Generated: 2025-08-28T02:42:25.880078
# Source Brief: brief_01783.md
# Brief Index: 1783

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to select a mirror, ↑↓ to rotate it. Press space to fire the laser."
    )

    game_description = (
        "A real-time puzzle game. Rotate mirrors to guide a laser beam and activate all targets before time runs out."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.FPS = 30

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors and Fonts
        self._setup_visuals()

        # Game constants
        self.MAX_STEPS = 1000  # Approx 33.3 seconds at 30 FPS
        self.LASER_DURATION = 5 # Frames the laser stays visible
        self.NUM_TARGETS = 6
        
        # Pre-defined levels to ensure solvability
        self._create_levels()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.emitter = {}
        self.mirrors = []
        self.targets = []
        self.walls = []
        self.selected_mirror_idx = 0
        self.laser_path = []
        self.laser_active_timer = 0
        self.activated_targets = set()

        self.np_random = None

        self.validate_implementation()

    def _setup_visuals(self):
        """Define colors and fonts for the game."""
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_EMITTER = (255, 100, 0)
        
        self.COLOR_TARGET_OFF = (0, 100, 200)
        self.COLOR_TARGET_ON = (0, 255, 255)
        self.COLOR_TARGET_ON_GLOW = (150, 255, 255)

        self.COLOR_MIRROR = (180, 180, 200)
        self.COLOR_SELECTOR = (255, 255, 0)

        self.COLOR_LASER = (50, 255, 50)
        self.COLOR_LASER_GLOW = (150, 255, 150)

        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_msg = pygame.font.SysFont("Consolas", 60, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_msg = pygame.font.Font(None, 70)
            
    def _create_levels(self):
        """Define several pre-built, solvable level layouts."""
        self.LEVELS = [
            {
                "emitter": {"pos": (0, 5), "dir": (1, 0)},
                "mirrors": [(3, 5, 90), (3, 2, 0), (8, 2, 90), (8, 8, 0), (11, 8, 90)],
                "targets": [(3, 0), (6, 2), (8, 5), (1, 8), (11, 1), (14, 8)],
                "walls": [(5, r) for r in range(10) if r != 2] + [(13, r) for r in range(10) if r != 8]
            },
            {
                "emitter": {"pos": (8, 9), "dir": (0, -1)},
                "mirrors": [(8, 6, 0), (4, 6, 90), (4, 2, 0), (12, 2, 90), (12, 6, 0)],
                "targets": [(8, 1), (1, 6), (4, 8), (12, 0), (15, 2), (12, 9)],
                "walls": [(c, 4) for c in range(16)]
            },
            {
                "emitter": {"pos": (0, 0), "dir": (1, 0)},
                "mirrors": [(4, 0, 90), (4, 4, 0), (9, 4, 90), (9, 8, 0), (2, 8, 90)],
                "targets": [(7, 0), (4, 2), (1, 4), (9, 6), (12, 8), (2, 5)],
                "walls": [(c, 2) for c in range(3, 13)] + [(c, 6) for c in range(3, 13)]
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        level_data = self.np_random.choice(self.LEVELS)
        self.emitter = dict(level_data["emitter"])
        self.mirrors = [{"pos": m[:2], "angle": m[2]} for m in level_data["mirrors"]]
        self.targets = [{"pos": t, "activated": False} for t in level_data["targets"]]
        self.walls = list(level_data["walls"])
        
        self.selected_mirror_idx = 0
        self.laser_path = []
        self.laser_active_timer = 0
        self.activated_targets = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage speed
        
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Action Handling ---
            if movement == 1: # Up: Rotate CW
                self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] + 90) % 360
            elif movement == 2: # Down: Rotate CCW
                self.mirrors[self.selected_mirror_idx]['angle'] = (self.mirrors[self.selected_mirror_idx]['angle'] - 90 + 360) % 360
            elif movement == 3: # Left: Prev mirror
                self.selected_mirror_idx = (self.selected_mirror_idx - 1 + len(self.mirrors)) % len(self.mirrors)
            elif movement == 4: # Right: Next mirror
                self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)

            if space_held and self.laser_active_timer == 0:
                # Fire laser
                # sfx: laser_fire.wav
                self.laser_active_timer = self.LASER_DURATION
                self.laser_path, new_hits = self._calculate_laser_path()
                
                newly_activated = new_hits - self.activated_targets
                if newly_activated:
                    # sfx: target_hit.wav
                    reward += len(newly_activated) * 5.0
                    self.score += len(newly_activated) * 5.0
                    self.activated_targets.update(newly_activated)
                    for i in newly_activated:
                        self.targets[i]["activated"] = True

            # --- Game State Update ---
            if self.laser_active_timer > 0:
                self.laser_active_timer -= 1
            else:
                self.laser_path = []

            # --- Termination Check ---
            if len(self.activated_targets) == len(self.targets):
                # sfx: victory.wav
                self.game_over = True
                self.win = True
                reward += 50.0
                self.score += 50.0
            elif self.steps >= self.MAX_STEPS - 1:
                # sfx: failure.wav
                self.game_over = True
                self.win = False
                reward -= 50.0
                self.score -= 50.0
        
        self.steps += 1
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _calculate_laser_path(self):
        path_points = []
        hit_targets = set()
        
        pos = np.array(self._grid_to_pixel(self.emitter['pos']), dtype=float)
        direction = np.array(self.emitter['dir'], dtype=float)
        path_points.append(tuple(pos.astype(int)))
        
        mirror_map = {m['pos']: m for m in self.mirrors}
        target_map = {t['pos']: i for i, t in enumerate(self.targets)}
        wall_set = set(self.walls)

        for _ in range(self.GRID_COLS * self.GRID_ROWS): # Max path segments
            current_grid_pos = tuple(self._pixel_to_grid(pos))
            
            # Step one grid cell at a time
            next_pos = pos + direction * self.CELL_SIZE
            next_grid_pos = tuple(self._pixel_to_grid(next_pos))
            
            # Check for out of bounds
            if not (0 <= next_grid_pos[0] < self.GRID_COLS and 0 <= next_grid_pos[1] < self.GRID_ROWS):
                path_points.append(tuple(next_pos.astype(int)))
                break

            pos = next_pos
            path_points.append(tuple(pos.astype(int)))

            # Check for collisions in the new cell
            if next_grid_pos in wall_set:
                break
            
            if next_grid_pos in target_map:
                hit_targets.add(target_map[next_grid_pos])

            if next_grid_pos in mirror_map:
                mirror = mirror_map[next_grid_pos]
                angle = mirror['angle']
                dx, dy = int(direction[0]), int(direction[1])

                # Reflection logic
                if angle in [0, 180]: # '/' shape
                    direction = np.array([-dy, -dx], dtype=float)
                elif angle in [90, 270]: # '\' shape
                    direction = np.array([dy, dx], dtype=float)
                
                # sfx: laser_reflect.wav
                continue
                
        return path_points, hit_targets

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Draw walls
        for wx, wy in self.walls:
            rect = pygame.Rect(wx * self.CELL_SIZE, wy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw emitter
        ex, ey = self._grid_to_pixel(self.emitter['pos'])
        pygame.gfxdraw.filled_circle(self.screen, ex, ey, self.CELL_SIZE // 3, self.COLOR_EMITTER)
        pygame.gfxdraw.aacircle(self.screen, ex, ey, self.CELL_SIZE // 3, self.COLOR_EMITTER)

        # Draw targets
        for target in self.targets:
            tx, ty = self._grid_to_pixel(target['pos'])
            color = self.COLOR_TARGET_ON if target['activated'] else self.COLOR_TARGET_OFF
            glow_color = self.COLOR_TARGET_ON_GLOW if target['activated'] else None
            if glow_color:
                pygame.gfxdraw.filled_circle(self.screen, tx, ty, self.CELL_SIZE // 2, glow_color)
                pygame.gfxdraw.aacircle(self.screen, tx, ty, self.CELL_SIZE // 2, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, tx, ty, self.CELL_SIZE // 3, color)
            pygame.gfxdraw.aacircle(self.screen, tx, ty, self.CELL_SIZE // 3, color)

        # Draw mirrors and selector
        for i, mirror in enumerate(self.mirrors):
            mx, my = self._grid_to_pixel(mirror['pos'])
            rect = pygame.Rect(mx - self.CELL_SIZE//2, my - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            
            # Draw selector
            if i == self.selected_mirror_idx and not self.game_over:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = int(self.CELL_SIZE * 0.5 * (1 + pulse * 0.1))
                alpha = int(100 + pulse * 100)
                sel_color = (*self.COLOR_SELECTOR, alpha)
                pygame.gfxdraw.aacircle(self.screen, mx, my, radius, sel_color)
                pygame.gfxdraw.aacircle(self.screen, mx, my, radius-1, sel_color)

            # Draw mirror line
            angle_rad = math.radians(mirror['angle'] + 45)
            half_diag = self.CELL_SIZE * 0.6
            start_pos = (mx - half_diag * math.cos(angle_rad), my - half_diag * math.sin(angle_rad))
            end_pos = (mx + half_diag * math.cos(angle_rad), my + half_diag * math.sin(angle_rad))
            pygame.draw.aaline(self.screen, self.COLOR_MIRROR, start_pos, end_pos, 3)

        # Draw laser
        if self.laser_path and len(self.laser_path) > 1:
            # Glow effect
            pygame.draw.lines(self.screen, self.COLOR_LASER_GLOW, False, self.laser_path, width=8)
            # Core beam
            pygame.draw.lines(self.screen, self.COLOR_LASER, False, self.laser_path, width=4)

    def _render_ui(self):
        # Targets activated
        targets_text = f"TARGETS: {len(self.activated_targets)}/{len(self.targets)}"
        text_surf = self.font_ui.render(targets_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 5))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {max(0, time_left):.1f}"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(text_surf, text_rect)

        # Game over message
        if self.game_over:
            msg = "VICTORY!" if self.win else "TIME UP"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "activated_targets": len(self.activated_targets),
            "total_targets": len(self.targets)
        }

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates (col, row) to pixel coordinates (center of cell)."""
        px = int((grid_pos[0] + 0.5) * self.CELL_SIZE)
        py = int((grid_pos[1] + 0.5) * self.CELL_SIZE)
        return px, py

    def _pixel_to_grid(self, pixel_pos):
        """Converts pixel coordinates to grid coordinates."""
        gx = int(pixel_pos[0] // self.CELL_SIZE)
        gy = int(pixel_pos[1] // self.CELL_SIZE)
        return gx, gy

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Labyrinth")
    clock = pygame.time.Clock()

    while running:
        # --- Action gathering for human play ---
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        # --- Gym step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Reset if done ---
        if done:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            done = False
            total_reward = 0

        clock.tick(env.FPS)

    env.close()