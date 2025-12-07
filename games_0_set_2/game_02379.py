import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Shift to place a rock. Press Space to rake the sand."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A tranquil zen garden simulation. Rake sand and place rocks to achieve a high aesthetic score before time runs out."
    )

    # Frames auto-advance for the timer mechanic.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 3600  # 120 seconds * 30fps
    WIN_SCORE_PERCENT = 80

    GRID_W, GRID_H = 40, 26
    MAX_ROCKS = 7

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_SAND = (210, 180, 140)
    COLOR_ROCK = (80, 80, 85)
    COLOR_ROCK_SHADOW = (60, 60, 65)
    COLOR_WALL_TOP = (89, 59, 29)
    COLOR_WALL_SIDE = (69, 39, 9)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 40, 50, 200)
    COLOR_CURSOR = (0, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('sans', 20, bold=True)
        self.font_title = pygame.font.SysFont('serif', 24, bold=True)

        # Isometric projection setup
        self.tile_w_iso = 16
        self.tile_h_iso = 8
        self.origin_x = self.WIDTH // 2
        self.origin_y = 100

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.last_score = 0.0
        self.game_over = False
        self.sand_grid = None
        self.rocks = None
        self.cursor_pos = None
        self.last_movement_dir = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.last_score = 0
        self.game_over = False

        # Initialize sand with subtle, natural-looking noise
        self.sand_grid = self.np_random.uniform(-0.1, 0.1, size=(self.GRID_W, self.GRID_H))
        
        self.rocks = []
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.last_movement_dir = 1  # Default up
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self.score = self._calculate_aesthetic_score()
        self.last_score = self.score

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Update cursor position
        if movement != 0:
            self.last_movement_dir = movement
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)

        # 2. Handle actions (on press)
        # Rake action
        if space_held and not self.last_space_held:
            self._rake_sand()
            # sfx: gentle sand scraping sound

        # Place rock action
        if shift_held and not self.last_shift_held:
            if self._place_rock():
                # Reward for successfully placing a rock
                reward += 5.0
                # sfx: soft 'thud' of rock on sand

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # 3. Update game state
        self.steps += 1
        self._update_particles()
        
        # 4. Calculate score and reward
        self.score = self._calculate_aesthetic_score()
        reward += (self.score - self.last_score) * 1.0 # Scaled continuous feedback
        self.last_score = self.score

        # 5. Check for termination
        terminated = bool(self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE_PERCENT)
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE_PERCENT:
                reward += 100.0 # Goal-oriented reward
            else:
                reward -= 10.0 # Penalty for running out of time

        truncated = False
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()
    
    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.origin_x + (x - y) * self.tile_w_iso / 2
        screen_y = self.origin_y + (x + y) * self.tile_h_iso / 2
        return int(screen_x), int(screen_y)

    def _rake_sand(self):
        """Modifies the sand grid to create a rake pattern."""
        rake_len = 7
        rake_width = 3
        rake_amp = 0.5
        
        cx, cy = self.cursor_pos
        
        # Create particles for visual feedback
        for _ in range(10):
            pos = self._iso_to_screen(cx + self.np_random.uniform(-1, 1), cy + self.np_random.uniform(-1, 1))
            self.particles.append([list(pos), self.np_random.uniform(-0.5, 0.5, 2).tolist(), self.np_random.integers(15, 30)])

        # Apply rake pattern based on last movement direction
        for i in range(-rake_len // 2, rake_len // 2 + 1):
            for j in range(-rake_width // 2, rake_width // 2 + 1):
                if self.last_movement_dir in [1, 2]: # Up/Down
                    px, py = cx + j, cy + i
                    val = math.sin(cy * 0.8 + i * 0.8) * rake_amp
                else: # Left/Right
                    px, py = cx + i, cy + j
                    val = math.sin(cx * 0.8 + i * 0.8) * rake_amp

                if 0 <= px < self.GRID_W and 0 <= py < self.GRID_H:
                    # Blend with existing pattern
                    self.sand_grid[px, py] = self.sand_grid[px, py] * 0.3 + val * 0.7

    def _place_rock(self):
        """Places a rock at the cursor's position if possible."""
        if len(self.rocks) >= self.MAX_ROCKS:
            return False
        
        pos = tuple(self.cursor_pos)
        if pos in [r['pos'] for r in self.rocks]:
            return False

        # Create a new rock with a unique shape
        rock_shape = []
        num_verts = self.np_random.integers(5, 8)
        base_radius = self.np_random.uniform(0.4, 0.6)
        for i in range(num_verts):
            angle = 2 * math.pi * i / num_verts
            radius = base_radius * self.np_random.uniform(0.8, 1.2)
            rock_shape.append((math.cos(angle) * radius, math.sin(angle) * radius * 0.7)) # Y is squashed for iso view

        self.rocks.append({'pos': pos, 'shape': rock_shape, 'size': base_radius})
        return True

    def _calculate_aesthetic_score(self):
        """Calculates the aesthetic score based on sand patterns and rock placement."""
        # --- Sand Score (max 50) ---
        # 1. Coverage Score (up to 20 pts): How much of the sand is raked?
        raked_cells = np.count_nonzero(np.abs(self.sand_grid) > 0.15)
        coverage_score = 20 * (raked_cells / self.sand_grid.size)

        # 2. Smoothness Score (up to 30 pts): How smooth are the rake patterns?
        # Lower gradient magnitude = smoother patterns
        grad_x = np.mean(np.abs(np.diff(self.sand_grid, axis=0)))
        grad_y = np.mean(np.abs(np.diff(self.sand_grid, axis=1)))
        smoothness_score = 30 * max(0, 1 - (grad_x + grad_y))
        sand_score = coverage_score + smoothness_score

        # --- Rock Score (max 50) ---
        num_rocks = len(self.rocks)
        if num_rocks == 0:
            return sand_score

        # 1. Quantity Score (up to 15 pts): Ideal number of rocks is 3 or 5.
        quantity_score = 15 * max(0, 1 - abs(num_rocks - 3.5) / 3.5)

        # 2. Balance/Composition Score (up to 20 pts): Are rocks placed harmoniously?
        # Rule of thirds
        thirds_score = 0
        grid_center = np.array([self.GRID_W / 2, self.GRID_H / 2])
        total_dist_from_center = 0
        
        thirds_x = [self.GRID_W / 3, 2 * self.GRID_W / 3]
        thirds_y = [self.GRID_H / 3, 2 * self.GRID_H / 3]
        
        for rock in self.rocks:
            rx, ry = rock['pos']
            total_dist_from_center += np.linalg.norm(np.array(rock['pos']) - grid_center)
            if any(abs(rx - tx) < 3 for tx in thirds_x) or any(abs(ry - ty) < 3 for ty in thirds_y):
                thirds_score += 5
        
        avg_dist_from_center = total_dist_from_center / num_rocks if num_rocks > 0 else 0
        max_dist = np.linalg.norm(grid_center)
        balance_score = 10 * (1 - avg_dist_from_center / max_dist) # Center-biased balance
        composition_score = min(10, thirds_score) + balance_score # Rule of thirds + balance

        # 3. Spacing Score (up to 15 pts): Are rocks not too clustered?
        spacing_score = 15
        if num_rocks > 1:
            min_dist = float('inf')
            positions = [np.array(r['pos']) for r in self.rocks]
            for i in range(num_rocks):
                for j in range(i + 1, num_rocks):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, dist)
            # Full score if min distance is > 8, penalty if < 3
            spacing_score = 15 * min(1, max(0, (min_dist - 3) / 5))

        rock_score = quantity_score + composition_score + spacing_score
        
        total_score = sand_score + rock_score
        return min(100, max(0, total_score))

    def _update_particles(self):
        """Updates position and lifetime of particles."""
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1 # Decrement lifetime

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all game elements."""
        # Render walls
        wall_points = [
            (0, 0), (self.GRID_W, 0), (self.GRID_W, self.GRID_H), (0, self.GRID_H)
        ]
        screen_wall_points = [self._iso_to_screen(p[0], p[1]) for p in wall_points]
        
        # Render side faces of the garden box
        p00, pW0, pWH, p0H = screen_wall_points
        depth = 80
        pygame.draw.polygon(self.screen, self.COLOR_WALL_SIDE, [p0H, (p0H[0], p0H[1]+depth), (pWH[0], pWH[1]+depth), pWH])
        pygame.draw.polygon(self.screen, self.COLOR_WALL_SIDE, [pWH, (pWH[0], pWH[1]+depth), (pW0[0], pW0[1]+depth), pW0])

        # Render sand and rocks in correct isometric order
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                # Render sand tile
                p = self._iso_to_screen(x, y)
                tile_points = [
                    p,
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x, y + 1)
                ]
                
                height_val = self.sand_grid[x, y]
                color_val = int(np.clip(180 - height_val * 40, 140, 220))
                sand_color = (color_val, int(color_val * 0.9), int(color_val * 0.75))
                pygame.gfxdraw.filled_polygon(self.screen, tile_points, sand_color)

                # Render rock if one is here
                for rock in self.rocks:
                    if rock['pos'] == (x, y):
                        self._render_rock(rock)
        
        # Render cursor on top
        self._render_cursor()

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(p[2] * 10)))
            color = (230, 200, 170, alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p[0][0]), int(p[0][1])))


    def _render_rock(self, rock):
        """Renders a single rock with a shadow."""
        cx, cy = self._iso_to_screen(rock['pos'][0], rock['pos'][1])
        cy -= 4 # Adjust to sit on top of sand
        
        scale = self.tile_w_iso * 0.5
        
        # Shadow
        shadow_points = [(cx + p[0] * scale + 2, cy + p[1] * scale + 4) for p in rock['shape']]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, self.COLOR_ROCK_SHADOW)
        pygame.gfxdraw.aapolygon(self.screen, shadow_points, self.COLOR_ROCK_SHADOW)

        # Rock
        rock_points = [(cx + p[0] * scale, cy + p[1] * scale) for p in rock['shape']]
        pygame.gfxdraw.filled_polygon(self.screen, rock_points, self.COLOR_ROCK)
        pygame.gfxdraw.aapolygon(self.screen, rock_points, (90,90,95))

    def _render_cursor(self):
        """Renders the player's cursor."""
        x, y = self.cursor_pos
        p = self._iso_to_screen(x, y)
        tile_points = [
            p,
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1)
        ]
        # Pulsing glow effect
        glow = (math.sin(self.steps * 0.2) + 1) / 2 * 2 + 1
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, tile_points, int(glow))
        
    def _render_ui(self):
        """Renders the UI elements."""
        # Create a semi-transparent background for the UI bar
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # Score
        score_text = f"Aesthetic: {self.score:.1f}%"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Timer
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        timer_width = 150
        bar_width = int(timer_width * time_ratio)
        bar_color = (int(255 * (1 - time_ratio)), int(255 * time_ratio), 0)
        
        pygame.draw.rect(self.screen, (60, 60, 70), (self.WIDTH - timer_width - 10, 15, timer_width, 10))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (self.WIDTH - timer_width - 10, 15, bar_width, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rocks_placed": len(self.rocks),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Example of running a few steps
    obs, info = env.reset()
    print(f"Initial state: {info}")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Info={info}, Terminated={terminated}")
        if terminated or truncated:
            print("Episode finished.")
            break
    env.close()