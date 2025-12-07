import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set a dummy video driver to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Your goal is to reach the blue escape ship "
        "at the top without being detected by the red alien guards."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Sneak past patrolling alien guards in a neon-lit facility to reach your escape ship. "
        "Observe patrol patterns and move strategically to avoid detection."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.GRID_W, self.GRID_H = 22, 22
        self.TILE_W, self.TILE_H = 28, 14
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 60
        self.DETECTION_RADIUS = 2.5 # in grid units

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ALIEN = (255, 50, 100)
        self.COLOR_ALIEN_GLOW = (255, 50, 100, 50)
        self.COLOR_SHIP = (50, 150, 255)
        self.COLOR_SHIP_GLOW = (50, 150, 255, 60)
        self.COLOR_OBSTACLE = (50, 60, 100)
        self.COLOR_OBSTACLE_TOP = (70, 80, 120)
        self.COLOR_DETECTION = (255, 255, 0)
        self.COLOR_PATH = (40, 50, 80)
        self.COLOR_CHECKPOINT = (200, 200, 0)
        self.COLOR_TEXT = (220, 220, 240)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.aliens = []
        self.obstacles = []
        self.ship_pos = (0, 0)
        self.checkpoint_pos = (0, 0)
        self.checkpoint_reached = False
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # Setup RNG
        self.np_random = None

        # self.reset() is called by the wrapper, but we can call it to initialize
        # We need a seed for the first reset to initialize the np_random generator
        self.reset(seed=0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.checkpoint_reached = False
        self.particles = []

        self.player_pos = (self.GRID_W - 2, self.GRID_H // 2)
        self.ship_pos = (1, self.GRID_H // 2)
        self.checkpoint_pos = (self.GRID_W // 2, self.GRID_H // 2)
        
        # Define static obstacles
        self.obstacles = [
            (5, 5, 3, 3), (14, 14, 3, 3), (5, 14, 3, 3), (14, 5, 3, 3)
        ]
        
        # Define aliens and their patrol paths
        self.aliens = [
            {
                "path": [(3, 10), (3, 1), (10, 1), (10, 10)],
                "path_idx": 0, "pos": (3, 10)
            },
            {
                "path": [(12, 12), (12, 18), (18, 18), (18, 12)],
                "path_idx": 0, "pos": (12, 12)
            },
            {
                "path": [(self.GRID_W // 2, 2), (self.GRID_W // 2, self.GRID_H - 3)],
                "path_idx": 0, "pos": (self.GRID_W // 2, 2)
            }
        ]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        terminated = False
        
        old_player_pos = self.player_pos
        old_alien_dists = [math.dist(self.player_pos, a['pos']) for a in self.aliens]

        # 1. Update Player Position
        dx, dy = 0, 0
        if movement == 1: dx, dy = -1, 0  # Up (iso up-left)
        elif movement == 2: dx, dy = 1, 0 # Down (iso down-right)
        elif movement == 3: dx, dy = 0, 1  # Left (iso down-left)
        elif movement == 4: dx, dy = 0, -1 # Right (iso up-right)

        if movement != 0:
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check boundaries
            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                # Check obstacles
                is_obstacle = False
                for ox, oy, ow, oh in self.obstacles:
                    if ox <= new_pos[0] < ox + ow and oy <= new_pos[1] < oy + oh:
                        is_obstacle = True
                        break
                if not is_obstacle:
                    self.player_pos = new_pos
                    # Add movement particle effect
                    self._add_particles(old_player_pos, self.COLOR_PLAYER, 5)

        # 2. Update Aliens
        for alien in self.aliens:
            target_pos = alien['path'][alien['path_idx']]
            if alien['pos'] == target_pos:
                alien['path_idx'] = (alien['path_idx'] + 1) % len(alien['path'])
                target_pos = alien['path'][alien['path_idx']]
            
            ax, ay = alien['pos']
            tx, ty = target_pos
            
            # Move one step towards target
            if ax < tx: ax += 1
            elif ax > tx: ax -= 1
            if ay < ty: ay += 1
            elif ay > ty: ay -= 1
            alien['pos'] = (ax, ay)
        
        # 3. Calculate Rewards
        new_alien_dists = [math.dist(self.player_pos, a['pos']) for a in self.aliens]
        closer_to_alien = any(new < old for new, old in zip(new_alien_dists, old_alien_dists))
        
        if closer_to_alien:
            reward -= 0.2
        else:
            reward += 0.1

        # 4. Check Termination Conditions
        # Detection (Loss)
        for alien in self.aliens:
            if math.dist(self.player_pos, alien['pos']) <= self.DETECTION_RADIUS:
                reward = -100.0
                terminated = True
                self.game_over = True
                self.game_outcome = "DETECTED"
                break
        
        if not terminated:
            # Reached Ship (Win)
            if self.player_pos == self.ship_pos:
                reward = 100.0
                terminated = True
                self.game_over = True
                self.game_outcome = "ESCAPED"
            # Reached Checkpoint
            elif not self.checkpoint_reached and self.player_pos == self.checkpoint_pos:
                reward += 5.0
                self.checkpoint_reached = True

        self.steps += 1
        # Max Steps
        truncated = False
        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            self.game_over = True
            self.game_outcome = "TIME OUT"

        self.score += reward
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "checkpoint_reached": self.checkpoint_reached,
        }

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_grid()
        for alien in self.aliens:
            self._render_patrol_path(alien['path'])
        self._render_checkpoint()
        self._render_ship()
        for obs in self.obstacles:
            self._draw_iso_cube(self.screen, *obs)
        self._render_aliens()
        self._render_player()
        self._draw_particles()

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            outcome_text = self.font_large.render(self.game_outcome, True, self.COLOR_TEXT)
            text_rect = outcome_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(outcome_text, text_rect)

    def _render_grid(self):
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                p1 = self._grid_to_iso(c, r)
                p2 = self._grid_to_iso(c + 1, r)
                p3 = self._grid_to_iso(c, r + 1)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p3)

    def _render_patrol_path(self, path):
        if len(path) > 1:
            points_iso = [self._grid_to_iso(p[0], p[1]) for p in path]
            pygame.draw.aalines(self.screen, self.COLOR_PATH, True, points_iso)

    def _render_checkpoint(self):
        if not self.checkpoint_reached:
            pos = self._grid_to_iso(self.checkpoint_pos[0], self.checkpoint_pos[1])
            points = [
                (pos[0], pos[1] - self.TILE_H / 2),
                (pos[0] + self.TILE_W / 4, pos[1]),
                (pos[0], pos[1] + self.TILE_H / 2),
                (pos[0] - self.TILE_W / 4, pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CHECKPOINT)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CHECKPOINT)

    def _render_ship(self):
        gx, gy = self.ship_pos
        
        # Glow
        center = self._grid_to_iso(gx, gy)
        self._draw_glow_circle(self.screen, self.COLOR_SHIP_GLOW, center, 20, 10)

        # Hull
        hull_points = [
            self._grid_to_iso(gx - 0.5, gy),
            self._grid_to_iso(gx + 1.5, gy - 1),
            self._grid_to_iso(gx + 1.5, gy + 1),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, hull_points, self.COLOR_SHIP)
        pygame.gfxdraw.aapolygon(self.screen, hull_points, self.COLOR_SHIP)


    def _render_aliens(self):
        for alien in self.aliens:
            gx, gy = alien['pos']
            center = self._grid_to_iso(gx, gy)
            
            # Detection radius
            is_detecting = math.dist(self.player_pos, alien['pos']) <= self.DETECTION_RADIUS
            radius_color = self.COLOR_DETECTION if is_detecting else self.COLOR_ALIEN_GLOW
            
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 
                                         int(self.DETECTION_RADIUS * self.TILE_W / 2), 
                                         (radius_color[0], radius_color[1], radius_color[2], 20))

            # Alien body
            points = [
                (center[0], center[1] - self.TILE_H * 0.7),
                (center[0] + self.TILE_W * 0.4, center[1]),
                (center[0], center[1] + self.TILE_H * 0.7),
                (center[0] - self.TILE_W * 0.4, center[1]),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
            
            # Detection indicator
            if is_detecting:
                text = self.font_small.render("!", True, self.COLOR_DETECTION)
                self.screen.blit(text, (center[0] - text.get_width() / 2, center[1] - 30))

    def _render_player(self):
        gx, gy = self.player_pos
        center = self._grid_to_iso(gx, gy)
        
        # Glow
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER_GLOW, center, 15, 8)
        
        # Player body
        points = [
            (center[0], center[1] - self.TILE_H * 0.6),
            (center[0] + self.TILE_W * 0.35, center[1] + self.TILE_H * 0.3),
            (center[0] - self.TILE_W * 0.35, center[1] + self.TILE_H * 0.3),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    # --- Helper Methods ---
    def _grid_to_iso(self, gx, gy):
        iso_x = self.ORIGIN_X + (gx - gy) * (self.TILE_W / 2)
        iso_y = self.ORIGIN_Y + (gx + gy) * (self.TILE_H / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, gx, gy, gw, gh, height=1):
        top_color = self.COLOR_OBSTACLE_TOP
        side_color = self.COLOR_OBSTACLE

        # Top face
        top_points = [
            self._grid_to_iso(gx, gy),
            self._grid_to_iso(gx + gw, gy),
            self._grid_to_iso(gx + gw, gy + gh),
            self._grid_to_iso(gx, gy + gh),
        ]
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        
        # Right face
        right_points = [
            self._grid_to_iso(gx + gw, gy),
            self._grid_to_iso(gx + gw, gy + gh),
            (self._grid_to_iso(gx + gw, gy + gh)[0], self._grid_to_iso(gx + gw, gy + gh)[1] + height * self.TILE_H),
            (self._grid_to_iso(gx + gw, gy)[0], self._grid_to_iso(gx + gw, gy)[1] + height * self.TILE_H),
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, side_color)
        
        # Left face
        left_points = [
            self._grid_to_iso(gx, gy + gh),
            self._grid_to_iso(gx + gw, gy + gh),
            (self._grid_to_iso(gx + gw, gy + gh)[0], self._grid_to_iso(gx + gw, gy + gh)[1] + height * self.TILE_H),
            (self._grid_to_iso(gx, gy + gh)[0], self._grid_to_iso(gx, gy + gh)[1] + height * self.TILE_H),
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, side_color)

    def _draw_glow_circle(self, surface, color, center, radius, glow_width):
        for i in range(glow_width):
            alpha = color[3] * (1 - i / glow_width)
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), 
                                    radius + i, (color[0], color[1], color[2], int(alpha)))

    def _add_particles(self, grid_pos, color, count):
        pos = self._grid_to_iso(grid_pos[0], grid_pos[1])
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha))
            radius = 2 * (p['life'] / p['max_life'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(radius), color)

# Example of how to run the environment
if __name__ == "__main__":
    # The validation part that was in __init__ is implicitly tested here
    try:
        env = GameEnv()
        obs, info = env.reset(seed=42)
        test_action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(test_action)
        assert obs.shape == (env.HEIGHT, env.WIDTH, 3)
        assert obs.dtype == np.uint8
        print("✓ Implementation validated successfully")
    except Exception as e:
        print(f"✗ Implementation validation failed: {e}")
        # raise e # uncomment to see full traceback

    # To display the game, we need a real video driver and a display loop
    # This might not work in all environments (e.g. a server without a display)
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        pygame.display.init()
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Isometric Stealth")
    except pygame.error:
        print("\nCould not set up a display. Running headlessly.")
        display_screen = None

    env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop for human play
    if display_screen:
        while not terminated and not truncated:
            movement_action = 0 # No-op by default
            
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: movement_action = 1; action_taken = True
                    elif event.key == pygame.K_DOWN: movement_action = 2; action_taken = True
                    elif event.key == pygame.K_LEFT: movement_action = 3; action_taken = True
                    elif event.key == pygame.K_RIGHT: movement_action = 4; action_taken = True
                    elif event.key == pygame.K_ESCAPE: terminated = True
            
            # In a turn-based game, we only step if an action is taken
            if action_taken:
                action = [movement_action, 0, 0] # Space/Shift not used
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Term: {terminated}, Trunc: {truncated}")

            # Render the observation to the display
            obs = env._get_observation()
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Since it's turn-based, we wait for input. A small delay prevents 100% CPU usage.
            pygame.time.wait(30) 

        print("Game Over!")
        pygame.quit()