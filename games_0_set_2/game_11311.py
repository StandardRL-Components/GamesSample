import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete

class GameEnv(gym.Env):
    """
    Dodecahedron Escape: A stealth-horror Gymnasium environment.

    The player, a shadow-shifting entity, must navigate a 2D projection of a
    dodecahedron face to reach an escape hatch. They are hunted by geometric
    alien guards who can only detect the player's 'human' form. The environment's
    gravity periodically flips, rotating the world and changing the controls.

    High-quality visuals, smooth animations, and tense gameplay are prioritized.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a rotating world as a shadow-shifting entity to reach the escape hatch. "
        "Avoid alien guards by switching to your shadow form, as they can only detect you as a human."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Press space to shift between human and shadow forms. "
        "The world rotates, which will also rotate your controls!"
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 14
    CELL_SIZE = 24
    GAME_AREA_WIDTH = GRID_SIZE * CELL_SIZE
    GAME_AREA_HEIGHT = GRID_SIZE * CELL_SIZE
    MAX_STEPS = 2000
    GRAVITY_FLIP_INTERVAL = 25
    INTERPOLATION_FACTOR = 0.25

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER_HUMAN = (0, 191, 255)
    COLOR_PLAYER_SHADOW = (138, 43, 226)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_HATCH = (0, 255, 127)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_ACCENT = (100, 100, 120)

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Center the game area
        self.grid_origin_x = (self.SCREEN_WIDTH - self.GAME_AREA_WIDTH) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.GAME_AREA_HEIGHT) // 2

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_visual_pos = None
        self.is_shadow_form = None
        self.aliens = None
        self.hatch_pos = None
        self.gravity_direction = None
        self.current_rotation = None
        self.target_rotation = None
        self.steps = None
        self.score = None
        self.last_space_state = None
        self.last_dist_to_hatch = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        
        # Player state
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 2]
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)
        self.is_shadow_form = False
        
        # Game objects
        self.hatch_pos = [self.GRID_SIZE // 2, 1]
        self.last_dist_to_hatch = self._manhattan_distance(self.player_pos, self.hatch_pos)
        
        # Alien state
        self.aliens = []
        self._spawn_aliens()

        # World state
        self.gravity_direction = 0  # 0:Down, 1:Right, 2:Up, 3:Left
        self.current_rotation = 0.0
        self.target_rotation = 0.0

        # Input state
        self.last_space_state = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, _ = action
        reward = 0.0
        terminated = False
        truncated = False

        # --- 1. Handle Player Form Shift ---
        form_shifted = space_pressed == 1 and not self.last_space_state
        if form_shifted:
            self.is_shadow_form = not self.is_shadow_form
            # SFX: Play form shift sound
            for alien in self.aliens:
                if self._manhattan_distance(self.player_pos, [int(alien['pos'][0]), int(alien['pos'][1])]) <= 2:
                    if self.is_shadow_form:
                        reward += 5.0 # Reward for tactical shift to shadow
                    break 
        self.last_space_state = (space_pressed == 1)

        # --- 2. Handle Player Movement ---
        dx, dy = self._map_action_to_delta(movement)
        if dx != 0 or dy != 0:
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos
        
        dist_to_hatch = self._manhattan_distance(self.player_pos, self.hatch_pos)
        if dist_to_hatch < self.last_dist_to_hatch:
            reward += 0.1
        elif dist_to_hatch > self.last_dist_to_hatch:
            reward -= 0.1
        self.last_dist_to_hatch = dist_to_hatch

        # --- 3. Update Aliens and Check for Detection ---
        self._update_aliens()
        for alien in self.aliens:
            alien_grid_pos = [int(alien['pos'][0]), int(alien['pos'][1])]
            dist_to_alien = self._manhattan_distance(self.player_pos, alien_grid_pos)
            
            if not self.is_shadow_form:
                if dist_to_alien <= 2:
                    # SFX: Play detection alert sound
                    terminated = True
                    reward = -100.0
                    break
                if dist_to_alien <= 3:
                    reward -= 1.0 # Penalty for being close while visible
        
        if terminated:
            self.score += reward
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # --- 4. Check Win/Loss/Termination Conditions ---
        if self.player_pos == self.hatch_pos:
            # SFX: Play win fanfare
            terminated = True
            reward = 100.0
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            
        # --- 5. Update World State (Gravity) ---
        if self.steps > 0 and self.steps % self.GRAVITY_FLIP_INTERVAL == 0:
            # SFX: Play gravity shift whoosh
            self.gravity_direction = (self.gravity_direction + 1) % 4
            self.target_rotation += 90.0

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Interpolate visual elements for smoothness
        if self.player_pos is not None:
            self._update_visuals()

        self.screen.fill(self.COLOR_BG)
        if self.player_pos is not None:
            self._render_game()
            self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        if self.player_pos is None:
            return {}
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_hatch": self._manhattan_distance(self.player_pos, self.hatch_pos),
            "player_form": "Shadow" if self.is_shadow_form else "Human"
        }

    # --- Helper and Rendering Methods ---

    def _spawn_aliens(self):
        # Spawn one alien for now
        path_type = self.np_random.choice(['horizontal', 'vertical'])
        if path_type == 'horizontal':
            y = self.np_random.integers(3, self.GRID_SIZE - 3)
            path = [[1, y], [self.GRID_SIZE - 2, y]]
        else:
            x = self.np_random.integers(3, self.GRID_SIZE - 3)
            path = [[x, 1], [x, self.GRID_SIZE - 2]]
        
        start_pos = path[0]
        self.aliens.append({
            'pos': list(start_pos),
            'visual_pos': self._grid_to_pixel(start_pos),
            'path': path,
            'path_index': 0,
            'direction': 1
        })

    def _update_aliens(self):
        base_speed = 0.05
        speed_increase = (self.steps // 100) * 0.02
        speed = min(base_speed + speed_increase, 0.2)

        for alien in self.aliens:
            target_node = alien['path'][alien['path_index']]
            
            # Move towards target node
            direction_vec = np.array(target_node) - np.array(alien['pos'])
            dist = np.linalg.norm(direction_vec)
            
            if dist < speed:
                # Reached node, select next one
                alien['pos'] = list(target_node)
                if alien['path_index'] == len(alien['path']) - 1:
                    alien['direction'] = -1
                elif alien['path_index'] == 0:
                    alien['direction'] = 1
                alien['path_index'] += alien['direction']
            else:
                move_vec = (direction_vec / dist) * speed
                alien['pos'][0] += move_vec[0]
                alien['pos'][1] += move_vec[1]

    def _update_visuals(self):
        # Interpolate player position
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        self.player_visual_pos[0] += (target_pixel_pos[0] - self.player_visual_pos[0]) * self.INTERPOLATION_FACTOR
        self.player_visual_pos[1] += (target_pixel_pos[1] - self.player_visual_pos[1]) * self.INTERPOLATION_FACTOR

        # Interpolate alien positions
        for alien in self.aliens:
            target_pixel_pos = self._grid_to_pixel(alien['pos'])
            alien['visual_pos'][0] += (target_pixel_pos[0] - alien['visual_pos'][0]) * self.INTERPOLATION_FACTOR
            alien['visual_pos'][1] += (target_pixel_pos[1] - alien['visual_pos'][1]) * self.INTERPOLATION_FACTOR
            
        # Interpolate rotation
        diff = (self.target_rotation - self.current_rotation + 180) % 360 - 180
        self.current_rotation += diff * self.INTERPOLATION_FACTOR
        self.current_rotation %= 360

    def _render_game(self):
        # Create a surface for the game world that will be rotated
        game_surface = pygame.Surface((self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT), pygame.SRCALPHA)

        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(game_surface, self.COLOR_GRID, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GAME_AREA_HEIGHT))
            pygame.draw.line(game_surface, self.COLOR_GRID, (0, i * self.CELL_SIZE), (self.GAME_AREA_WIDTH, i * self.CELL_SIZE))

        # Draw hatch
        hatch_px = self._grid_to_pixel(self.hatch_pos, on_surface=True)
        self._render_glow_circle(game_surface, self.COLOR_HATCH, hatch_px, self.CELL_SIZE * 0.4, 15)

        # Draw aliens
        for alien in self.aliens:
            alien_px = self._grid_to_pixel(alien['pos'], on_surface=True)
            self._render_glow_circle(game_surface, self.COLOR_ALIEN, alien_px, self.CELL_SIZE * 0.3, 10)
            
            # Draw detection radius if player is human
            if not self.is_shadow_form:
                radius = 2.5 * self.CELL_SIZE
                pulse = abs(math.sin(self.steps * 0.1))
                color = (*self.COLOR_ALIEN, int(30 + pulse * 30))
                pygame.gfxdraw.filled_circle(game_surface, int(alien_px[0]), int(alien_px[1]), int(radius), color)
                pygame.gfxdraw.aacircle(game_surface, int(alien_px[0]), int(alien_px[1]), int(radius), color)

        # Draw player
        player_px = self._grid_to_pixel(self.player_pos, on_surface=True)
        player_color = self.COLOR_PLAYER_SHADOW if self.is_shadow_form else self.COLOR_PLAYER_HUMAN
        self._render_glow_circle(game_surface, player_color, player_px, self.CELL_SIZE * 0.4, 20)

        # Rotate and blit the game surface
        rotated_surface = pygame.transform.rotate(game_surface, self.current_rotation)
        rect = rotated_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(rotated_surface, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_ACCENT)
        self.screen.blit(steps_text, (20, 40))

        # Form status
        form_str = "SHADOW" if self.is_shadow_form else "HUMAN"
        form_color = self.COLOR_PLAYER_SHADOW if self.is_shadow_form else self.COLOR_PLAYER_HUMAN
        form_text = self.font_large.render(f"FORM: {form_str}", True, form_color)
        form_rect = form_text.get_rect(right=self.SCREEN_WIDTH - 20, top=10)
        self.screen.blit(form_text, form_rect)

        # Distance
        dist = self._manhattan_distance(self.player_pos, self.hatch_pos)
        dist_text = self.font_small.render(f"HATCH DIST: {dist}", True, self.COLOR_UI_ACCENT)
        dist_rect = dist_text.get_rect(right=self.SCREEN_WIDTH - 20, top=40)
        self.screen.blit(dist_text, dist_rect)

        # Gravity Indicator
        self._render_gravity_arrow()

    def _render_gravity_arrow(self):
        arrow_size = 15
        center = (self.SCREEN_WIDTH / 2, 30)
        angle_rad = math.radians(self.current_rotation)
        
        points = [
            (0, -arrow_size),
            (arrow_size * 0.5, 0),
            (-arrow_size * 0.5, 0)
        ]
        
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            rotated_points.append((center[0] + rx, center[1] + ry))
            
        pygame.draw.polygon(self.screen, self.COLOR_UI_ACCENT, rotated_points)

    def _render_glow_circle(self, surface, color, center, radius, max_glow):
        center_int = (int(center[0]), int(center[1]))
        for i in range(max_glow, 0, -2):
            alpha = int(150 * (1 - (i / max_glow))**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius + i), glow_color)
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius + i), glow_color)
        
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)

    def _grid_to_pixel(self, grid_pos, on_surface=False):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        if on_surface:
            return [px, py]
        return [px + self.grid_origin_x, py + self.grid_origin_y]

    def _map_action_to_delta(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        deltas = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = deltas.get(movement, (0, 0))

        # Rotate delta based on gravity
        for _ in range(self.gravity_direction):
            dx, dy = dy, -dx
        return dx, dy

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Script ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dodecahedron Escape - Manual Control")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            obs, info = env.reset()
            terminated = False
            truncated = False
            action.fill(0)
            pygame.time.wait(2000)

        # --- Keyboard to MultiDiscrete mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1 # Up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2 # Down
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3 # Left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4 # Right
        
        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button (unused in this game)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()