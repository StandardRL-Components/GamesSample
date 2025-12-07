
# Generated: 2025-08-28T06:17:19.532046
# Source Brief: brief_05852.md
# Brief Index: 5852

        
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


class GameEnv(gym.Env):
    """
    An isometric arcade-style game where the player collects toxic waste while avoiding obstacles.
    The environment is procedurally generated for each episode.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character. "
        "Space and Shift are not used."
    )

    game_description = (
        "Navigate a procedurally generated isometric environment, absorbing toxic waste "
        "while avoiding deadly obstacles to cleanse the world."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 10
        self.INITIAL_OBSTACLES = 2
        self.INITIAL_WASTE_BLOBS = 6

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_WASTE = (50, 255, 50)
        self.COLOR_WASTE_GLOW = (50, 200, 50)
        self.COLOR_OBSTACLE_SIDE = (80, 80, 90)
        self.COLOR_OBSTACLE_TOP = (100, 100, 110)
        self.COLOR_TEXT = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
            self.font_small = pygame.font.SysFont('Consolas', 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 24)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.waste_locations = []
        self.obstacle_locations = []
        self.particles = []
        
        # --- Pre-calculate grid offset for centering ---
        self.grid_offset_x = self.WIDTH / 2
        self.grid_offset_y = self.HEIGHT / 2 - (self.GRID_SIZE * self.TILE_HEIGHT / 2)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.waste_collected = 0
        self.particles = []

        self._procedurally_generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        # --- Reward Calculation Setup ---
        old_dist_to_waste = self._get_dist_to_nearest(self.player_pos, [w['pos'] for w in self.waste_locations])
        old_dist_to_obstacle = self._get_dist_to_nearest(self.player_pos, self.obstacle_locations)

        # --- Update Game State ---
        self._move_player(movement)
        reward = 0
        terminated = False

        # --- Movement Reward ---
        new_dist_to_waste = self._get_dist_to_nearest(self.player_pos, [w['pos'] for w in self.waste_locations])
        new_dist_to_obstacle = self._get_dist_to_nearest(self.player_pos, self.obstacle_locations)
        
        if old_dist_to_waste is not None and new_dist_to_waste < old_dist_to_waste:
            reward += 0.1
        if old_dist_to_obstacle is not None and new_dist_to_obstacle < old_dist_to_obstacle:
            reward -= 0.1

        # --- Collision Checks and Event Rewards ---
        # Obstacle collision
        if self.player_pos in self.obstacle_locations:
            reward = -100.0
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
            # sfx: # Player death explosion

        # Waste collection
        else:
            collected_this_step = False
            for i, waste in reversed(list(enumerate(self.waste_locations))):
                if self.player_pos == waste['pos']:
                    waste_size = waste['size']
                    reward += 1.0 * waste_size
                    self.waste_collected += waste_size
                    self.score = self.waste_collected
                    self._create_implosion(self.player_pos, self.COLOR_WASTE, 15 + 5 * waste_size)
                    self.waste_locations.pop(i)
                    collected_this_step = True
                    # sfx: # Powerup collect sound
                    break

            if collected_this_step:
                # Add new waste to maintain a constant number on map
                self._add_waste()
                # Difficulty scaling: add an obstacle every 2 waste units
                num_obstacles_should_have = self.INITIAL_OBSTACLES + (self.waste_collected // 2)
                if len(self.obstacle_locations) < num_obstacles_should_have:
                    self._add_obstacle()

        # --- Win/Loss Condition Checks ---
        if self.waste_collected >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _procedurally_generate_level(self):
        """Generates a new level layout for obstacles and waste."""
        self.player_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        
        available_coords = set((x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE))
        available_coords.remove(self.player_pos)

        # Place initial obstacles
        self.obstacle_locations = []
        for _ in range(self.INITIAL_OBSTACLES):
            if not available_coords: break
            pos = self.np_random.choice(list(available_coords))
            pos = tuple(pos)
            self.obstacle_locations.append(pos)
            available_coords.remove(pos)

        # Place initial waste
        self.waste_locations = []
        for _ in range(self.INITIAL_WASTE_BLOBS):
            if not available_coords: break
            pos = self.np_random.choice(list(available_coords))
            pos = tuple(pos)
            size = self.np_random.integers(1, 4) # size 1, 2, or 3
            self.waste_locations.append({'pos': pos, 'size': size})
            available_coords.remove(pos)

    def _add_item(self, item_list, is_waste=False):
        """Adds a single item (waste or obstacle) to a random valid location."""
        occupied = {self.player_pos} | set(self.obstacle_locations) | {w['pos'] for w in self.waste_locations}
        available_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE) if (x, y) not in occupied]
        if not available_coords:
            return

        pos = self.np_random.choice(list(available_coords))
        pos = tuple(pos)
        if is_waste:
            size = self.np_random.integers(1, 4)
            item_list.append({'pos': pos, 'size': size})
        else:
            item_list.append(pos)

    def _add_obstacle(self):
        self._add_item(self.obstacle_locations, is_waste=False)
    
    def _add_waste(self):
        self._add_item(self.waste_locations, is_waste=True)

    def _move_player(self, movement):
        """Updates player position based on action, with boundary checks."""
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1
        
        px = np.clip(px, 0, self.GRID_SIZE - 1)
        py = np.clip(py, 0, self.GRID_SIZE - 1)
        self.player_pos = (px, py)

    def _get_dist_to_nearest(self, pos, locations):
        """Calculates Euclidean distance to the nearest location in a list."""
        if not locations:
            return None
        distances = [math.hypot(pos[0] - loc[0], pos[1] - loc[1]) for loc in locations]
        return min(distances)

    def _get_iso_coords(self, grid_x, grid_y, z=0):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.grid_offset_x + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.grid_offset_y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        screen_y -= z * self.TILE_HEIGHT
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_grid()
        
        # Sort all renderable objects by their y-coordinate for correct isometric layering
        render_queue = []
        for pos in self.obstacle_locations:
            render_queue.append(('obstacle', pos))
        for waste in self.waste_locations:
            render_queue.append(('waste', waste))
        if not self.game_over or self.waste_collected >= self.WIN_SCORE:
            render_queue.append(('player', self.player_pos))
        
        render_queue.sort(key=lambda item: item[1][0] + item[1][1] if item[0] != 'waste' else item[1]['pos'][0] + item[1]['pos'][1])

        for item_type, item_data in render_queue:
            if item_type == 'obstacle':
                self._render_obstacle(item_data)
            elif item_type == 'waste':
                self._render_waste(item_data)
            elif item_type == 'player':
                self._render_player(item_data)
        
        self._render_particles()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Horizontal lines
            start = self._get_iso_coords(i, 0)
            end = self._get_iso_coords(i, self.GRID_SIZE)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
            # Vertical lines
            start = self._get_iso_coords(0, i)
            end = self._get_iso_coords(self.GRID_SIZE, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
    
    def _render_obstacle(self, pos):
        height = 1.5 # Visual height of the obstacle
        cx, cy = self._get_iso_coords(pos[0], pos[1])
        
        # Define points for a 3D-looking rock shape
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        points_top = [
            (cx, cy - int(h * height)),
            (cx + w // 2, cy - int(h * height) + h // 2),
            (cx, cy - int(h * height) + h),
            (cx - w // 2, cy - int(h * height) + h // 2)
        ]
        points_left = [
            (cx - w // 2, cy - int(h * height) + h // 2),
            (cx, cy - int(h * height) + h),
            (cx, cy + h // 2),
            (cx - w // 2, cy)
        ]
        points_right = [
            (cx + w // 2, cy - int(h * height) + h // 2),
            (cx, cy - int(h * height) + h),
            (cx, cy + h // 2),
            (cx + w // 2, cy)
        ]

        pygame.gfxdraw.filled_polygon(self.screen, points_top, self.COLOR_OBSTACLE_TOP)
        pygame.gfxdraw.filled_polygon(self.screen, points_left, self.COLOR_OBSTACLE_SIDE)
        pygame.gfxdraw.filled_polygon(self.screen, points_right, self.COLOR_OBSTACLE_SIDE)
        pygame.gfxdraw.aapolygon(self.screen, points_top, self.COLOR_OBSTACLE_SIDE)
        pygame.gfxdraw.aapolygon(self.screen, points_left, self.COLOR_OBSTACLE_SIDE)
        pygame.gfxdraw.aapolygon(self.screen, points_right, self.COLOR_OBSTACLE_SIDE)

    def _render_waste(self, waste):
        pos, size = waste['pos'], waste['size']
        cx, cy = self._get_iso_coords(pos[0], pos[1])
        
        # Pulsing glow effect
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        base_radius = 5 + size * 2
        glow_radius = base_radius + 2 + pulse * 3
        
        # Draw glow
        for i in range(int(glow_radius), int(base_radius), -1):
            alpha = int(100 * (1 - (i - base_radius) / (glow_radius - base_radius)))
            color = (*self.COLOR_WASTE_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, i, color)

        # Draw core
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(base_radius), self.COLOR_WASTE)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, int(base_radius), self.COLOR_WASTE)

    def _render_player(self, pos):
        cx, cy = self._get_iso_coords(pos[0], pos[1])
        radius = 8
        
        # Draw glow
        for i in range(radius + 5, radius, -1):
            alpha = int(80 * (1 - (i - radius) / 5))
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, i, (*self.COLOR_PLAYER_GLOW, alpha))

        # Draw core
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.waste_collected}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        if self.game_over:
            if self.waste_collected >= self.WIN_SCORE:
                end_text_str = "CLEANSED"
            else:
                end_text_str = "CONTAMINATED"
            
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            steps_text = self.font_small.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
            steps_rect = steps_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(steps_text, steps_rect)

    # --- Particle System ---
    def _create_explosion(self, grid_pos, color, count):
        cx, cy = self._get_iso_coords(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _create_implosion(self, grid_pos, color, count):
        cx, cy = self._get_iso_coords(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            start_dist = self.np_random.uniform(15, 30)
            start_pos = [cx + math.cos(angle) * start_dist, cy + math.sin(angle) * start_dist]
            speed = self.np_random.uniform(1, 3)
            vel = [-math.cos(angle) * speed, -math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': start_pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 3)
            if radius < 1: continue
            alpha = int(life_ratio * 255)
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for interactive play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Toxic Taker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("="*30)

    while not terminated:
        action = [0, 0, 0]  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    terminated = True
                
                if action[0] != 0: # If a move key was pressed
                    obs, reward, term, trunc, info = env.step(action)
                    print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {term}")
                    if term:
                        terminated = True

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    print("Game Over!")
    env.close()