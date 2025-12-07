import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:14:35.208056
# Source Brief: brief_00945.md
# Brief Index: 945
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Drone Collector".

    The player controls three drones (red, green, blue) with varying speeds.
    The goal is to collect 20 energy cells for each drone before time runs out.
    Switching between drones is done with the 'space' action.
    Collecting cells provides a temporary speed boost.
    Cell locations are periodically reshuffled to keep the gameplay dynamic.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control a team of three drones to collect energy cells. Switch between drones and gather all the cells for each drone before time runs out."
    user_guide = "Use the arrow keys (↑↓←→) to move the selected drone. Press 'space' to switch between drones."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60

    # Colors
    COLOR_BG = (16, 16, 24)  # Dark blue/black
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_BLUE = (50, 100, 255)
    COLOR_CELL = (255, 220, 50) # Bright yellow
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    TOTAL_TIME_SECONDS = 120
    CELLS_PER_DRONE = 20
    CELL_RESHUFFLE_THRESHOLD = 5
    DRONE_RADIUS = 10
    CELL_RADIUS = 5

    # Drone Specs (Base Speed, Color)
    DRONE_SPECS = [
        {'base_speed': 3.0, 'color': COLOR_RED},    # Red: Fast
        {'base_speed': 2.5, 'color': COLOR_GREEN},  # Green: Medium
        {'base_speed': 2.0, 'color': COLOR_BLUE},   # Blue: Slow
    ]
    NUM_DRONES = len(DRONE_SPECS)

    # Physics & Effects
    SPEED_BOOST_FACTOR = 1.8
    SPEED_BOOST_DURATION = 1.5 * TARGET_FPS # 1.5 seconds
    SPEED_DECAY_RATE = 0.98
    PARTICLE_LIFETIME = 20
    PARTICLE_SPAWN_RATE = 2 # Spawn a particle every N steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_drone = pygame.font.SysFont("Consolas", 14, bold=True)
        
        # --- Internal State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = 0
        self.drones = []
        self.cells = []
        self.selected_drone_idx = 0
        self.prev_space_held = False
        self.total_cells_collected = 0
        self.particles = []
        self.last_distances = [0.0] * self.NUM_DRONES
        
        self.reset()
        
        # self.validate_implementation() # Commented out for submission, but useful for dev

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.TOTAL_TIME_SECONDS * self.TARGET_FPS
        self.selected_drone_idx = 0
        self.prev_space_held = False
        self.total_cells_collected = 0
        self.particles = []
        
        # Initialize drones
        self.drones = []
        for i, spec in enumerate(self.DRONE_SPECS):
            self.drones.append({
                'pos': pygame.Vector2(
                    self.SCREEN_WIDTH / (self.NUM_DRONES + 1) * (i + 1),
                    self.SCREEN_HEIGHT / 2
                ),
                'vel': pygame.Vector2(0, 0),
                'color': spec['color'],
                'base_speed': spec['base_speed'],
                'current_speed': spec['base_speed'],
                'collected_count': 0,
                'boost_timer': 0,
                'last_collect_time': -1000,
                'goal_reached': False
            })

        # Initialize cells for each drone
        self.cells = [[] for _ in range(self.NUM_DRONES)]
        for i in range(self.NUM_DRONES):
            self._spawn_cells(i, self.CELLS_PER_DRONE)

        # Initialize distances for proximity reward
        self.last_distances = [self._find_closest_cell_dist(i) for i in range(self.NUM_DRONES)]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0.0

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Drone switching on space press (rising edge)
        if space_held and not self.prev_space_held:
            self.selected_drone_idx = (self.selected_drone_idx + 1) % self.NUM_DRONES
            # Sound effect placeholder: # sfx_switch_drone()
        self.prev_space_held = space_held

        # --- Update Game Logic ---
        self._update_particles()
        
        # Update selected drone movement
        selected_drone = self.drones[self.selected_drone_idx]
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            selected_drone['vel'] = move_vec
        
        selected_drone['pos'] += selected_drone['vel'] * selected_drone['current_speed']

        # Spawn trail particle
        if self.steps % self.PARTICLE_SPAWN_RATE == 0 and move_vec.length() > 0:
            self._add_particle(selected_drone['pos'], selected_drone['color'])
        
        # Clamp drone to screen bounds
        selected_drone['pos'].x = np.clip(selected_drone['pos'].x, self.DRONE_RADIUS, self.SCREEN_WIDTH - self.DRONE_RADIUS)
        selected_drone['pos'].y = np.clip(selected_drone['pos'].y, self.DRONE_RADIUS, self.SCREEN_HEIGHT - self.DRONE_RADIUS)

        # Update all drones' speed and boost timers
        for drone in self.drones:
            if drone['boost_timer'] > 0:
                drone['boost_timer'] -= 1
            else:
                # Decay speed back to base
                drone['current_speed'] = max(drone['base_speed'], drone['current_speed'] * self.SPEED_DECAY_RATE)

        # --- Collision Detection and Rewards ---
        drone_idx = self.selected_drone_idx
        drone = self.drones[drone_idx]
        
        # Proximity reward
        new_dist = self._find_closest_cell_dist(drone_idx)
        if new_dist < self.last_distances[drone_idx]:
            reward += 0.01 # Small reward for getting closer
        self.last_distances[drone_idx] = new_dist

        # Cell collection check
        cells_to_remove = []
        for i, cell_pos in enumerate(self.cells[drone_idx]):
            if drone['pos'].distance_to(cell_pos) < self.DRONE_RADIUS + self.CELL_RADIUS:
                cells_to_remove.append(i)
                
                # --- Collection Event ---
                reward += 1.0  # Base collection reward
                # Sound effect placeholder: # sfx_collect_cell()
                
                # Consecutive collection bonus
                if self.steps - drone['last_collect_time'] < self.TARGET_FPS * 0.75: # 0.75s window
                    reward += 0.5
                drone['last_collect_time'] = self.steps

                # Apply speed boost
                drone['current_speed'] = drone['base_speed'] * self.SPEED_BOOST_FACTOR
                drone['boost_timer'] = self.SPEED_BOOST_DURATION

                # Update counts
                if not drone['goal_reached']:
                    drone['collected_count'] += 1
                self.total_cells_collected += 1
                
                # Check for drone goal completion
                if drone['collected_count'] >= self.CELLS_PER_DRONE and not drone['goal_reached']:
                    drone['goal_reached'] = True
                    reward += 50.0
                    # Sound effect placeholder: # sfx_drone_complete()

                # Reshuffle cells
                if self.total_cells_collected > 0 and self.total_cells_collected % self.CELL_RESHUFFLE_THRESHOLD == 0:
                    self._reshuffle_all_cells()

        # Remove collected cells
        for i in sorted(cells_to_remove, reverse=True):
            del self.cells[drone_idx][i]
        
        # --- Check Termination Conditions ---
        terminated = False
        
        # Victory condition
        if all(d['goal_reached'] for d in self.drones):
            reward += 100.0 # Victory bonus
            terminated = True
            self.game_over = True
            # Sound effect placeholder: # sfx_victory()

        # Timeout condition
        if self.time_remaining <= 0:
            if not self.game_over: # Only apply penalty if not already won
                reward -= 50.0
            terminated = True
            self.game_over = True
            # Sound effect placeholder: # sfx_timeout()

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.TARGET_FPS,
            "drones_completed": [d['goal_reached'] for d in self.drones],
        }

    # --- Helper and Rendering Methods ---

    def _spawn_cells(self, drone_idx, num_cells):
        self.cells[drone_idx].clear()
        for _ in range(num_cells):
            self.cells[drone_idx].append(pygame.Vector2(
                self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            ))

    def _reshuffle_all_cells(self):
        for i in range(self.NUM_DRONES):
            num_remaining = len(self.cells[i])
            if num_remaining > 0:
                self._spawn_cells(i, num_remaining)
        # Sound effect placeholder: # sfx_reshuffle()

    def _find_closest_cell_dist(self, drone_idx):
        drone_pos = self.drones[drone_idx]['pos']
        if not self.cells[drone_idx]:
            return float('inf')
        
        min_dist_sq = float('inf')
        for cell_pos in self.cells[drone_idx]:
            dist_sq = drone_pos.distance_squared_to(cell_pos)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return math.sqrt(min_dist_sq)

    def _add_particle(self, pos, color):
        self.particles.append({
            'pos': pygame.Vector2(pos),
            'lifetime': self.PARTICLE_LIFETIME,
            'max_lifetime': self.PARTICLE_LIFETIME,
            'color': color,
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1

    def _draw_glowing_circle(self, surface, pos, radius, color):
        x, y = int(pos.x), int(pos.y)
        
        # Glow effect
        glow_radius = int(radius * 2.0)
        glow_alpha = 60
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core circle
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)

    def _render_game(self):
        # Render particles (background)
        for p in self.particles:
            life_ratio = p['lifetime'] / p['max_lifetime']
            radius = int(self.DRONE_RADIUS * 0.5 * life_ratio)
            alpha = int(150 * life_ratio)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, (*p['color'], alpha))

        # Render cells
        for i, drone_cells in enumerate(self.cells):
            drone_color = self.drones[i]['color']
            for cell_pos in drone_cells:
                self._draw_glowing_circle(self.screen, cell_pos, self.CELL_RADIUS, self.COLOR_CELL)
                # Draw a small indicator of which drone this cell belongs to
                pygame.gfxdraw.aacircle(self.screen, int(cell_pos.x), int(cell_pos.y), self.CELL_RADIUS + 3, drone_color)

        # Render drones
        for i, drone in enumerate(self.drones):
            pos = drone['pos']
            
            # Draw selection indicator
            if i == self.selected_drone_idx:
                sel_radius = int(self.DRONE_RADIUS * 1.8)
                sel_alpha = int(100 + 50 * math.sin(self.steps * 0.2))
                temp_surf = pygame.Surface((sel_radius * 2, sel_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*drone['color'], sel_alpha), (sel_radius, sel_radius), sel_radius, width=2)
                self.screen.blit(temp_surf, (int(pos.x) - sel_radius, int(pos.y) - sel_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Calculate drone rotation
            angle = drone['vel'].angle_to(pygame.Vector2(1, 0)) if drone['vel'].length() > 0 else 0
            
            # Define triangle points relative to origin
            points = [
                pygame.Vector2(self.DRONE_RADIUS, 0),
                pygame.Vector2(-self.DRONE_RADIUS * 0.5, -self.DRONE_RADIUS * 0.8),
                pygame.Vector2(-self.DRONE_RADIUS * 0.5, self.DRONE_RADIUS * 0.8),
            ]
            
            # Rotate and translate points
            rotated_points = [p.rotate(-angle) + pos for p in points]
            int_points = [(int(p.x), int(p.y)) for p in rotated_points]
            
            # Draw glow
            pygame.draw.polygon(self.screen, (*drone['color'], 80), int_points, width=6)
            
            # Draw main body
            pygame.gfxdraw.aapolygon(self.screen, int_points, drone['color'])
            pygame.gfxdraw.filled_polygon(self.screen, int_points, drone['color'])

    def _render_ui(self):
        # Render timer
        time_text = f"TIME: {max(0, self.time_remaining // self.TARGET_FPS):03}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Render drone collection counts
        for drone in self.drones:
            count_text = f"{drone['collected_count']}/{self.CELLS_PER_DRONE}"
            count_surf = self.font_drone.render(count_text, True, drone['color'])
            pos_x = drone['pos'].x - count_surf.get_width() / 2
            pos_y = drone['pos'].y - self.DRONE_RADIUS - 20
            self.screen.blit(count_surf, (int(pos_x), int(pos_y)))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Pygame Event Handling for Manual Control ---
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render the Observation to the Screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface.
        # env._get_observation already creates the surface, so we can just use that.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            # Wait for reset key
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0.0
                        wait_for_reset = False
                clock.tick(GameEnv.TARGET_FPS)

        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()