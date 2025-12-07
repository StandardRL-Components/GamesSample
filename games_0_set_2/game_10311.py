import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:16:38.514061
# Source Brief: brief_00311.md
# Brief Index: 311
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Drone:
    """A helper class to store drone state."""
    def __init__(self, color, pos, name):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.name = name
        self.fuel = 1.0  # 0.0 to 1.0

    def reset(self, pos):
        self.pos = pygame.Vector2(pos)
        self.fuel = 1.0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a team of drones to deliver packages. Synchronize all drones at each delivery point to complete the drop."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the active drone. Press shift to switch to the next drone."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    DRONE_SPEED = 6
    DRONE_SIZE = 10
    FUEL_CONSUMPTION_RATE = 0.995 # 0.5% of current fuel per move
    FUEL_CRITICAL_THRESHOLD = 0.2
    
    DELIVERY_POINT_SIZE = 15
    SYNC_RADIUS = 30
    SYNC_DURATION_STEPS = 120 # 4 seconds at 30 FPS

    MAX_STEPS = 2000
    NUM_PACKAGES_TO_WIN = 5
    NUM_DRONES = 3
    
    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 35, 45)
    DRONE_COLORS = [(255, 70, 70), (70, 255, 70), (70, 120, 255)] # Red, Green, Blue
    COLOR_DELIVERY_POINT = (255, 200, 0)
    COLOR_DELIVERY_POINT_DONE = (100, 80, 0)
    COLOR_DELIVERY_POINT_ACTIVE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game State Variables ---
        self.drones = [Drone(self.DRONE_COLORS[i], (0,0), f"DRN-{i+1}") for i in range(self.NUM_DRONES)]
        self.delivery_points = []
        self.active_drone_index = 0
        self.target_delivery_point_index = 0
        self.packages_delivered = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_shift_state = 0
        
        # Sync state
        self.sync_timer = -1
        self.drones_in_sync_zone = set()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.active_drone_index = 0
        self.packages_delivered = 0
        self.target_delivery_point_index = 0
        self.last_shift_state = 0
        self.sync_timer = -1
        self.drones_in_sync_zone = set()

        # Reset drones
        start_positions = [(50, 50), (50, 100), (50, 150)]
        for i, pos in enumerate(start_positions):
            self.drones[i].reset(pos)

        # Generate new delivery points
        self.delivery_points = []
        for _ in range(self.NUM_PACKAGES_TO_WIN):
            while True:
                x = self.np_random.integers(self.WIDTH // 2, self.WIDTH - 50)
                y = self.np_random.integers(50, self.HEIGHT - 50)
                new_point = pygame.Vector2(x, y)
                # Ensure points are not too close to each other
                if all(p.distance_to(new_point) > 100 for p in self.delivery_points):
                    self.delivery_points.append(new_point)
                    break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # --- 1. Handle Input ---
        self._handle_input(action)

        # --- 2. Update Game State ---
        sync_success, sync_fail = self._update_sync_state()

        # --- 3. Calculate Rewards & Score ---
        if sync_success:
            # SFX: Success chime
            self.packages_delivered += 1
            self.target_delivery_point_index += 1
            reward += 10.0
        
        if sync_fail:
            # SFX: Failure buzz
            self.game_over = True
            reward = -100.0
        
        # Continuous reward for fuel conservation
        if all(d.fuel > self.FUEL_CRITICAL_THRESHOLD for d in self.drones):
            reward += 0.1
        
        self.score += reward

        # --- 4. Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Handle win/loss terminal rewards
            if self.packages_delivered >= self.NUM_PACKAGES_TO_WIN:
                reward += 100.0 # Win
                self.score += 100.0
            else:
                reward = -100.0 # Loss (timeout or fuel empty)
                self.score = -100.0
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, _, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cycle active drone on SHIFT press (not hold)
        if shift_held and not self.last_shift_state:
            # SFX: UI click
            self.active_drone_index = (self.active_drone_index + 1) % self.NUM_DRONES
        self.last_shift_state = shift_held

        # Move active drone
        active_drone = self.drones[self.active_drone_index]
        if movement != 0 and active_drone.fuel > 0:
            vel = pygame.Vector2(0, 0)
            if movement == 1: vel.y = -1 # Up
            elif movement == 2: vel.y = 1 # Down
            elif movement == 3: vel.x = -1 # Left
            elif movement == 4: vel.x = 1 # Right
            
            active_drone.pos += vel * self.DRONE_SPEED
            
            # Clamp position to screen bounds
            active_drone.pos.x = max(0, min(self.WIDTH, active_drone.pos.x))
            active_drone.pos.y = max(0, min(self.HEIGHT, active_drone.pos.y))
            
            # Consume fuel
            active_drone.fuel *= self.FUEL_CONSUMPTION_RATE

    def _update_sync_state(self):
        if self.target_delivery_point_index >= self.NUM_PACKAGES_TO_WIN:
            return False, False
            
        target_pos = self.delivery_points[self.target_delivery_point_index]
        
        drones_in_radius_now = {
            i for i, d in enumerate(self.drones) 
            if d.pos.distance_to(target_pos) < self.SYNC_RADIUS
        }

        # Start timer if it's not running and a drone enters the zone
        if self.sync_timer < 0 and drones_in_radius_now:
            self.sync_timer = self.SYNC_DURATION_STEPS
            self.drones_in_sync_zone = drones_in_radius_now
        
        # If timer is running
        if self.sync_timer >= 0:
            self.sync_timer -= 1
            self.drones_in_sync_zone.update(drones_in_radius_now)

            # Abort if all drones leave the zone
            if not drones_in_radius_now:
                self.sync_timer = -1
                self.drones_in_sync_zone = set()
                return False, False

            # Check for success
            if len(self.drones_in_sync_zone) == self.NUM_DRONES:
                self.sync_timer = -1
                self.drones_in_sync_zone = set()
                return True, False
            
            # Check for timeout
            if self.sync_timer == 0:
                return False, True
        
        return False, False

    def _check_termination(self):
        if self.game_over:
            return True
        if self.packages_delivered >= self.NUM_PACKAGES_TO_WIN:
            return True
        if any(d.fuel <= 0 for d in self.drones):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_delivery_points()
        self._render_drones()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_delivery_points(self):
        for i, pos in enumerate(self.delivery_points):
            int_pos = (int(pos.x), int(pos.y))
            if i < self.target_delivery_point_index:
                color = self.COLOR_DELIVERY_POINT_DONE
            elif i == self.target_delivery_point_index:
                color = self.COLOR_DELIVERY_POINT_ACTIVE
                # Pulsing effect for active target
                pulse = (math.sin(self.steps * 0.1) + 1) / 2
                radius = self.DELIVERY_POINT_SIZE + pulse * 5
                pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], int(radius), (*color, 50))
                pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], int(radius), (*color, 150))
            else:
                color = self.COLOR_DELIVERY_POINT
            
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], self.DELIVERY_POINT_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], self.DELIVERY_POINT_SIZE, color)
            
            # Render sync timer if active
            if i == self.target_delivery_point_index and self.sync_timer >= 0:
                timer_text = f"{self.sync_timer / 30:.1f}s"
                text_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
                text_rect = text_surf.get_rect(center=(int_pos[0], int_pos[1] - 30))
                self.screen.blit(text_surf, text_rect)

    def _render_drones(self):
        for i, drone in enumerate(self.drones):
            # --- Draw Active Drone Highlight ---
            if i == self.active_drone_index:
                glow_color = (*drone.color, 20)
                for r in range(4):
                    pygame.gfxdraw.filled_circle(self.screen, int(drone.pos.x), int(drone.pos.y), self.DRONE_SIZE + 8 - r*2, glow_color)
            
            # --- Draw Drone Triangle ---
            p1 = (drone.pos.x, drone.pos.y - self.DRONE_SIZE)
            p2 = (drone.pos.x - self.DRONE_SIZE / 1.5, drone.pos.y + self.DRONE_SIZE / 2)
            p3 = (drone.pos.x + self.DRONE_SIZE / 1.5, drone.pos.y + self.DRONE_SIZE / 2)
            points = [p1, p2, p3]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            
            pygame.gfxdraw.aapolygon(self.screen, int_points, drone.color)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, drone.color)
            
            # --- Draw Fuel Bar ---
            bar_width = 30
            bar_height = 5
            bar_pos_x = drone.pos.x - bar_width / 2
            bar_pos_y = drone.pos.y - self.DRONE_SIZE - 12
            
            # Fuel color gradient (Green -> Yellow -> Red)
            fuel_color = (
                min(255, 255 * 2 * (1 - drone.fuel)),
                min(255, 255 * 2 * drone.fuel),
                0
            )
            
            # Background bar
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_pos_x, bar_pos_y, bar_width, bar_height))
            # Fuel level
            if drone.fuel > 0:
                pygame.draw.rect(self.screen, fuel_color, (bar_pos_x, bar_pos_y, max(0, bar_width * drone.fuel), bar_height))
            # Border
            pygame.draw.rect(self.screen, (150, 150, 150), (bar_pos_x, bar_pos_y, bar_width, bar_height), 1)

    def _render_ui(self):
        # Packages delivered
        package_text = f"PACKAGES: {self.packages_delivered} / {self.NUM_PACKAGES_TO_WIN}"
        text_surf = self.font_small.render(package_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        # Active drone indicator
        active_drone_text = f"ACTIVE: {self.drones[self.active_drone_index].name}"
        active_surf = self.font_small.render(active_drone_text, True, self.drones[self.active_drone_index].color)
        active_rect = active_surf.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(active_surf, active_rect)

        # Game over text
        if self.game_over:
            if self.packages_delivered >= self.NUM_PACKAGES_TO_WIN:
                end_text = "MISSION COMPLETE"
                end_color = (100, 255, 100)
            else:
                end_text = "MISSION FAILED"
                end_color = (255, 100, 100)
            
            end_surf = self.font_large.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            # Add a background for readability
            bg_rect = end_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, (*self.COLOR_BG, 200), bg_rect, border_radius=10)
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "packages_delivered": self.packages_delivered,
            "active_drone": self.active_drone_index,
            "drone_fuels": [d.fuel for d in self.drones]
        }

    def close(self):
        pygame.quit()

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To do so, you need to remove the SDL_VIDEODRIVER dummy setting
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # The validation call is useful for dev, but not needed for the main loop
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        # Decide if you want to exit or continue
        # exit() 
    
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Synchronous Drone Delivery")
    
    running = True
    terminated = False
    truncated = False
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0
    
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key == pygame.K_SPACE:
                    space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        if not (terminated or truncated):
            # Get movement from held keys
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset shift after one frame to simulate a "press"
            # This logic is handled inside the environment now based on last_shift_state
            # No, the manual loop needs to manage this to simulate a press vs hold
            shift_held = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()