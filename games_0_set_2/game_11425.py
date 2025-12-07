import gymnasium as gym
import os
import pygame
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for a minimalist space game.
    The player controls a ship, collecting resources to manage speed and an energy field.
    The goal is to reach 150% speed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a minimalist spaceship to collect green resources that increase your speed. "
        "Avoid red resources and reach 150% speed to win."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to navigate your ship."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_SPEED_MULTIPLIER = 1.5

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_SHIP = (255, 255, 255)
    COLOR_GREEN_RESOURCE = (0, 255, 150)
    COLOR_RED_RESOURCE = (255, 50, 100)
    COLOR_ENERGY_FIELD = (100, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    # Ship properties
    BASE_SHIP_SPEED = 4.0
    SHIP_SIZE = 12
    SHIP_ROTATION_SPEED = 0.15

    # Resource properties
    RESOURCE_RADIUS = 8
    INITIAL_RESOURCE_COUNT = 10

    # Trail & Particles
    TRAIL_MAX_LENGTH = 40
    PARTICLE_LIFETIME = 30
    PARTICLE_MAX_SPEED = 3.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = 0.0
        self.target_ship_angle = 0.0
        self.ship_speed_multiplier = 1.0
        self.resources = []
        self.trail = deque(maxlen=self.TRAIL_MAX_LENGTH)
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Ship
        self.ship_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.ship_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.ship_angle = -math.pi / 2
        self.target_ship_angle = -math.pi / 2
        self.ship_speed_multiplier = 1.0

        # Resources
        self.resources = []
        for i in range(self.INITIAL_RESOURCE_COUNT):
            res_type = 1 if i < self.INITIAL_RESOURCE_COUNT / 2 else -1
            self._spawn_resource(res_type)

        # Effects
        self.trail.clear()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, _, _ = action  # space_held and shift_held are unused

        # --- Calculate Pre-move State for Reward ---
        dist_before, _ = self._get_closest_resource_dist()

        # --- Game Logic ---
        self._update_ship(movement)
        self._update_trail()
        self._update_resources_physics()
        self._update_particles()

        # --- Event-based Rewards & State Changes ---
        event_reward = self._handle_collisions()
        self.score += event_reward

        # --- Continuous Reward ---
        dist_after, _ = self._get_closest_resource_dist()
        # Reward for getting closer to the nearest resource
        distance_reward = (dist_before - dist_after) * 0.01
        self.score += distance_reward

        reward = event_reward + distance_reward

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.ship_speed_multiplier >= self.WIN_SPEED_MULTIPLIER:
            terminated = True
            reward += 100.0  # Goal-oriented reward
            self.score += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_energy_field()
        self._render_trail()
        self._render_particles()
        self._render_resources()
        self._render_ship()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "speed_multiplier": self.ship_speed_multiplier,
        }

    def close(self):
        pygame.quit()

    # --- Game Logic Helpers ---

    def _update_ship(self, movement):
        # Smoothly rotate ship towards target angle
        self.ship_angle = self._lerp_angle(self.ship_angle, self.target_ship_angle, self.SHIP_ROTATION_SPEED)

        # Update target angle and velocity based on input
        move_vec = np.array([0.0, 0.0], dtype=np.float32)
        if movement == 1:  # Up
            move_vec[1] = -1
            self.target_ship_angle = -math.pi / 2
        elif movement == 2:  # Down
            move_vec[1] = 1
            self.target_ship_angle = math.pi / 2
        elif movement == 3:  # Left
            move_vec[0] = -1
            self.target_ship_angle = math.pi
        elif movement == 4:  # Right
            move_vec[0] = 1
            self.target_ship_angle = 0

        # Apply movement
        self.ship_vel = move_vec * self.BASE_SHIP_SPEED * self.ship_speed_multiplier
        self.ship_pos += self.ship_vel

        # World wrap-around
        self.ship_pos[0] %= self.SCREEN_WIDTH
        self.ship_pos[1] %= self.SCREEN_HEIGHT

    def _update_trail(self):
        if np.linalg.norm(self.ship_vel) > 0.1:
            self.trail.append(self.ship_pos.copy())

    def _update_resources_physics(self):
        field_strength = (1.0 - self.ship_speed_multiplier) * 20.0  # Positive for attraction, negative for repulsion
        field_radius = 150 + abs(field_strength) * 20

        for res in self.resources:
            dist_vec, dist = self._get_toroidal_distance(self.ship_pos, res['pos'])
            
            if dist < field_radius and dist > 1:
                force_mag = field_strength / (dist + 10) # Avoid division by zero
                force_vec = (dist_vec / dist) * force_mag
                res['vel'] += force_vec

            # Apply drag/friction
            res['vel'] *= 0.95
            
            # Update position and wrap around
            res['pos'] += res['vel']
            res['pos'][0] %= self.SCREEN_WIDTH
            res['pos'][1] %= self.SCREEN_HEIGHT

    def _handle_collisions(self):
        reward = 0.0
        collected_indices = []
        for i, res in enumerate(self.resources):
            _, dist = self._get_toroidal_distance(self.ship_pos, res['pos'])
            if dist < self.SHIP_SIZE + self.RESOURCE_RADIUS:
                collected_indices.append(i)
                if res['type'] == 1:  # Green
                    self.ship_speed_multiplier += 0.05
                    reward += 1.0
                    self._create_particles(res['pos'], self.COLOR_GREEN_RESOURCE, 20)
                else:  # Red
                    self.ship_speed_multiplier -= 0.10
                    reward -= 2.0
                    self._create_particles(res['pos'], self.COLOR_RED_RESOURCE, 20)
                
                # Clamp speed to a reasonable range
                self.ship_speed_multiplier = max(0.1, self.ship_speed_multiplier)

        # Remove collected resources and spawn new ones
        if collected_indices:
            # Process in reverse to avoid index errors
            for i in sorted(collected_indices, reverse=True):
                res_type = self.resources[i]['type']
                del self.resources[i]
                self._spawn_resource(res_type)
        
        return reward

    def _spawn_resource(self, res_type):
        while True:
            pos = self.np_random.uniform(low=0, high=[self.SCREEN_WIDTH, self.SCREEN_HEIGHT], size=(2,)).astype(np.float32)
            
            # Ensure it doesn't spawn too close to the player
            _, dist_to_ship = self._get_toroidal_distance(self.ship_pos, pos)
            if dist_to_ship > 100:
                self.resources.append({'pos': pos, 'type': res_type, 'vel': np.array([0.0, 0.0], dtype=np.float32)})
                break

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, self.PARTICLE_MAX_SPEED)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'lifetime': self.PARTICLE_LIFETIME,
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.98  # friction
            p['lifetime'] -= 1

    # --- Rendering Helpers ---

    def _render_ship(self):
        # Ship body
        p1 = (self.SHIP_SIZE, 0)
        p2 = (-self.SHIP_SIZE / 2, -self.SHIP_SIZE / 2)
        p3 = (-self.SHIP_SIZE / 2, self.SHIP_SIZE / 2)
        points = [p1, p2, p3]
        
        # Glow effect
        glow_radius = int(self.SHIP_SIZE * 1.5)
        self._draw_glow_circle(self.screen, self.COLOR_SHIP, self.ship_pos, glow_radius, 0.3)
        
        # Rotated ship
        self._draw_rotated_polygon(self.screen, self.COLOR_SHIP, points, self.ship_angle, self.ship_pos)

    def _render_resources(self):
        for res in self.resources:
            color = self.COLOR_GREEN_RESOURCE if res['type'] == 1 else self.COLOR_RED_RESOURCE
            self._draw_glow_circle(self.screen, color, res['pos'], self.RESOURCE_RADIUS, 0.5)

    def _render_trail(self):
        if len(self.trail) > 1:
            for i, pos in enumerate(self.trail):
                alpha = int(255 * (i / self.TRAIL_MAX_LENGTH) * 0.3)
                radius = int(self.SHIP_SIZE * 0.3 * (i / self.TRAIL_MAX_LENGTH))
                if radius > 0:
                    color = self.COLOR_ENERGY_FIELD + (alpha,)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)

    def _render_energy_field(self):
        field_strength_factor = abs(1.0 - self.ship_speed_multiplier)
        max_radius = int(80 + field_strength_factor * 100)
        
        for i in range(4, 0, -1):
            radius = int(max_radius * (i / 4.0))
            alpha = int(30 * field_strength_factor * (1 - (i / 5.0)))
            if alpha > 0:
                color = self.COLOR_ENERGY_FIELD + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(self.ship_pos[0]), int(self.ship_pos[1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(self.ship_pos[0]), int(self.ship_pos[1]), radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / self.PARTICLE_LIFETIME))
            radius = int(3 * (p['lifetime'] / self.PARTICLE_LIFETIME))
            if alpha > 0 and radius > 0:
                color = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_ui(self):
        # Speed display
        speed_text = f"Speed: {self.ship_speed_multiplier * 100:.0f}%"
        text_surface = self.font.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Step display
        step_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        text_surface = self.font.render(step_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

    # --- Utility Helpers ---

    def _get_toroidal_distance(self, pos1, pos2):
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if dx > self.SCREEN_WIDTH / 2:
            dx = self.SCREEN_WIDTH - dx
        if dy > self.SCREEN_HEIGHT / 2:
            dy = self.SCREEN_HEIGHT - dy
            
        vec_x = pos2[0] - pos1[0]
        if abs(vec_x) > self.SCREEN_WIDTH / 2:
            vec_x = -np.sign(vec_x) * (self.SCREEN_WIDTH - abs(vec_x))
            
        vec_y = pos2[1] - pos1[1]
        if abs(vec_y) > self.SCREEN_HEIGHT / 2:
            vec_y = -np.sign(vec_y) * (self.SCREEN_HEIGHT - abs(vec_y))

        return np.array([vec_x, vec_y]), math.sqrt(dx**2 + dy**2)

    def _get_closest_resource_dist(self):
        if not self.resources:
            return float('inf'), None
        
        min_dist = float('inf')
        closest_res = None
        for res in self.resources:
            _, dist = self._get_toroidal_distance(self.ship_pos, res['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_res = res
        return min_dist, closest_res

    def _lerp_angle(self, start, end, t):
        diff = (end - start + math.pi) % (2 * math.pi) - math.pi
        return start + diff * t

    def _draw_rotated_polygon(self, surface, color, points, angle, pivot):
        rotated_points = []
        for x, y in points:
            x_rot = x * math.cos(angle) - y * math.sin(angle) + pivot[0]
            y_rot = x * math.sin(angle) + y * math.cos(angle) + pivot[1]
            rotated_points.append((int(x_rot), int(y_rot)))
        pygame.draw.polygon(surface, color, rotated_points)
        pygame.gfxdraw.aapolygon(surface, rotated_points, color)

    def _draw_glow_circle(self, surface, color, center, radius, glow_factor):
        center_int = (int(center[0]), int(center[1]))
        glow_radius = int(radius * (1 + glow_factor))
        
        # Draw soft outer glow
        if glow_radius > radius:
            for i in range(glow_radius - radius):
                alpha = int(50 * (1 - i / (glow_radius - radius)))
                glow_color = color + (alpha,)
                pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius + i, glow_color)

        # Draw main circle
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)


if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Create a display window
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for manual play
    pygame.display.init()
    pygame.display.set_caption("Energy Field Navigator")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    while not done:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # space/shift not used

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_q]:
                done = True
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    print(f"\nEpisode Finished.")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")

    env.close()