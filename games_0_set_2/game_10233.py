import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:02:30.527997
# Source Brief: brief_00233.md
# Brief Index: 233
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A retro-arcade style racing game Gymnasium environment.

    The goal is to complete 3 laps on a procedurally generated track as fast as
    possible. The player controls a car with acceleration, braking, and steering.
    They can use a limited nitro boost for extra speed and initiate drifts to
    corner more effectively. Crashing into the track boundaries ends the episode.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up/accelerate, 2=down/brake, 3=left, 4=right)
    - actions[1]: Nitro (0=released, 1=held)
    - actions[2]: Drift (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB array representing the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A retro-arcade style racing game where you complete laps on a procedurally generated track. "
        "Use nitro boosts and drift around corners to get the best time."
    )
    user_guide = (
        "Controls: Use ↑ to accelerate, ↓ to brake, and ←→ to steer. "
        "Hold space for a nitro boost and shift to drift around corners."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000 # Increased from 1000 to allow for 3 laps

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_TRACK = (80, 90, 100)
    COLOR_LINES = (200, 200, 210)
    COLOR_CAR = (255, 80, 80)
    COLOR_CAR_GLOW = (255, 80, 80, 60)
    COLOR_NITRO = (80, 150, 255)
    COLOR_NITRO_PARTICLE = (80, 150, 255)
    COLOR_SMOKE_PARTICLE = (180, 180, 180)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (30, 40, 50, 180)
    COLOR_UI_GAUGE = (50, 60, 70)
    COLOR_UI_SPEED = (200, 200, 80)
    COLOR_UI_NITRO = (80, 150, 255)

    # Car Physics
    MAX_SPEED = 8.0
    ACCELERATION = 0.2
    BRAKING = 0.4
    FRICTION = 0.98  # Natural deceleration
    STEER_ANGLE = 0.08
    MAX_NITRO_SPEED = 14.0
    NITRO_ACCELERATION = 0.5
    NITRO_CAPACITY = 100.0
    NITRO_CONSUMPTION = 1.5

    # Drifting
    DRIFT_GRIP_REDUCTION = 0.95
    DRIFT_STEER_BOOST = 1.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.car = None
        self.particles = None
        self.track_points = None
        self.checkpoints = None
        self.track_surface = None
        self.current_checkpoint = None
        self.laps_completed = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_track()

        start_pos = self.track_points[0]
        start_angle = math.atan2(
            self.track_points[1][1] - start_pos[1],
            self.track_points[1][0] - start_pos[0]
        )

        self.car = {
            "pos": pygame.Vector2(start_pos),
            "vel": pygame.Vector2(0, 0),
            "angle": start_angle,
            "speed": 0.0,
            "nitro": self.NITRO_CAPACITY,
            "is_drifting": False
        }

        self.particles = []
        self.current_checkpoint = 0
        self.laps_completed = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        # --- Update Car Physics ---
        self._update_car_physics(movement, space_held, shift_held)
        
        # --- Update Particles ---
        self._update_particles()

        # --- Check Game State ---
        is_off_track = self._check_collision()
        lap_completed = self._check_laps()

        # --- Calculate Reward ---
        # Reward for forward speed
        reward += self.car["speed"] * 0.01
        
        # Penalty for using nitro
        if space_held and self.car['nitro'] > 0:
            reward -= 0.02 # encourage strategic use
        
        # Reward for drifting
        if self.car['is_drifting']:
            reward += 0.01

        if is_off_track:
            self.game_over = True
            reward = -10.0 # Crash penalty
        
        if lap_completed:
            self.laps_completed += 1
            reward += 5.0 # Lap bonus
            # Refill some nitro
            self.car["nitro"] = min(self.NITRO_CAPACITY, self.car["nitro"] + 30)

        # --- Termination Conditions ---
        terminated = self.game_over
        truncated = False
        if self.laps_completed >= 3:
            terminated = True
            reward += 100.0 # Victory bonus
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_car_physics(self, movement, space_held, shift_held):
        car = self.car
        
        # Drifting state
        car["is_drifting"] = shift_held and car["speed"] > 2.0

        # Steering
        steer_input = 0
        if movement == 3: # Left
            steer_input = -1
        elif movement == 4: # Right
            steer_input = 1
        
        steer_multiplier = self.DRIFT_STEER_BOOST if car["is_drifting"] else 1.0
        # Reduced steering at high speed for stability
        turn_speed_factor = max(0.2, 1.0 - (car["speed"] / (self.MAX_NITRO_SPEED * 1.5)))
        car["angle"] += steer_input * self.STEER_ANGLE * steer_multiplier * turn_speed_factor

        # Acceleration and Braking
        is_accelerating = False
        if movement == 1: # Up
            car["speed"] += self.ACCELERATION
            is_accelerating = True
        elif movement == 2: # Down
            car["speed"] -= self.BRAKING
        
        # Nitro
        if space_held and car["nitro"] > 0:
            car["speed"] += self.NITRO_ACCELERATION
            car["nitro"] -= self.NITRO_CONSUMPTION
            # sfx: nitro_boost
            if self.np_random.random() < 0.8: # Spawn particles frequently
                self._spawn_particles(2, "nitro")
        
        # Apply friction and clamp speed
        car["speed"] *= self.FRICTION
        car["speed"] = max(0, min(car["speed"], self.MAX_NITRO_SPEED if space_held and car["nitro"] > 0 else self.MAX_SPEED))
        car["nitro"] = max(0, car["nitro"])

        # Update velocity vector
        forward_vec = pygame.Vector2(math.cos(car["angle"]), math.sin(car["angle"]))
        
        if car["is_drifting"]:
            # sfx: tire_squeal
            # In drift, velocity changes slower, creating a slide
            car["vel"] = car["vel"].lerp(forward_vec * car["speed"], self.DRIFT_GRIP_REDUCTION)
            if self.np_random.random() < 0.9:
                self._spawn_particles(1, "smoke")
        else:
            # Normal grip
            car["vel"] = forward_vec * car["speed"]

        # Update position
        car["pos"] += car["vel"]

    def _generate_track(self, num_points=12, min_radius=100, max_radius=160, irregularity=0.8):
        self.track_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.track_surface.fill(self.COLOR_BG)
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = self.np_random.uniform(min_radius, max_radius)
            x = center_x + radius * math.cos(angle) + self.np_random.uniform(-irregularity, irregularity) * radius
            y = center_y + radius * math.sin(angle) + self.np_random.uniform(-irregularity, irregularity) * radius
            points.append((x, y))
        
        self.track_points = points
        
        # Draw wide track path for collision
        pygame.draw.polygon(self.track_surface, self.COLOR_TRACK, self.track_points, 0)
        pygame.draw.polygon(self.track_surface, self.COLOR_LINES, self.track_points, 6)

        # Create checkpoints
        self.checkpoints = []
        for i in range(num_points):
            p1 = pygame.Vector2(points[i])
            p2 = pygame.Vector2(points[(i + 1) % num_points])
            mid_point = p1.lerp(p2, 0.5)
            direction = (p2 - p1).normalize()
            perp_vec = pygame.Vector2(-direction.y, direction.x) * 50 # 50 is half track width
            
            cp_start = mid_point - perp_vec
            cp_end = mid_point + perp_vec
            self.checkpoints.append((cp_start, cp_end))

    def _check_collision(self):
        car_pos_int = (int(self.car["pos"].x), int(self.car["pos"].y))
        
        # Check boundaries
        if not (0 <= car_pos_int[0] < self.WIDTH and 0 <= car_pos_int[1] < self.HEIGHT):
            # sfx: crash_sound
            return True

        # Check if car center is on track
        try:
            color = self.track_surface.get_at(car_pos_int)
            if color != self.COLOR_TRACK and color != self.COLOR_LINES:
                # sfx: crash_sound
                return True
        except IndexError:
            # sfx: crash_sound
            return True
            
        return False

    def _check_laps(self):
        car_pos = self.car["pos"]
        next_checkpoint_idx = (self.current_checkpoint + 1) % len(self.checkpoints)
        
        # Finish line is checkpoint 0
        is_finish_line = next_checkpoint_idx == 0
        
        p1, p2 = self.checkpoints[next_checkpoint_idx]
        
        # Simple line intersection check
        prev_pos = car_pos - self.car["vel"]
        
        def line_intersect(p1, p2, p3, p4):
            den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
            if den == 0: return False
            t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / den
            u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / den
            return 0 < t < 1 and 0 < u < 1

        if line_intersect(prev_pos, car_pos, p1, p2):
            # To complete a lap, must have passed all other checkpoints
            if is_finish_line and self.current_checkpoint == len(self.checkpoints) - 1:
                self.current_checkpoint = 0
                # sfx: lap_complete_chime
                return True
            elif not is_finish_line:
                self.current_checkpoint = next_checkpoint_idx
        
        return False

    def _spawn_particles(self, count, p_type):
        for _ in range(count):
            if p_type == "nitro":
                vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
                life = 15
                size = self.np_random.uniform(4, 8)
                color = self.COLOR_NITRO_PARTICLE
                pos_offset = pygame.Vector2(-12, 0).rotate(-math.degrees(self.car["angle"]))
            elif p_type == "smoke":
                side_offset = self.np_random.choice([-6, 6])
                pos_offset = pygame.Vector2(-8, side_offset).rotate(-math.degrees(self.car["angle"]))
                base_vel = self.car["vel"] * 0.1
                rand_vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)) * 0.5
                vel = base_vel + rand_vel
                life = 30
                size = self.np_random.uniform(3, 7)
                color = self.COLOR_SMOKE_PARTICLE

            self.particles.append({
                "pos": self.car["pos"] + pos_offset,
                "vel": vel,
                "life": life,
                "max_life": life,
                "size": size,
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

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
            "laps": self.laps_completed,
            "nitro": self.car["nitro"],
        }
    
    def _render_game(self):
        # --- Draw Track ---
        # The track path is drawn from the pre-rendered surface for performance
        self.screen.blit(self.track_surface, (0, 0))
        
        # --- Draw Particles ---
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = p["size"] * (p["life"] / p["max_life"])
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(size), color
            )

        # --- Draw Car ---
        car_size = (24, 12)
        car_angle_deg = -math.degrees(self.car["angle"])
        
        # Glow effect
        glow_surf = pygame.Surface((car_size[0]*2, car_size[0]*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_CAR_GLOW, (car_size[0], car_size[0]), car_size[0])
        rotated_glow = pygame.transform.rotate(glow_surf, car_angle_deg)
        self.screen.blit(rotated_glow, rotated_glow.get_rect(center=self.car["pos"]))

        # Car body
        car_surf = pygame.Surface(car_size, pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.COLOR_CAR, (0, 0, *car_size), border_radius=3)
        # Add a "windshield" for direction
        pygame.draw.rect(car_surf, (50, 50, 70), (car_size[0] - 8, 2, 6, car_size[1] - 4), border_radius=2)
        
        rotated_car = pygame.transform.rotate(car_surf, car_angle_deg)
        self.screen.blit(rotated_car, rotated_car.get_rect(center=self.car["pos"]))
    
    def _render_ui(self):
        # --- Speedometer ---
        center_x, center_y = self.WIDTH - 90, self.HEIGHT - 90
        radius = 60
        
        # Gauge background
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_UI_BG)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_UI_GAUGE)
        
        # Speed text
        speed_text = f"{int(self.car['speed'] * 10)}"
        text_surf = self.font_large.render(speed_text, True, self.COLOR_UI_SPEED)
        self.screen.blit(text_surf, text_surf.get_rect(center=(center_x, center_y)))
        
        # Speed needle
        speed_frac = self.car["speed"] / self.MAX_NITRO_SPEED
        angle = math.pi * 1.5 * speed_frac - math.pi * 1.25
        end_x = center_x + (radius - 10) * math.cos(angle)
        end_y = center_y + (radius - 10) * math.sin(angle)
        pygame.draw.line(self.screen, self.COLOR_UI_SPEED, (center_x, center_y), (end_x, end_y), 3)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 5, self.COLOR_UI_SPEED)

        # --- Nitro Bar ---
        nitro_bar_pos = (self.WIDTH - 220, self.HEIGHT - 25)
        nitro_bar_size = (200, 15)
        nitro_frac = self.car["nitro"] / self.NITRO_CAPACITY
        
        pygame.draw.rect(self.screen, self.COLOR_UI_GAUGE, (*nitro_bar_pos, *nitro_bar_size), border_radius=3)
        if nitro_frac > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_NITRO, (*nitro_bar_pos, nitro_bar_size[0] * nitro_frac, nitro_bar_size[1]), border_radius=3)
        
        nitro_text = self.font_small.render("NITRO", True, self.COLOR_UI_TEXT)
        self.screen.blit(nitro_text, (nitro_bar_pos[0] - 60, nitro_bar_pos[1] - 2))

        # --- Lap Counter ---
        lap_text = f"LAP: {min(3, self.laps_completed + 1)} / 3"
        lap_surf = self.font_large.render(lap_text, True, self.COLOR_UI_TEXT)
        lap_rect = lap_surf.get_rect(midtop=(self.WIDTH // 2, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, lap_rect.inflate(20, 10), border_radius=5)
        self.screen.blit(lap_surf, lap_rect)

        # --- Score ---
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block needs a display, so it won't run in a truly headless environment
    # but is useful for local testing and visualization.
    # To run, you might need to unset the SDL_VIDEODRIVER variable.
    # For example:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    # Mapping from Pygame keys to MultiDiscrete action
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for rendering
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()

    print("--- Controls ---")
    print("Arrows: Steer, Accelerate, Brake")
    print("Space: Nitro Boost")
    print("Shift: Drift")
    print("R: Reset")
    print("----------------")

    while not (terminated or truncated):
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            continue
        
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Laps: {info['laps']}")

    env.close()