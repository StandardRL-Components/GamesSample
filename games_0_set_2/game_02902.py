
# Generated: 2025-08-27T21:46:55.109619
# Source Brief: brief_02902.md
# Brief Index: 2902

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to apply thrust. Space for main engine boost. Shift to brake."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a hopping spaceship through a procedurally generated asteroid field to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ASTEROID = (255, 80, 80)
    COLOR_ASTEROID_GLOW = (255, 80, 80, 40)
    COLOR_FINISH_LINE = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_THRUSTER = (0, 191, 255)
    COLOR_BOOST = (255, 165, 0)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STAGES = 3
    STAGE_LENGTH = 2000 # Meters (world units)
    MAX_EPISODE_STEPS = 5000
    PLAYER_SIZE = 12
    
    # Physics
    MAIN_THRUST = 0.4
    MANEUVER_THRUST = 0.2
    BRAKE_DRAG = 0.1
    NATURAL_DRAG = 0.02
    MAX_VELOCITY = 8.0
    
    # Rewards
    REWARD_PROGRESS = 0.1
    REWARD_RISKY_PASS = 1.0
    REWARD_GETTING_CLOSER_TO_DANGER = -0.2
    REWARD_WIN = 100.0
    REWARD_FAIL = -10.0
    RISKY_DISTANCE = PLAYER_SIZE * 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 36, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.asteroids = None
        self.particles = None
        self.steps = None
        self.score = None
        self.stage = None
        self.max_distance_reached = None
        self.camera_x = None
        self.starfield_surface = None
        self.last_risky_reward_step = -100

        self.reset()
        
        # This check is for development and ensures the implementation matches the spec
        # self.validate_implementation() 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.max_distance_reached = 0.0
        
        self.player_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        
        self.camera_x = 0
        self.asteroids = []
        self._generate_asteroids_for_stage(self.stage)
        
        self.particles = []
        
        if self.starfield_surface is None:
            self._create_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Pre-computation for rewards ---
        old_max_distance = self.max_distance_reached
        min_dist_before = self._get_min_dist_to_asteroids()

        # --- Player Control & Physics ---
        self._handle_input(movement, space_held, shift_held)
        self._update_physics(shift_held)
        
        # --- Update Game State ---
        self.max_distance_reached = max(self.max_distance_reached, self.player_pos[0])
        self.camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 3

        self._update_asteroids()
        self._update_particles()
        
        # --- Check for Termination ---
        terminated = self._check_termination()
        
        # --- Calculate Reward ---
        # Progress reward
        progress = self.max_distance_reached - old_max_distance
        if progress > 0:
            reward += progress * self.REWARD_PROGRESS

        # Risky pass / getting closer reward
        min_dist_after = self._get_min_dist_to_asteroids()
        if not terminated:
            if min_dist_after < self.RISKY_DISTANCE and (self.steps - self.last_risky_reward_step > 30):
                reward += self.REWARD_RISKY_PASS
                self.last_risky_reward_step = self.steps
            elif min_dist_after < min_dist_before:
                reward += self.REWARD_GETTING_CLOSER_TO_DANGER

        if terminated and not self.game_over: # Collision
             reward = self.REWARD_FAIL
        elif terminated and self.game_over: # Win
             reward += self.REWARD_WIN
        
        self.score += reward
        self.steps += 1
        
        # --- Stage Progression ---
        if self.stage < self.MAX_STAGES and self.player_pos[0] > self.stage * self.STAGE_LENGTH:
            self.stage += 1
            self._generate_asteroids_for_stage(self.stage)
            # sound: stage_clear.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        thrust_vec = np.array([0.0, 0.0])
        
        if space_held:
            thrust_vec[0] += self.MAIN_THRUST
            self._create_particles(self.player_pos, self.COLOR_BOOST, 10, angle_offset=math.pi, speed_mult=1.5)
            # sound: boost.wav
        
        if movement == 1: # Up
            thrust_vec[1] -= self.MANEUVER_THRUST
            self._create_particles(self.player_pos, self.COLOR_THRUSTER, 3, angle_offset=math.pi/2)
        elif movement == 2: # Down
            thrust_vec[1] += self.MANEUVER_THRUST
            self._create_particles(self.player_pos, self.COLOR_THRUSTER, 3, angle_offset=-math.pi/2)
        elif movement == 3: # Left
            thrust_vec[0] -= self.MANEUVER_THRUST
            self._create_particles(self.player_pos, self.COLOR_THRUSTER, 3, angle_offset=0)
        elif movement == 4: # Right
            thrust_vec[0] += self.MANEUVER_THRUST
            self._create_particles(self.player_pos, self.COLOR_THRUSTER, 3, angle_offset=math.pi)
            
        self.player_vel += thrust_vec

    def _update_physics(self, shift_held):
        # Apply drag
        drag = self.BRAKE_DRAG if shift_held else self.NATURAL_DRAG
        self.player_vel *= (1.0 - drag)
        
        # Clamp velocity
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_VELOCITY:
            self.player_vel = self.player_vel / speed * self.MAX_VELOCITY
            
        # Update position
        self.player_pos += self.player_vel
        
        # Screen boundaries
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _check_termination(self):
        # Collision with asteroids
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE + asteroid["size"]:
                self.game_over = False # Lost
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, speed_mult=3.0)
                # sound: explosion.wav
                return True
        
        # Reached final goal
        if self.player_pos[0] >= self.MAX_STAGES * self.STAGE_LENGTH:
            self.game_over = True # Won
            self._create_particles(self.player_pos, self.COLOR_FINISH_LINE, 100, speed_mult=4.0)
            # sound: win.wav
            return True
            
        # Max steps
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = False # Timed out
            return True
            
        return False

    def _get_min_dist_to_asteroids(self):
        if not self.asteroids:
            return float('inf')
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"]) - asteroid["size"] - self.PLAYER_SIZE
            if dist < min_dist:
                min_dist = dist
        return max(0, min_dist)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(self.starfield_surface, (-self.camera_x * 0.1 % self.SCREEN_WIDTH, 0))
        self.screen.blit(self.starfield_surface, (-self.camera_x * 0.1 % self.SCREEN_WIDTH + self.SCREEN_WIDTH, 0))

        self._render_finish_lines()
        self._render_particles()
        self._render_asteroids()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "distance": self.max_distance_reached,
        }

    def _create_starfield(self):
        self.starfield_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for _ in range(200):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.integers(1, 3)
            brightness = self.np_random.integers(50, 150)
            color = (brightness, brightness, int(brightness * 1.2))
            if size == 1:
                self.starfield_surface.set_at((x, y), color)
            else:
                pygame.draw.circle(self.starfield_surface, color, (x, y), 1)

    def _generate_asteroids_for_stage(self, stage_num):
        density = 5 + int(5 * (stage_num - 1) * 0.2)
        start_x = (stage_num - 1) * self.STAGE_LENGTH
        end_x = stage_num * self.STAGE_LENGTH
        
        for _ in range(density):
            asteroid = {
                "pos": np.array([
                    start_x + self.np_random.uniform(self.SCREEN_WIDTH, end_x - start_x),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT)
                ]),
                "size": self.np_random.uniform(15, 40),
                "angle": self.np_random.uniform(0, 360),
                "rot_speed": self.np_random.uniform(-1.5, 1.5),
                "shape_points": self._create_asteroid_shape(self.np_random.integers(5, 9))
            }
            self.asteroids.append(asteroid)

    def _create_asteroid_shape(self, num_points):
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = self.np_random.uniform(0.7, 1.1)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["angle"] = (asteroid["angle"] + asteroid["rot_speed"]) % 360

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _create_particles(self, pos, color, count, angle_offset=0.0, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(-0.5, 0.5) + angle_offset
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy() - vel * 3, # Spawn behind the emitter
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    def _render_player(self):
        screen_pos = self.player_pos - np.array([self.camera_x, 0])
        x, y = int(screen_pos[0]), int(screen_pos[1])

        # Bobbing animation for "hopping" feel
        bob = math.sin(self.steps * 0.2) * 2
        y += int(bob)

        # Draw glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.PLAYER_SIZE + 5, self.COLOR_PLAYER_GLOW)
        
        # Draw ship body
        angle_rad = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else 0
        points = [
            (x + math.cos(angle_rad) * self.PLAYER_SIZE, y + math.sin(angle_rad) * self.PLAYER_SIZE),
            (x + math.cos(angle_rad + 2.5) * self.PLAYER_SIZE * 0.8, y + math.sin(angle_rad + 2.5) * self.PLAYER_SIZE * 0.8),
            (x + math.cos(angle_rad - 2.5) * self.PLAYER_SIZE * 0.8, y + math.sin(angle_rad - 2.5) * self.PLAYER_SIZE * 0.8)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            screen_pos = asteroid["pos"] - np.array([self.camera_x, 0])
            if -50 < screen_pos[0] < self.SCREEN_WIDTH + 50:
                x, y = int(screen_pos[0]), int(screen_pos[1])
                size = asteroid["size"]
                
                # Draw glow
                pygame.gfxdraw.filled_circle(self.screen, x, y, int(size + 5), self.COLOR_ASTEROID_GLOW)
                
                # Draw asteroid body
                rad_angle = math.radians(asteroid["angle"])
                points = []
                for p_x, p_y in asteroid["shape_points"]:
                    rotated_x = p_x * math.cos(rad_angle) - p_y * math.sin(rad_angle)
                    rotated_y = p_x * math.sin(rad_angle) + p_y * math.cos(rad_angle)
                    points.append((x + rotated_x * size, y + rotated_y * size))

                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p["pos"] - np.array([self.camera_x, 0])
            x, y = int(screen_pos[0]), int(screen_pos[1])
            size = int(p["lifespan"] / 6)
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], (x,y), size)

    def _render_finish_lines(self):
        for i in range(1, self.MAX_STAGES + 1):
            finish_x = i * self.STAGE_LENGTH - self.camera_x
            if 0 < finish_x < self.SCREEN_WIDTH:
                for y in range(0, self.SCREEN_HEIGHT, 20):
                    pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (finish_x, y, 5, 10))

    def _render_ui(self):
        dist_text = f"DISTANCE: {int(self.max_distance_reached):05d}m"
        stage_text = f"STAGE: {self.stage}/{self.MAX_STAGES}"
        
        dist_surf = self.font_ui.render(dist_text, True, self.COLOR_UI_TEXT)
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(dist_surf, (10, 10))
        self.screen.blit(stage_surf, (self.SCREEN_WIDTH - stage_surf.get_width() - 10, 10))

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This part will not run in the headless environment but is useful for local testing
    # To run locally, comment out the `os.environ` line at the top
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass # It was not set, which is fine
        
    env = GameEnv(render_mode="rgb_array")
    env.reset()

    # --- Manual Play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Astro Hopper")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            print("Episode finished. Resetting.")
            env.reset()
            terminated = False

        # Get keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the game
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()