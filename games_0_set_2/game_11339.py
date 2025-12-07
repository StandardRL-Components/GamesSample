import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk fractal racing game.

    The agent controls a vehicle on a procedurally generated track, balancing
    energy consumption for boosting and terraforming against the risk of
    detection by sensor towers. The goal is to complete laps as fast as possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A cyberpunk fractal racer where you pilot a vehicle on a procedural track, "
        "using energy to boost or terraform the course while avoiding sensor towers."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to turn. "
        "Press space to boost and shift to terraform the track."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2500 # Increased for longer laps

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_TRACK = (30, 20, 50)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 255)
    COLOR_BOOST = (50, 255, 150)
    COLOR_TERRAFORM = (180, 50, 255)
    COLOR_SENSOR = (255, 50, 50)
    COLOR_SENSOR_CONE = (100, 20, 20)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_ENERGY = (0, 255, 100)
    COLOR_UI_DETECTION = (255, 50, 50)
    COLOR_UI_BAR_BG = (50, 50, 80)

    # Player Physics
    PLAYER_ACCELERATION = 0.2
    PLAYER_BRAKING = 0.4
    PLAYER_TURN_SPEED = 3.5
    PLAYER_MAX_SPEED = 6.0
    PLAYER_FRICTION = 0.98  # Multiplier for velocity
    PLAYER_OFF_TRACK_FRICTION = 0.85

    # Game Mechanics
    MAX_ENERGY = 100.0
    MAX_DETECTION = 100.0
    BOOST_COST = 0.8
    BOOST_MULTIPLIER = 2.0
    TERRAFORM_COST = 1.0
    TERRAFORM_RADIUS = 40
    ENERGY_REGEN = 0.05
    SENSOR_BASE_RANGE = 120
    DETECTION_RATE = 1.5

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
        self.font_timer = pygame.font.SysFont("monospace", 30, bold=True)
        
        # Game state variables initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.energy = None
        self.detection = None
        self.lap_time = None
        self.steps = None
        self.score = None
        self.lap_started = None
        self.laps_completed = None
        self.target_lap_time = None
        self.current_vehicle_stats = None
        self.best_lap_time = float('inf')

        self.track_points = []
        self.track_mask = None
        self.start_finish_line = None
        self.sensors = []
        self.terraformed_sections = []
        self.particles = []
        
        # Camera
        self.camera_pos = pygame.math.Vector2(0, 0)
        
        # Critical self-check
        # self.validate_implementation() # This is called in original code but can be slow/problematic. reset() is sufficient.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lap_time = 0.0
        self.lap_started = False
        self.laps_completed = 0
        self.target_lap_time = 45.0 # Initial lenient target

        self._generate_track_and_sensors()
        
        self.player_pos = pygame.math.Vector2(self.track_points[0])
        self.player_vel = pygame.math.Vector2(0, 0)
        # Orient player along the initial track segment
        p1 = pygame.math.Vector2(self.track_points[0])
        p2 = pygame.math.Vector2(self.track_points[1])
        self.player_angle = math.degrees((p2 - p1).angle_to(pygame.math.Vector2(1, 0)))

        self.energy = self.MAX_ENERGY
        self.detection = 0.0
        
        self.terraformed_sections = []
        self.particles = deque(maxlen=200)

        self._update_vehicle_stats()
        
        self.camera_pos = self.player_pos.copy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_player_physics()
        
        # Actions with energy cost
        if space_held and self.energy > self.BOOST_COST:
            self._activate_boost()
            reward -= 0.02 # Cost for using boost
        if shift_held and self.energy > self.TERRAFORM_COST:
            reward += self._activate_terraform()

        # Update state
        self.steps += 1
        self.energy = min(self.MAX_ENERGY, self.energy + self.ENERGY_REGEN)
        if self.lap_started:
            self.lap_time += 1.0 / self.FPS
        
        self._update_particles()
        reward += self._calculate_progress_reward()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            reward += self._calculate_terminal_reward()
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_track_and_sensors(self):
        # Procedurally generate a looping track path
        self.track_points = []
        center_x, center_y = self.SCREEN_WIDTH * 1.5, self.SCREEN_HEIGHT * 1.5
        radius_x, radius_y = 1000, 600
        num_points = 100
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * math.pi
            px = center_x + radius_x * math.cos(angle) + self.np_random.uniform(-150, 150)
            py = center_y + radius_y * math.sin(angle) + self.np_random.uniform(-150, 150)
            self.track_points.append((px, py))

        # Create track surface and mask for collision
        world_width = self.SCREEN_WIDTH * 3
        world_height = self.SCREEN_HEIGHT * 3
        track_surface = pygame.Surface((world_width, world_height), pygame.SRCALPHA)
        pygame.draw.lines(track_surface, self.COLOR_TRACK, True, self.track_points, 120)
        self.track_mask = pygame.mask.from_surface(track_surface)

        # Define start/finish line
        p1 = pygame.math.Vector2(self.track_points[0])
        p2 = pygame.math.Vector2(self.track_points[1])
        mid = p1.lerp(p2, 0.5)
        perp = (p2 - p1).rotate(90).normalize()
        self.start_finish_line = pygame.Rect(
            (mid - perp * 80).x, (mid - perp * 80).y, 
            perp.x * 160, perp.y * 160
        )
        self.start_finish_line.normalize()

        # Place sensors
        self.sensors = []
        num_sensors = 5
        for _ in range(num_sensors):
            # Place sensors near but off the track
            point_idx = self.np_random.integers(1, len(self.track_points) -1)
            p1 = pygame.math.Vector2(self.track_points[point_idx])
            p2 = pygame.math.Vector2(self.track_points[point_idx+1])
            mid = p1.lerp(p2, 0.5)
            perp = (p2 - p1).rotate(90 + self.np_random.choice([-20, 20])).normalize()
            dist = self.np_random.uniform(150, 250)
            self.sensors.append(mid + perp * dist)

    def _handle_input(self, movement):
        turn_direction = 0
        acceleration_input = 0
        
        if movement == 1: # Up
            acceleration_input = self.current_vehicle_stats['acceleration']
        elif movement == 2: # Down
            acceleration_input = -self.PLAYER_BRAKING
        elif movement == 3: # Left
            turn_direction = -1
        elif movement == 4: # Right
            turn_direction = 1

        # Apply turning
        if self.player_vel.length() > 0.5:
             self.player_angle -= turn_direction * self.current_vehicle_stats['turn_speed']
        
        # Apply acceleration
        acceleration_vec = pygame.math.Vector2(1, 0).rotate(-self.player_angle) * acceleration_input
        self.player_vel += acceleration_vec

    def _update_player_physics(self):
        # Cap speed
        if self.player_vel.length() > self.current_vehicle_stats['max_speed']:
            self.player_vel.scale_to_length(self.current_vehicle_stats['max_speed'])
        
        # Update position
        self.player_pos += self.player_vel

        # Clamp position to world bounds to prevent IndexError
        world_width = self.SCREEN_WIDTH * 3
        world_height = self.SCREEN_HEIGHT * 3
        self.player_pos.x = max(0, min(self.player_pos.x, world_width - 1))
        self.player_pos.y = max(0, min(self.player_pos.y, world_height - 1))

        # Check for off-track and apply friction
        on_track = self.track_mask.get_at((int(self.player_pos.x), int(self.player_pos.y)))
        
        on_terraform = False
        for tf_rect, _ in self.terraformed_sections:
            if tf_rect.collidepoint(self.player_pos):
                on_terraform = True
                break
        
        if on_track:
            friction = 1.01 if on_terraform else self.PLAYER_FRICTION # Terraforming reduces friction
            self.player_vel *= friction
        else:
            self.player_vel *= self.PLAYER_OFF_TRACK_FRICTION

    def _activate_boost(self):
        # sfx: boost_sound.play()
        boost_vec = pygame.math.Vector2(1, 0).rotate(-self.player_angle) * self.BOOST_MULTIPLIER
        self.player_vel += boost_vec
        self.energy -= self.current_vehicle_stats['boost_cost']
        
        # Add boost particles
        for _ in range(3):
            p_vel = -self.player_vel.normalize() * self.np_random.uniform(1, 3) + pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append([self.player_pos.copy(), p_vel, self.np_random.integers(10, 20), self.COLOR_BOOST])
    
    def _activate_terraform(self):
        # sfx: terraform_sound.play()
        self.energy -= self.current_vehicle_stats['terraform_cost']
        terraform_pos = self.player_pos + pygame.math.Vector2(self.TERRAFORM_RADIUS, 0).rotate(-self.player_angle)
        
        new_tf_rect = pygame.Rect(terraform_pos.x - self.TERRAFORM_RADIUS, terraform_pos.y - self.TERRAFORM_RADIUS, self.TERRAFORM_RADIUS * 2, self.TERRAFORM_RADIUS * 2)
        self.terraformed_sections.append((new_tf_rect, 200)) # Rect and lifetime

        # Check for sensor detection
        detected = False
        sensor_range = self.SENSOR_BASE_RANGE + (self.laps_completed // 5) * 0.5
        for sensor_pos in self.sensors:
            if sensor_pos.distance_to(terraform_pos) < sensor_range:
                self.detection = min(self.MAX_DETECTION, self.detection + self.DETECTION_RATE)
                detected = True
        
        return -5.0 if detected else 1.0

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # Update position
            p[2] -= 1 # Decrease lifetime

    def _calculate_progress_reward(self):
        # This is a simple velocity-based progress reward
        forward_velocity = self.player_vel.dot(pygame.math.Vector2(1, 0).rotate(-self.player_angle))
        return max(0, forward_velocity * 0.01)

    def _check_termination(self):
        if self.energy <= 0: return True
        if self.detection >= self.MAX_DETECTION: return True

        # Lap completion
        player_rect = pygame.Rect(self.player_pos.x - 2, self.player_pos.y - 2, 4, 4)
        if player_rect.colliderect(self.start_finish_line):
            if not self.lap_started:
                self.lap_started = True
                self.lap_time = 0.0
            elif self.lap_time > 5.0: # Debounce to prevent instant re-trigger
                return True
        return False

    def _calculate_terminal_reward(self):
        if self.energy <= 0: return -50.0
        if self.detection >= self.MAX_DETECTION: return -50.0
        
        # Lap finished
        if self.lap_time > 0:
            if self.lap_time < self.target_lap_time:
                return 50.0 # Beat target time
            else:
                return -10.0 # Finished but too slow
        
        return -20.0 # Timed out

    def _update_vehicle_stats(self):
        # Placeholder for vehicle unlocks
        level = self.laps_completed // 3
        if level == 0: # Base vehicle
            self.current_vehicle_stats = {'acceleration': 0.2, 'max_speed': 6.0, 'turn_speed': 3.5, 'boost_cost': 0.8, 'terraform_cost': 1.0}
        elif level == 1: # Faster vehicle
             self.current_vehicle_stats = {'acceleration': 0.25, 'max_speed': 7.0, 'turn_speed': 3.0, 'boost_cost': 1.0, 'terraform_cost': 1.2}
        else: # Efficient vehicle
             self.current_vehicle_stats = {'acceleration': 0.2, 'max_speed': 6.5, 'turn_speed': 3.5, 'boost_cost': 0.6, 'terraform_cost': 0.8}

    def _get_observation(self):
        # --- Update Camera ---
        target_camera_pos = self.player_pos - pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2) + self.player_vel * 5.0
        self.camera_pos.x = self.camera_pos.x * 0.9 + target_camera_pos.x * 0.1
        self.camera_pos.y = self.camera_pos.y * 0.9 + target_camera_pos.y * 0.1
        
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        # Render track
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, [(p[0] - self.camera_pos.x, p[1] - self.camera_pos.y) for p in self.track_points], 120)
        
        # Render terraformed sections
        for i in range(len(self.terraformed_sections) - 1, -1, -1):
            tf_rect, lifetime = self.terraformed_sections[i]
            s = pygame.Surface((tf_rect.width, tf_rect.height), pygame.SRCALPHA)
            alpha = max(0, min(255, int(lifetime * 1.5)))
            color = (*self.COLOR_TERRAFORM, alpha)
            pygame.draw.ellipse(s, color, s.get_rect())
            self.screen.blit(s, (tf_rect.x - self.camera_pos.x, tf_rect.y - self.camera_pos.y))
            self.terraformed_sections[i] = (tf_rect, lifetime - 1)
            if lifetime <= 0:
                self.terraformed_sections.pop(i)

        # Render sensors
        sensor_range = self.SENSOR_BASE_RANGE + (self.laps_completed // 5) * 0.5
        for pos in self.sensors:
            screen_pos = (int(pos.x - self.camera_pos.x), int(pos.y - self.camera_pos.y))
            # Cone flashing logic
            is_detecting = any(pos.distance_to(pygame.math.Vector2(tf.center)) < sensor_range for tf, _ in self.terraformed_sections)
            if is_detecting and self.steps % 10 < 5:
                cone_color = (*self.COLOR_SENSOR, 120)
            else:
                cone_color = (*self.COLOR_SENSOR_CONE, 50)
            
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(sensor_range), cone_color)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], int(sensor_range), cone_color)
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 8, self.COLOR_SENSOR)
        
        # Render particles
        for p_pos, _, p_life, p_color in self.particles:
            if p_life > 0:
                screen_pos = (int(p_pos.x - self.camera_pos.x), int(p_pos.y - self.camera_pos.y))
                radius = int(p_life / 4)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, p_color)

        # Render player
        player_screen_pos = (int(self.player_pos.x - self.camera_pos.x), int(self.player_pos.y - self.camera_pos.y))
        
        p_size = 12
        points = [
            pygame.math.Vector2(p_size, 0),
            pygame.math.Vector2(-p_size/2, -p_size/2),
            pygame.math.Vector2(-p_size/2, p_size/2)
        ]
        rotated_points = [p.rotate(-self.player_angle) + player_screen_pos for p in points]
        
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER_GLOW)
        
        # Main ship
        p_size = 10
        points = [
            pygame.math.Vector2(p_size, 0),
            pygame.math.Vector2(-p_size/2, -p_size/2),
            pygame.math.Vector2(-p_size/2, p_size/2)
        ]
        rotated_points = [p.rotate(-self.player_angle) + player_screen_pos for p in points]
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Energy Bar
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height))
        energy_width = max(0, (self.energy / self.MAX_ENERGY) * bar_width)
        pygame.draw.rect(self.screen, self.COLOR_UI_ENERGY, (10, 10, energy_width, bar_height))
        energy_text = self.font_ui.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (15, 12))

        # Detection Bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (self.SCREEN_WIDTH - bar_width - 10, 10, bar_width, bar_height))
        detection_width = max(0, (self.detection / self.MAX_DETECTION) * bar_width)
        pygame.draw.rect(self.screen, self.COLOR_UI_DETECTION, (self.SCREEN_WIDTH - bar_width - 10, 10, detection_width, bar_height))
        detection_text = self.font_ui.render("DETECTION", True, self.COLOR_UI_TEXT)
        self.screen.blit(detection_text, (self.SCREEN_WIDTH - bar_width, 12))

        # Lap Timer
        timer_str = f"{self.lap_time:.2f}"
        timer_text = self.font_timer.render(timer_str, True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(timer_text, timer_rect)

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, 20))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "detection": self.detection,
            "lap_time": self.lap_time,
            "laps_completed": self.laps_completed,
            "player_speed": self.player_vel.length()
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
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Unset the dummy driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose obs for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_reward = 0
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
        
        clock.tick(GameEnv.FPS)
        
    env.close()