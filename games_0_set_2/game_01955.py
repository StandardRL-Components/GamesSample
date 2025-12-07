
# Generated: 2025-08-28T03:12:34.265490
# Source Brief: brief_01955.md
# Brief Index: 1955

        
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


# Helper class for particles
class Particle:
    def __init__(self, x, y, angle, speed, color, life):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = life
        self.initial_life = life

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface, camera_offset, world_to_screen):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color = self.color + (alpha,)
            size = int(3 * (self.life / self.initial_life))
            if size > 0:
                screen_x, screen_y = world_to_screen(self.x - camera_offset[0], self.y - camera_offset[1])
                pygame.gfxdraw.filled_circle(surface, int(screen_x), int(screen_y), size, color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to drift."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners to complete 3 laps on a procedurally generated track before you run out of fuel."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_TRACK = (100, 100, 110)
        self.COLOR_TRACK_BORDER = (150, 150, 160)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_CHECKPOINT = (255, 255, 0, 100)
        self.COLOR_START_FINISH = (255, 255, 255, 100)
        self.COLOR_FUEL = (0, 255, 0)
        self.COLOR_FUEL_LOW = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_DRIFT = (200, 200, 220)

        # Game constants
        self.MAX_STEPS = 5000
        self.LAPS_TO_WIN = 3
        self.FPS = 30

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None
        self.last_speed = 0
        self.is_drifting = False
        self.fuel = None
        self.laps_completed = None
        self.lap_start_time = None
        self.current_lap_time = None
        self.track_centerline = []
        self.track_polygons = []
        self.checkpoints = []
        self.next_checkpoint_index = 0
        self.is_off_track = False
        self.particles = []
        self.np_random = None

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _generate_track(self):
        self.track_centerline = []
        self.track_polygons = []
        self.checkpoints = []

        num_points = 150
        track_radius = 1200
        track_width = 150
        
        # Use seeded random for reproducibility
        c1 = self.np_random.uniform(0.3, 0.6)
        c2 = self.np_random.uniform(2, 4)
        c3 = self.np_random.uniform(0.3, 0.6)
        c4 = self.np_random.uniform(2, 4)

        # Generate centerline points
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * math.pi
            x = track_radius * (math.cos(angle) + c1 * math.cos(c2 * angle))
            y = track_radius * (math.sin(angle) + c3 * math.sin(c4 * angle))
            self.track_centerline.append(pygame.Vector2(x, y))

        # Generate track mesh polygons from centerline
        for i in range(num_points):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i + 1]

            direction = (p2 - p1).normalize()
            normal = pygame.Vector2(-direction.y, direction.x)

            v1 = p1 + normal * track_width / 2
            v2 = p2 + normal * track_width / 2
            v3 = p2 - normal * track_width / 2
            v4 = p1 - normal * track_width / 2
            self.track_polygons.append([v1, v2, v3, v4])
        
        # Create checkpoints
        num_checkpoints = 10
        for i in range(num_checkpoints):
            index = int(i * (num_points / num_checkpoints))
            self.checkpoints.append(self.track_centerline[index])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_track()

        self.player_pos = self.track_centerline[0].copy() + pygame.Vector2(10, 10)
        p1 = self.track_centerline[0]
        p2 = self.track_centerline[1]
        self.player_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        self.player_speed = 0
        self.last_speed = 0
        self.is_drifting = False

        self.fuel = 100.0
        self.laps_completed = 0
        self.lap_start_time = 0
        self.current_lap_time = 0
        self.next_checkpoint_index = 1
        self.is_off_track = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.clock.tick(self.FPS)
        
        movement, space_held, _ = action
        is_drifting_input = space_held == 1

        reward = 0
        self._update_physics(movement, is_drifting_input)
        
        # Fuel consumption
        fuel_consumed = 0.02  # base consumption
        if self.player_speed > 0:
            fuel_consumed += self.player_speed * 0.001
        if self.is_drifting:
            fuel_consumed += 0.05
        if self.is_off_track:
            fuel_consumed += 0.03 # Penalty for being off-track

        self.fuel = max(0, self.fuel - fuel_consumed)
        reward -= 0.01 * fuel_consumed

        # Check game state (laps, off-track)
        lap_completed = self._check_checkpoints()
        if lap_completed:
            self.laps_completed += 1
            self.score += 1
            reward += 1
            self.lap_start_time = self.steps
            if self.laps_completed < self.LAPS_TO_WIN:
                # Play lap complete sound
                pass

        self._check_off_track()

        # Update timers
        self.current_lap_time = (self.steps - self.lap_start_time) / self.FPS
        
        # Calculate rewards from brief
        if self.player_speed > 0:
            reward += 0.1
        if self.player_speed < self.last_speed:
            reward -= 0.1
        self.last_speed = self.player_speed

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.laps_completed >= self.LAPS_TO_WIN:
                reward += 100
                self.score += 100
            elif self.fuel <= 0:
                reward -= 100
                self.score -= 100

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self, movement, is_drifting_input):
        # Constants
        ACCEL = 0.2
        BRAKE = 0.4
        TURN_SPEED = 0.05
        FRICTION = 0.98
        DRIFT_FRICTION = 0.99
        DRIFT_TURN_MULT = 1.8
        MAX_SPEED = 8.0
        OFF_TRACK_FRICTION = 0.85

        # Drifting state
        self.is_drifting = is_drifting_input and self.player_speed > 2.0 and not self.is_off_track

        turn_multiplier = DRIFT_TURN_MULT if self.is_drifting else 1.0

        # Input handling
        if movement == 1:  # Accelerate
            self.player_speed = min(MAX_SPEED, self.player_speed + ACCEL)
        if movement == 2:  # Brake
            self.player_speed = max(0, self.player_speed - BRAKE)
        if movement == 3:  # Turn Left
            if self.player_speed > 0.1:
                self.player_angle -= TURN_SPEED * turn_multiplier
        if movement == 4:  # Turn Right
            if self.player_speed > 0.1:
                self.player_angle += TURN_SPEED * turn_multiplier

        # Apply friction
        if self.is_off_track:
            self.player_speed *= OFF_TRACK_FRICTION
        elif self.is_drifting:
            self.player_speed *= DRIFT_FRICTION
        else:
            self.player_speed *= FRICTION

        # Update position
        self.player_pos.x += math.cos(self.player_angle) * self.player_speed
        self.player_pos.y += math.sin(self.player_angle) * self.player_speed
        
        # Particle effects
        if self.is_drifting:
            # Play drift sound
            for _ in range(2):
                offset_angle = self.player_angle + math.pi/2 + self.np_random.uniform(-0.2, 0.2)
                p_speed = self.np_random.uniform(1, 3)
                p_life = self.np_random.integers(10, 20)
                # Emit from rear wheels
                rear_offset = pygame.Vector2(-15, 0).rotate(-math.degrees(self.player_angle))
                particle_pos = self.player_pos + rear_offset
                self.particles.append(Particle(particle_pos.x, particle_pos.y, offset_angle, p_speed, self.COLOR_DRIFT, p_life))

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_checkpoints(self):
        if self.next_checkpoint_index >= len(self.checkpoints):
            return False
            
        next_cp = self.checkpoints[self.next_checkpoint_index]
        dist_to_cp = self.player_pos.distance_to(next_cp)

        if dist_to_cp < 100: # Checkpoint radius
            self.next_checkpoint_index += 1
            if self.next_checkpoint_index >= len(self.checkpoints):
                # Completed a lap
                self.next_checkpoint_index = 0
                return True
        return False

    def _check_off_track(self):
        min_dist_sq = float('inf')
        for i in range(len(self.track_centerline) - 1):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i+1]
            l2 = p1.distance_squared_to(p2)
            if l2 == 0:
                dist_sq = self.player_pos.distance_squared_to(p1)
            else:
                t = max(0, min(1, (self.player_pos - p1).dot(p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist_sq = self.player_pos.distance_squared_to(projection)
            min_dist_sq = min(min_dist_sq, dist_sq)
        
        if math.sqrt(min_dist_sq) > 150 / 2: # track_width / 2
            self.is_off_track = True
        else:
            self.is_off_track = False

    def _check_termination(self):
        return (
            self.laps_completed >= self.LAPS_TO_WIN
            or self.fuel <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps_completed,
            "fuel": self.fuel,
        }

    def _world_to_screen(self, world_x, world_y):
        iso_x = (world_x - world_y) * 0.707
        iso_y = (world_x + world_y) * 0.4
        
        screen_x = self.WIDTH / 2 + iso_x
        screen_y = self.HEIGHT / 2 + iso_y - 100
        return screen_x, screen_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        camera_offset = self.player_pos
        
        # Render track
        for poly in self.track_polygons:
            screen_poly = [self._world_to_screen(p.x - camera_offset.x, p.y - camera_offset.y) for p in poly]
            pygame.gfxdraw.aapolygon(self.screen, screen_poly, self.COLOR_TRACK_BORDER)
            pygame.gfxdraw.filled_polygon(self.screen, screen_poly, self.COLOR_TRACK)

        # Render checkpoints
        for i, cp in enumerate(self.checkpoints):
            color = self.COLOR_START_FINISH if i == 0 else self.COLOR_CHECKPOINT
            if i == self.next_checkpoint_index or (i == 0 and self.next_checkpoint_index == 0 and self.laps_completed > 0):
                color = (255, 255, 255, 200) # Highlight next checkpoint
            
            p1_idx = i
            p2_idx = (i + 1) % (len(self.track_centerline) -1)
            p1 = self.track_centerline[p1_idx]
            p2 = self.track_centerline[p2_idx]
            direction = (p2 - p1).normalize() if (p2 - p1).length() > 0 else pygame.Vector2(1,0)
            normal = pygame.Vector2(-direction.y, direction.x)
            
            start_pos_w = cp - normal * 90
            end_pos_w = cp + normal * 90
            
            start_pos_s = self._world_to_screen(start_pos_w.x - camera_offset.x, start_pos_w.y - camera_offset.y)
            end_pos_s = self._world_to_screen(end_pos_w.x - camera_offset.x, end_pos_w.y - camera_offset.y)
            
            pygame.draw.line(self.screen, color, start_pos_s, end_pos_s, 5)

        # Render particles
        for p in self.particles:
            p.draw(self.screen, camera_offset, self._world_to_screen)
            
        # Render player
        player_screen_pos = self._world_to_screen(0, 0)
        
        car_points = [
            pygame.Vector2(15, 0), pygame.Vector2(-10, 8),
            pygame.Vector2(-10, -8)
        ]
        
        render_angle_rad = self.player_angle - math.pi / 4
        
        rotated_points = [p.rotate(math.degrees(-render_angle_rad)) for p in car_points]
        screen_points = [(p.x + player_screen_pos[0], p.y + player_screen_pos[1]) for p in rotated_points]
        
        glow_radius = 20 + 5 * abs(math.sin(self.steps * 0.2))
        pygame.gfxdraw.filled_circle(self.screen, int(player_screen_pos[0]), int(player_screen_pos[1]), int(glow_radius), self.COLOR_PLAYER_GLOW)
        
        pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Laps
        lap_text = f"LAP: {min(self.laps_completed + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}"
        text_surf = self.font_small.render(lap_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Fuel
        fuel_rect_bg = pygame.Rect(self.WIDTH - 110, 10, 100, 20)
        pygame.draw.rect(self.screen, (50, 50, 50), fuel_rect_bg)
        fuel_width = max(0, self.fuel / 100 * 100)
        fuel_color = self.COLOR_FUEL if self.fuel > 25 else self.COLOR_FUEL_LOW
        if self.fuel < 25 and self.steps % 10 < 5: # Blinking effect
             fuel_color = (255, 255, 0)
        fuel_rect_fg = pygame.Rect(self.WIDTH - 110, 10, fuel_width, 20)
        pygame.draw.rect(self.screen, fuel_color, fuel_rect_fg)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, fuel_rect_bg, 1)

        # Lap Time
        time_str = f"{self.current_lap_time:.2f}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(time_surf, time_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "RACE COMPLETE!" if self.laps_completed >= self.LAPS_TO_WIN else "OUT OF FUEL"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # To run and play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = [0, 0, 0] # no-op, no-drift, no-shift
        keys = pygame.key.get_pressed()

        # Prioritize turning over accelerating/braking if both are pressed
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Display the output on a real screen
        real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()