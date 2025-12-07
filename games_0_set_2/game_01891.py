
# Generated: 2025-08-28T03:02:00.331608
# Source Brief: brief_01891.md
# Brief Index: 1891

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Shift to drift. Press Space to use boost."
    )

    game_description = (
        "A fast-paced isometric arcade racer. Drift through corners to gain an edge, "
        "collect crystals to fuel your boost, and race against the clock to complete 3 laps."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (25, 20, 40)
    COLOR_TRACK = (55, 50, 75)
    COLOR_TRACK_BORDER = (90, 85, 110)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 150, 150)
    COLOR_CRYSTAL = (100, 200, 255)
    COLOR_CRYSTAL_GLOW = (180, 230, 255)
    COLOR_BOOST_PARTICLE = (255, 180, 50)
    COLOR_DRIFT_PARTICLE = (150, 150, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_GREEN = (100, 255, 100)
    COLOR_TIMER_YELLOW = (255, 255, 100)
    COLOR_TIMER_RED = (255, 100, 100)

    # Physics
    ACCELERATION = 0.2
    BRAKING = 0.3
    FRICTION = 0.96
    TURN_SPEED = 0.06
    MAX_SPEED = 5.0
    DRIFT_TURN_MOD = 1.5
    DRIFT_FRICTION = 0.98
    BOOST_FORCE = 0.6
    BOOST_CONSUMPTION = 2.0
    MAX_BOOST = 100.0

    # Game
    TOTAL_LAPS = 3
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)

        self._define_track()
        
        # These will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.is_drifting = None
        self.is_boosting = None
        self.particles = None
        self.crystals = None
        self.boost_meter = None
        self.lap = None
        self.checkpoints_passed = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.camera_pos = None
        
        self.reset()
        self.validate_implementation()

    def _define_track(self):
        # World coordinates for the track path
        self.track_points = [
            (100, 100), (400, 100), (550, 200), (550, 400),
            (400, 550), (200, 550), (100, 450), (100, 250),
            (200, 150), (100, 100)
        ]
        self.track_width = 80

        # Define checkpoints as lines to cross
        self.checkpoints = []
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            self.checkpoints.append((p1, p2))
        
        self.start_pos = np.array([150.0, 120.0])
        self.start_angle = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = self.start_pos.copy()
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = self.start_angle
        self.is_drifting = False
        self.is_boosting = False
        self.particles = []
        self.boost_meter = 0
        self.lap = 0
        self.checkpoints_passed = set()
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_pos = self.player_pos.copy()

        self._spawn_crystals()
        
        return self._get_observation(), self._get_info()

    def _spawn_crystals(self):
        self.crystals = []
        # Spawn crystals along the track
        crystal_offsets = [-25, 25]
        for i in range(len(self.track_points) - 2):
            p1 = np.array(self.track_points[i])
            p2 = np.array(self.track_points[i+1])
            
            direction = (p2 - p1)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
            normal = np.array([-direction[1], direction[0]])
            
            mid_point = p1 + (p2 - p1) * 0.5
            offset_val = crystal_offsets[i % len(crystal_offsets)]
            crystal_pos = mid_point + normal * (self.track_width / 2 + offset_val)
            self.crystals.append(crystal_pos)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.02  # Time penalty

        self._handle_input_and_physics(movement, space_held, shift_held)
        self._update_particles()
        
        collected_reward = self._check_crystal_collisions()
        lap_reward = self._check_checkpoints()
        reward += collected_reward + lap_reward

        # Reward for forward movement
        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        forward_speed = np.dot(self.player_vel, forward_vec)
        if forward_speed > 0:
            reward += 0.01 * forward_speed

        self.steps += 1
        self.time_remaining -= 1

        terminated = self._check_termination()
        if terminated:
            if self.lap >= self.TOTAL_LAPS:
                reward += 100  # Victory bonus
            else:
                reward -= 100  # Timeout penalty
            self.game_over = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input_and_physics(self, movement, space_held, shift_held):
        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        speed = np.linalg.norm(self.player_vel)

        # Drifting
        self.is_drifting = shift_held and speed > 1.5 and (movement == 3 or movement == 4)
        
        # Turning
        turn_mod = self.DRIFT_TURN_MOD if self.is_drifting else 1.0
        if speed > 0.5:
            if movement == 3:  # Left
                self.player_angle -= self.TURN_SPEED * turn_mod
            if movement == 4:  # Right
                self.player_angle += self.TURN_SPEED * turn_mod

        # Acceleration / Braking
        if movement == 1:  # Up
            self.player_vel += forward_vec * self.ACCELERATION
        if movement == 2:  # Down
            self.player_vel -= forward_vec * self.BRAKING

        # Boosting
        self.is_boosting = space_held and self.boost_meter > 0
        if self.is_boosting:
            self.player_vel += forward_vec * self.BOOST_FORCE
            self.boost_meter = max(0, self.boost_meter - self.BOOST_CONSUMPTION)
            # sfx: boost sound

        # Friction
        friction_mod = self.DRIFT_FRICTION if self.is_drifting else self.FRICTION
        self.player_vel *= friction_mod
        if np.linalg.norm(self.player_vel) < 0.05:
            self.player_vel *= 0
        
        # Cap speed
        current_speed = np.linalg.norm(self.player_vel)
        if current_speed > self.MAX_SPEED:
            self.player_vel = self.player_vel / current_speed * self.MAX_SPEED

        # Update position
        self.player_pos += self.player_vel

        # Spawn particles
        if self.is_drifting:
            self._spawn_particle(self.player_pos, self.COLOR_DRIFT_PARTICLE, 20, count=2)
        if self.is_boosting:
            self._spawn_particle(self.player_pos - forward_vec * 10, self.COLOR_BOOST_PARTICLE, 15, count=3, angle_offset=math.pi)

    def _check_crystal_collisions(self):
        reward = 0
        for crystal_pos in self.crystals[:]:
            if np.linalg.norm(self.player_pos - crystal_pos) < 20:
                self.crystals.remove(crystal_pos)
                self.boost_meter = min(self.MAX_BOOST, self.boost_meter + 34)
                reward += 1
                self._spawn_particle(crystal_pos, self.COLOR_CRYSTAL, 25, count=10, speed=3)
                # sfx: crystal collect
        return reward

    def _check_checkpoints(self):
        reward = 0
        num_checkpoints = len(self.checkpoints)
        
        # Check collision with each checkpoint line segment
        player_segment = (self.player_pos - self.player_vel, self.player_pos)
        
        for i, (p1, p2) in enumerate(self.checkpoints):
            if i not in self.checkpoints_passed and self._line_segment_intersect(player_segment[0], player_segment[1], p1, p2):
                # Ensure checkpoints are passed in order
                required_prev = (i - 1 + num_checkpoints) % num_checkpoints
                if i == 0 or required_prev in self.checkpoints_passed:
                    self.checkpoints_passed.add(i)
                    reward += 5
                    
                    # Lap completion
                    if i == 0 and len(self.checkpoints_passed) == num_checkpoints:
                        self.lap += 1
                        self.checkpoints_passed = {0} # Reset for next lap, keeping start line
                        self._spawn_crystals()
                        # sfx: lap complete
        return reward
    
    def _line_segment_intersect(self, p1, p2, p3, p4):
        # Basic line segment intersection check
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True

        return False


    def _check_termination(self):
        return (
            self.lap >= self.TOTAL_LAPS
            or self.time_remaining <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "time_remaining": self.time_remaining / self.FPS,
            "boost": self.boost_meter
        }

    def _spawn_particle(self, pos, color, lifetime, count=1, speed=1.5, angle_offset=0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi) + angle_offset
            vel = np.array([math.cos(angle), math.sin(angle)]) * random.uniform(0.5, 1.0) * speed
            self.particles.append([pos.copy(), vel, color, lifetime])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]
            p[3] -= 1
        self.particles = [p for p in self.particles if p[3] > 0]

    def _world_to_screen(self, world_pos):
        # Isometric projection
        iso_x = (world_pos[0] - world_pos[1]) * 0.7
        iso_y = (world_pos[0] + world_pos[1]) * 0.4
        
        # Apply camera offset
        screen_x = iso_x - self.camera_pos[0] + self.SCREEN_WIDTH / 2
        screen_y = iso_y - self.camera_pos[1] + self.SCREEN_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _update_camera(self):
        # Camera smoothly follows the player, looking slightly ahead
        lookahead = self.player_vel * 15
        target_cam_world_pos = self.player_pos + lookahead
        
        # Convert world target to iso screen coords to calculate camera offset
        target_iso_x = (target_cam_world_pos[0] - target_cam_world_pos[1]) * 0.7
        target_iso_y = (target_cam_world_pos[0] + target_cam_world_pos[1]) * 0.4
        
        # Smoothly move camera
        self.camera_pos = self.camera_pos * 0.92 + np.array([target_iso_x, target_iso_y]) * 0.08

    def _get_observation(self):
        self._update_camera()
        self.screen.fill(self.COLOR_BG)
        
        self._render_track()
        self._render_particles()
        self._render_crystals()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self):
        # Render track base
        screen_track_points = [self._world_to_screen(p) for p in self.track_points[:-1]]
        if len(screen_track_points) > 2:
            pygame.draw.polygon(self.screen, self.COLOR_TRACK, screen_track_points)
        
        # Render track border
        pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, True, screen_track_points, 5)

        # Render start/finish line
        p1 = self._world_to_screen(self.checkpoints[0][0])
        p2 = self._world_to_screen(self.checkpoints[0][1])
        pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 5)


    def _render_crystals(self):
        for pos in self.crystals:
            screen_pos = self._world_to_screen(pos)
            size = 8
            points = [
                (screen_pos[0], screen_pos[1] - size),
                (screen_pos[0] + size, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + size),
                (screen_pos[0] - size, screen_pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)


    def _render_particles(self):
        for pos, vel, color, life in self.particles:
            screen_pos = self._world_to_screen(pos)
            radius = int(max(1, life / 5))
            alpha_color = (*color, max(0, min(255, int(life * 10))))
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (radius, radius), radius)
            self.screen.blit(temp_surf, (screen_pos[0] - radius, screen_pos[1] - radius))


    def _render_player(self):
        screen_pos = self._world_to_screen(self.player_pos)
        
        # Simple diamond shape for the cart
        angle = self.player_angle
        length = 15
        width = 8
        
        points = [
            (length, 0),
            (0, -width),
            (-length, 0),
            (0, width)
        ]
        
        # Rotate points
        rotated_points = []
        for p in points:
            x_rot = p[0] * math.cos(angle) - p[1] * math.sin(angle)
            y_rot = p[0] * math.sin(angle) + p[1] * math.cos(angle)
            
            # Since world is not iso but rendering is, we don't apply iso transform to rotation
            # but we do apply it to the final position. This is a common trick for iso games.
            rotated_points.append((screen_pos[0] + x_rot, screen_pos[1] + y_rot * 0.7))

        # Draw glow
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER_GLOW)
        # Draw cart
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time
        time_str = f"TIME: {max(0, self.time_remaining / self.FPS):.1f}"
        time_frac = max(0, self.time_remaining) / (self.TIME_LIMIT_SECONDS * self.FPS)
        time_color = self.COLOR_TIMER_RED if time_frac < 0.2 else (self.COLOR_TIMER_YELLOW if time_frac < 0.5 else self.COLOR_TIMER_GREEN)
        time_surf = self.font_large.render(time_str, True, time_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))

        # Laps
        lap_str = f"LAP: {min(self.lap + 1, self.TOTAL_LAPS)} / {self.TOTAL_LAPS}"
        lap_surf = self.font_small.render(lap_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_surf, (20, 10))

        # Score
        score_str = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 30))

        # Boost Meter
        boost_label_surf = self.font_small.render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_label_surf, (self.SCREEN_WIDTH - 180, 10))
        
        bar_x, bar_y, bar_w, bar_h = self.SCREEN_WIDTH - 180, 30, 160, 15
        pygame.draw.rect(self.screen, self.COLOR_TRACK_BORDER, (bar_x, bar_y, bar_w, bar_h))
        
        boost_w = bar_w * (self.boost_meter / self.MAX_BOOST)
        if boost_w > 0:
            pygame.draw.rect(self.screen, self.COLOR_BOOST_PARTICLE, (bar_x, bar_y, boost_w, bar_h))

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Crystal Drift Racer")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample() # Start with a sample action
    action.fill(0) # Default to no-op

    while not done:
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
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

        # Space (Boost)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift (Drift)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()