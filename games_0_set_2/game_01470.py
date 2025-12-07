import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to accelerate, ←→ to turn, ↓ to brake. Hold shift to drift. Press space to use a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade racer. Navigate a procedurally generated track, avoid obstacles, and complete three laps before time runs out."
    )

    # Frames auto-advance for smooth gameplay
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_TRACK = (70, 70, 80)
    COLOR_TRACK_BORDER = (120, 120, 130)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 150, 150, 50)
    COLOR_OBSTACLE = (80, 150, 255)
    COLOR_OBSTACLE_GLOW = (150, 200, 255, 80)
    COLOR_BOOST = (255, 220, 50)
    COLOR_BOOST_GLOW = (255, 240, 150, 100)
    COLOR_FINISH_LINE_1 = (240, 240, 240)
    COLOR_FINISH_LINE_2 = (40, 40, 40)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_TRAIL = (200, 80, 80, 150)
    COLOR_DRIFT_SMOKE = (200, 200, 200, 20)
    COLOR_BOOST_TRAIL = (255, 255, 150, 200)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game parameters
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    LAPS_TO_WIN = 3
    
    # Player physics
    ACCELERATION = 0.25
    BRAKING = 0.4
    FRICTION = 0.97
    MAX_SPEED = 5.0
    TURN_SPEED = 3.5
    DRIFT_TURN_MOD = 1.8
    DRIFT_FRICTION_MOD = 1.01 # Slight speed loss penalty for tight turns
    PLAYER_RADIUS = 10
    
    # Boost mechanics
    BOOST_SPEED_MULT = 1.7
    BOOST_DURATION = 90 # 3 seconds at 30 FPS
    
    # Track generation
    TRACK_WAYPOINTS = 50
    TRACK_RADIUS_BASE = 400
    TRACK_RADIUS_NOISE = 150
    TRACK_WIDTH = 60
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.render_mode = render_mode
        self.game_objects_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        self.player_glow_surface = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        self._create_glow_surface(self.player_glow_surface, self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
        
        # This is to ensure a seed is available for the first reset.
        self.np_random = None
        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0
        
        # Player state
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_trail = []
        self.particles = []

        # Lap state
        self.laps_completed = 0
        self.last_waypoint_index = 0
        self.lap_start_step = 0
        
        # Controls state
        self.prev_space_held = False
        self.shift_held = False
        
        # Boost state
        self.boost_active = False
        self.boost_timer = 0
        
        # World generation
        self._generate_track()
        self._populate_track()
        
        # Place player at start
        start_pos = self.track_waypoints[0]
        next_pos = self.track_waypoints[1]
        self.player_pos = pygame.math.Vector2(start_pos)
        self.player_angle = math.degrees(math.atan2(-(next_pos[1] - start_pos[1]), next_pos[0] - start_pos[0]))

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_player_physics()
        self._update_game_state()
        
        self.steps += 1
        
        terminated = self._check_termination()
        
        # Base reward for survival
        self.reward += 0.01 
        self.score += self.reward
        
        return (
            self._get_observation(),
            self.reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, self.shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        turn_direction = 0
        if movement == 1:  # Up
            self.player_vel += pygame.math.Vector2(self.ACCELERATION, 0).rotate(-self.player_angle)
        elif movement == 2:  # Down
            self.player_vel -= pygame.math.Vector2(self.BRAKING, 0).rotate(-self.player_angle)
            if self.player_vel.length() > 0.1: # Add brake particles
                self._spawn_particles(2, self.COLOR_TRAIL, 2, 4, -self.player_vel.normalize() * 2, 0.9)
        elif movement == 3:  # Left
            turn_direction = 1
        elif movement == 4:  # Right
            turn_direction = -1
            
        # Turning
        turn_speed = self.TURN_SPEED
        if self.shift_held:
            turn_speed *= self.DRIFT_TURN_MOD
            if self.player_vel.length() > 1.5 and turn_direction != 0:
                # Spawn drift smoke
                self._spawn_particles(2, self.COLOR_DRIFT_SMOKE, 5, 10, pygame.math.Vector2(0,0), 0.85)

        # Inhibit turning at very low speeds for better control
        speed_factor = min(1.0, self.player_vel.length() / 2.0)
        self.player_angle += turn_direction * turn_speed * speed_factor

        # Boost
        if space_held and not self.prev_space_held and not self.boost_active:
            # For simplicity, this design doesn't require collecting boosts first.
            # A more complex game would check `if self.boost_charges > 0`.
            self.boost_active = True
            self.boost_timer = self.BOOST_DURATION
            # sfx: boost_activate.wav
        self.prev_space_held = space_held
        
    def _update_player_physics(self):
        # Boost effect
        current_max_speed = self.MAX_SPEED
        if self.boost_active:
            current_max_speed *= self.BOOST_SPEED_MULT
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.boost_active = False
                # sfx: boost_end.wav
        
        # Apply friction
        friction = self.FRICTION if not self.shift_held else self.FRICTION * self.DRIFT_FRICTION_MOD
        self.player_vel *= friction
        
        # Cap speed
        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)
            
        # Update position
        self.player_pos += self.player_vel

    def _update_game_state(self):
        # Update trail
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 20:
            self.player_trail.pop(0)
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= p['decay']

        # Check collisions with obstacles
        for obs in self.obstacles:
            if self.player_pos.distance_to(obs['pos']) < self.PLAYER_RADIUS + obs['radius']:
                self.game_over = True
                self.reward = -100 # Penalty for crashing
                # sfx: crash.wav
                return
        
        # Check collisions with boosts
        for boost in self.boosts[:]:
            if self.player_pos.distance_to(boost['pos']) < self.PLAYER_RADIUS + boost['radius']:
                self.boosts.remove(boost)
                self.boost_active = True
                self.boost_timer = self.BOOST_DURATION
                self.reward += 10 # Reward for collecting boost
                # sfx: collect_boost.wav
                
        # Check lap progress
        self._check_lap_completion()

    def _check_lap_completion(self):
        # Find closest waypoint
        min_dist = float('inf')
        closest_waypoint_index = -1
        for i, wp in enumerate(self.track_waypoints):
            dist = self.player_pos.distance_to(wp)
            if dist < min_dist:
                min_dist = dist
                closest_waypoint_index = i
        
        # Check for crossing finish line (waypoint 0)
        if self.last_waypoint_index > len(self.track_waypoints) * 0.8 and closest_waypoint_index < len(self.track_waypoints) * 0.2:
            self.laps_completed += 1
            self.lap_start_step = self.steps
            # sfx: lap_complete.wav
            if self.laps_completed < self.LAPS_TO_WIN:
                 self._populate_track() # Repopulate for next lap with increased difficulty
                 self.reward += 25 # Reward for completing a lap
            else:
                self.game_over = True # Win condition
                time_bonus = max(0, self.MAX_STEPS - self.steps)
                self.reward += 100 + (time_bonus / self.MAX_STEPS) * 50 # Win reward + time bonus
        
        self.last_waypoint_index = closest_waypoint_index

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.reward = -50 # Penalty for timeout
            return True
        return False

    def _generate_track(self):
        self.track_waypoints = []
        track_center = pygame.math.Vector2(0,0)
        
        # Use a seeded random for consistent track shapes per seed
        rng = random.Random(int(self.np_random.integers(0, 10000)))
        noise_offsets = [rng.uniform(0, 1000) for _ in range(2)]

        for i in range(self.TRACK_WAYPOINTS):
            angle = (i / self.TRACK_WAYPOINTS) * 2 * math.pi
            
            # Perlin-like noise for radius variation
            noise_val = (self._perlin_noise(i * 0.1 + noise_offsets[0]) - 0.5) * 2
            radius = self.TRACK_RADIUS_BASE + noise_val * self.TRACK_RADIUS_NOISE
            
            x = track_center.x + math.cos(angle) * radius
            y = track_center.y + math.sin(angle) * radius
            self.track_waypoints.append(pygame.math.Vector2(x, y))

    def _populate_track(self):
        # Clear existing objects for new lap
        if self.laps_completed == 0:
            self.obstacles = []
            self.boosts = []
        
        obstacle_density = 0.15 + (self.laps_completed * 0.1)
        boost_density = 0.05

        for i in range(len(self.track_waypoints)):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % len(self.track_waypoints)]
            
            # Don't place near finish line
            if i < 2 or i > len(self.track_waypoints) - 3:
                continue

            segment_vec = p2 - p1
            segment_len = segment_vec.length()
            if segment_len == 0: continue
            
            normal = segment_vec.rotate(90).normalize()
            
            # Place obstacles
            if self.np_random.random() < obstacle_density:
                pos = p1 + segment_vec * self.np_random.random()
                offset = normal * (self.np_random.random() - 0.5) * self.TRACK_WIDTH * 1.5
                self.obstacles.append({
                    'pos': pos + offset,
                    'radius': self.np_random.integers(8, 15)
                })

            # Place boosts
            if self.np_random.random() < boost_density:
                pos = p1 + segment_vec * self.np_random.random()
                offset = normal * (self.np_random.random() - 0.5) * self.TRACK_WIDTH * 0.5
                self.boosts.append({
                    'pos': pos + offset,
                    'radius': 10
                })

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self.game_objects_surface.fill((0,0,0,0))

        # Camera transform
        cam_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        cam_y = self.player_pos.y - self.SCREEN_HEIGHT / 2

        # Render track
        track_points_screen = [(p.x - cam_x, p.y - cam_y) for p in self.track_waypoints]
        pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, True, track_points_screen, self.TRACK_WIDTH + 10)
        pygame.draw.lines(self.screen, self.COLOR_TRACK, True, track_points_screen, self.TRACK_WIDTH)

        # Render finish line
        p1 = self.track_waypoints[0]
        p2 = self.track_waypoints[1]
        p_mid = p1.lerp(p2, 0.5)
        normal = (p2 - p1).rotate(90).normalize()
        
        f1 = p_mid - normal * (self.TRACK_WIDTH / 2 + 5)
        f2 = p_mid + normal * (self.TRACK_WIDTH / 2 + 5)
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE_1, (f1.x-cam_x, f1.y-cam_y), (f2.x-cam_x, f2.y-cam_y), 5)

        # Render world objects (obstacles, boosts)
        self._render_world_objects(cam_x, cam_y)
        self.screen.blit(self.game_objects_surface, (0,0))
        
        # Render player
        self._render_player()

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_world_objects(self, cam_x, cam_y):
        for obs in self.obstacles:
            sx, sy = int(obs['pos'].x - cam_x), int(obs['pos'].y - cam_y)
            if -50 < sx < self.SCREEN_WIDTH + 50 and -50 < sy < self.SCREEN_HEIGHT + 50:
                pygame.gfxdraw.filled_circle(self.game_objects_surface, sx, sy, obs['radius'], self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.game_objects_surface, sx, sy, obs['radius'], self.COLOR_OBSTACLE)
                
        for boost in self.boosts:
            sx, sy = int(boost['pos'].x - cam_x), int(boost['pos'].y - cam_y)
            if -50 < sx < self.SCREEN_WIDTH + 50 and -50 < sy < self.SCREEN_HEIGHT + 50:
                r = boost['radius']
                points = [
                    (sx, sy - r),
                    (sx - r * 0.866, sy + r * 0.5),
                    (sx + r * 0.866, sy + r * 0.5),
                ]
                pygame.gfxdraw.filled_trigon(self.game_objects_surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_BOOST)
                pygame.gfxdraw.aatrigon(self.game_objects_surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_BOOST)

    def _render_player(self):
        player_screen_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        # Render trail
        if self.player_trail:
            trail_color = self.COLOR_BOOST_TRAIL if self.boost_active else self.COLOR_TRAIL
            for i, pos in enumerate(self.player_trail):
                alpha = int(trail_color[3] * (i / len(self.player_trail)))
                cam_x = self.player_pos.x - self.SCREEN_WIDTH / 2
                cam_y = self.player_pos.y - self.SCREEN_HEIGHT / 2
                sx, sy = int(pos.x - cam_x), int(pos.y - cam_y)
                pygame.draw.circle(self.screen, (trail_color[0], trail_color[1], trail_color[2], alpha), (sx, sy), 2)

        # Render particles
        for p in self.particles:
            cam_x = self.player_pos.x - self.SCREEN_WIDTH / 2
            cam_y = self.player_pos.y - self.SCREEN_HEIGHT / 2
            sx, sy = int(p['pos'].x - cam_x), int(p['pos'].y - cam_y)
            alpha = int(p['color'][3] * (p['life'] / p['start_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pygame.draw.circle(self.screen, color, (sx, sy), int(p['radius']))

        # Render player glow
        if self.boost_active:
             self.screen.blit(self.player_glow_surface, (player_screen_pos[0] - self.PLAYER_RADIUS * 2, player_screen_pos[1] - self.PLAYER_RADIUS * 2), special_flags=pygame.BLEND_RGBA_ADD)

        # Render player car
        car_points = [
            pygame.math.Vector2(self.PLAYER_RADIUS * 1.2, 0),
            pygame.math.Vector2(-self.PLAYER_RADIUS * 0.6, -self.PLAYER_RADIUS * 0.8),
            pygame.math.Vector2(-self.PLAYER_RADIUS * 0.6, self.PLAYER_RADIUS * 0.8),
        ]
        rotated_points = [p.rotate(-self.player_angle) + player_screen_pos for p in car_points]
        pygame.gfxdraw.filled_trigon(self.screen, 
                                     int(rotated_points[0].x), int(rotated_points[0].y),
                                     int(rotated_points[1].x), int(rotated_points[1].y),
                                     int(rotated_points[2].x), int(rotated_points[2].y),
                                     self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, 
                                     int(rotated_points[0].x), int(rotated_points[0].y),
                                     int(rotated_points[1].x), int(rotated_points[1].y),
                                     int(rotated_points[2].x), int(rotated_points[2].y),
                                     self.COLOR_PLAYER)
        
    def _render_ui(self):
        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        self._draw_text(time_text, (10, 10), self.font_small, self.COLOR_UI_TEXT)

        # Laps
        lap_text = f"LAP: {min(self.laps_completed + 1, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}"
        self._draw_text(lap_text, (self.SCREEN_WIDTH - 10, 10), self.font_small, self.COLOR_UI_TEXT, align="right")
        
        # Speed
        speed_kmh = self.player_vel.length() * 20 # Arbitrary multiplier for display
        speed_text = f"{speed_kmh:.0f} KPH"
        self._draw_text(speed_text, (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 10), self.font_large, self.COLOR_UI_TEXT, align="right-bottom")

        # Game Over Message
        if self.game_over:
            if self.laps_completed >= self.LAPS_TO_WIN:
                msg = "FINISH!"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            else:
                msg = "CRASHED"
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, self.COLOR_UI_TEXT, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps_completed,
            "speed": self.player_vel.length()
        }
        
    def _draw_text(self, text, pos, font, color, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        elif align == "right-bottom":
            text_rect.bottomright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _perlin_noise(self, x, persistence=0.5, octaves=3):
        # Simple 1D Perlin-like noise function
        total = 0
        frequency = 1
        amplitude = 1
        for i in range(octaves):
            total += self._interpolated_noise(x * frequency) * amplitude
            amplitude *= persistence
            frequency *= 2
        return total / 2 + 0.5

    def _interpolated_noise(self, x):
        integer_x = int(x)
        fractional_x = x - integer_x
        v1 = self._smooth_noise(integer_x)
        v2 = self._smooth_noise(integer_x + 1)
        return self._interpolate(v1, v2, fractional_x)

    def _smooth_noise(self, x):
        # Simple hashing to get pseudo-random value
        x = (x << 13) ^ x
        return (1.0 - ((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)

    def _interpolate(self, a, b, x):
        ft = x * math.pi
        f = (1 - math.cos(ft)) * 0.5
        return a * (1 - f) + b * f
        
    def _create_glow_surface(self, surface, radius, color):
        width, height = surface.get_size()
        center = (width // 2, height // 2)
        for i in range(radius, 0, -1):
            alpha = int(color[3] * (1 - i / radius)**2)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(surface, center[0], center[1], i, (color[0], color[1], color[2], alpha))

    def _spawn_particles(self, count, color, min_radius, max_radius, base_vel, decay):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(0.5, 1.5)
            vel = pygame.math.Vector2(speed, 0).rotate(angle) + base_vel
            radius = self.np_random.uniform(min_radius, max_radius)
            life = int(radius / decay) * 2
            
            # Spawn particles from rear of car
            offset = pygame.math.Vector2(-self.PLAYER_RADIUS*0.8, 0).rotate(-self.player_angle)
            
            self.particles.append({
                'pos': self.player_pos + offset,
                'vel': vel,
                'radius': radius,
                'color': color,
                'decay': decay,
                'life': life,
                'start_life': life
            })

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys
    
    # Unset the dummy video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['laps']}, Steps: {info['steps']}")
    env.close()
    pygame.quit()
    sys.exit()