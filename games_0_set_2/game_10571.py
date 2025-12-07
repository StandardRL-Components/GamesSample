import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:42:16.532540
# Source Brief: brief_00571.md
# Brief Index: 571
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Race through a neon-drenched cyberpunk circuit, using portals and boosts to outmaneuver opponents."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to drive. Press space to boost and shift to activate portals."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_OPPONENT_1 = (255, 50, 50)
    COLOR_OPPONENT_2 = (50, 100, 255)
    COLOR_TRACK = (100, 30, 150)
    COLOR_PORTAL = (200, 0, 255)
    COLOR_BOOST = (255, 150, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_HIGH = (0, 200, 0)
    COLOR_HEALTH_LOW = (200, 0, 0)
    
    # Physics
    PLAYER_MAX_SPEED = 12
    PLAYER_ACCEL = 0.4
    PLAYER_BRAKE = 0.8
    PLAYER_TURN_SPEED = 0.08
    FRICTION = 0.96
    BOOST_MULTIPLIER = 1.8
    BOOST_CAPACITY = 100
    BOOST_REGEN = 0.3
    BOOST_COST = 1.5
    
    # Game Rules
    NUM_OPPONENTS = 5
    NUM_LAPS = 3
    MAX_STEPS = 2500
    COLLISION_DAMAGE = 15
    COLLISION_SPEED_PENALTY = 0.5

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
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)

        # Defer track creation to reset() so it can be seeded
        self.track_waypoints = None
        self.portals = None
        
        # Initialize state variables that are set in reset()
        self.player = None
        self.opponents = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_offset = pygame.Vector2(0, 0)
        
        self.reset()
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.track_waypoints = self._create_track()
        self.portals = [
            {'pos': self.track_waypoints[len(self.track_waypoints) // 3], 'type': 'forward'},
            {'pos': self.track_waypoints[2 * len(self.track_waypoints) // 3], 'type': 'backward'}
        ]
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        start_pos = self.track_waypoints[0]
        start_angle = math.atan2(
            self.track_waypoints[1].y - start_pos.y,
            self.track_waypoints[1].x - start_pos.x
        )

        self.player = self._Player(
            pos=pygame.Vector2(start_pos.x, start_pos.y),
            angle=start_angle
        )
        
        self.opponents = []
        for i in range(self.NUM_OPPONENTS):
            stagger_index = (len(self.track_waypoints) - (i + 1) * 5) % len(self.track_waypoints)
            pos = self.track_waypoints[stagger_index]
            opponent = self._Opponent(
                pos=pygame.Vector2(pos.x, pos.y),
                angle=start_angle,
                color=self.COLOR_OPPONENT_1 if i % 2 == 0 else self.COLOR_OPPONENT_2,
                waypoints=self.track_waypoints,
                speed=self.PLAYER_MAX_SPEED * 0.8
            )
            self.opponents.append(opponent)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        reward = 0
        
        # --- Player Logic ---
        self.player.update(movement, space_held, shift_held, self.portals, self.track_waypoints)
        
        # Reward for progress
        old_waypoint_idx = self.player.current_waypoint_index
        self.player.check_waypoints(self.track_waypoints)
        if self.player.current_waypoint_index != old_waypoint_idx:
            # Handle lap completion
            if self.player.current_waypoint_index < old_waypoint_idx:
                self.player.laps += 1
                reward += 5.0 # Lap completion bonus
            else:
                 reward += 0.1 # Progress bonus
        
        # --- Opponent Logic ---
        for i, opp in enumerate(self.opponents):
            opp.update()
            # Difficulty scaling
            if self.steps % 100 == 0:
                opp.max_speed = min(self.PLAYER_MAX_SPEED * 1.2, opp.max_speed + 0.005 * self.PLAYER_MAX_SPEED)
            
            # Simple opponent collision avoidance
            for j in range(i + 1, len(self.opponents)):
                other_opp = self.opponents[j]
                dist_vec = opp.pos - other_opp.pos
                if 0 < dist_vec.length() < 20:
                    dist_vec.scale_to_length(1)
                    opp.pos += dist_vec
                    other_opp.pos -= dist_vec

        # --- Collision Logic ---
        # Player vs Opponents
        for opp in self.opponents:
            if self.player.pos.distance_to(opp.pos) < 15: # Car radius approx 7.5
                self.player.handle_collision(opp.pos)
                opp.handle_collision(self.player.pos)
                reward -= 0.5
                self._create_sparks(self.player.pos)
        
        # Player vs Track boundaries
        if not self._is_on_track(self.player.pos):
            self.player.health = 0 # Fall off track
            reward -= 50
            self.game_over = True

        # --- Portal Logic ---
        if self.player.used_portal_this_step:
            reward += 0.2

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p.update()]
        if self.player.is_boosting:
            self._create_boost_particles()

        # --- Termination Conditions ---
        terminated = False
        if self.player.laps > self.NUM_LAPS:
            reward += 50 # Win bonus
            terminated = True
        elif self.player.health <= 0:
            reward -= 50 # Crash penalty
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Update camera to follow player
        self.camera_offset.x = self.player.pos.x - self.SCREEN_WIDTH / 2
        self.camera_offset.y = self.player.pos.y - self.SCREEN_HEIGHT / 2
        
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_track()
        self._render_portals()
        self._render_particles()

        for opponent in self.opponents:
            opponent.draw(self.screen, self.camera_offset)
        
        self.player.draw(self.screen, self.camera_offset)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        race_pos = 1 + sum(1 for opp in self.opponents if opp.get_progress() > self.player.get_progress())
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.player.laps,
            "position": race_pos,
            "health": self.player.health,
            "boost": self.player.boost,
        }

    # --- Rendering Methods ---
    def _render_track(self):
        track_width = 45
        for i in range(len(self.track_waypoints)):
            p1 = self.track_waypoints[i] - self.camera_offset
            p2 = self.track_waypoints[(i + 1) % len(self.track_waypoints)] - self.camera_offset
            
            # Draw glowing effect
            for w in range(track_width, 5, -5):
                alpha = 40 - w
                pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, w)
            
            pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 2) # Center line

        # Finish line
        p1 = self.track_waypoints[0]
        p2 = self.track_waypoints[1]
        perp = (p2 - p1).rotate(90).normalize()
        f1 = p1 + perp * track_width/2 - self.camera_offset
        f2 = p1 - perp * track_width/2 - self.camera_offset
        pygame.draw.line(self.screen, (255, 255, 255), f1, f2, 5)

    def _render_portals(self):
        for portal in self.portals:
            pos = portal['pos'] - self.camera_offset
            radius = 20 + math.sin(self.steps * 0.1) * 3
            for r in range(int(radius), 0, -3):
                alpha = 150 * (r / radius)
                color = self.COLOR_PORTAL + (int(alpha),)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), r, color)

    def _render_stars(self):
        if not hasattr(self, '_stars'):
            self._stars = [
                (self.np_random.integers(0, 2001), 
                 self.np_random.integers(0, 2001), 
                 self.np_random.integers(1, 4)) 
                for _ in range(200)
            ]
        
        for x, y, size in self._stars:
            # Parallax effect
            screen_x = (x - self.camera_offset.x * 0.1) % self.SCREEN_WIDTH
            screen_y = (y - self.camera_offset.y * 0.1) % self.SCREEN_HEIGHT
            color = (size * 50, size * 50, size * 60)
            self.screen.fill(color, (screen_x, screen_y, size, size))
    
    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self.camera_offset)

    def _render_ui(self):
        lap_text = self.font.render(f"LAP: {min(self.player.laps, self.NUM_LAPS)}/{self.NUM_LAPS}", True, self.COLOR_UI_TEXT)
        pos_text = self.font.render(f"POS: {self._get_info()['position']}/{self.NUM_OPPONENTS + 1}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))
        self.screen.blit(pos_text, (10, 35))

        # Health Bar
        health_pct = max(0, self.player.health / 100)
        health_color = [int(a + (b - a) * health_pct) for a, b in zip(self.COLOR_HEALTH_LOW, self.COLOR_HEALTH_HIGH)]
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 65, 150, 15))
        pygame.draw.rect(self.screen, health_color, (10, 65, 150 * health_pct, 15))
        
        # Boost Bar
        boost_pct = max(0, self.player.boost / self.BOOST_CAPACITY)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 85, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_BOOST, (10, 85, 150 * boost_pct, 15))

    # --- Helper Methods ---
    def _create_track(self):
        points = []
        center_x, center_y = 1000, 1000
        radius = 800
        num_points = 60
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + math.cos(angle) * (radius + self.np_random.uniform(-1, 1) * 150)
            y = center_y + math.sin(angle) * (radius + self.np_random.uniform(-1, 1) * 150)
            points.append(pygame.Vector2(x, y))
        return points

    def _is_on_track(self, pos):
        min_dist_sq = float('inf')
        for i in range(len(self.track_waypoints)):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % len(self.track_waypoints)]
            
            l2 = p1.distance_squared_to(p2)
            if l2 == 0:
                dist_sq = pos.distance_squared_to(p1)
            else:
                t = max(0, min(1, (pos - p1).dot(p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist_sq = pos.distance_squared_to(projection)
            min_dist_sq = min(min_dist_sq, dist_sq)
        
        return min_dist_sq < (45 ** 2) # Track width squared

    def _create_sparks(self, pos):
        for _ in range(10):
            self.particles.append(self._Particle(pos, (255, 200, 100), 20, 2, self.np_random))
    
    def _create_boost_particles(self):
        for _ in range(2):
            offset = pygame.Vector2(-10, 0).rotate(-math.degrees(self.player.angle))
            self.particles.append(self._Particle(self.player.pos + offset, self.COLOR_BOOST, 15, 4, self.np_random))
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    # --- Inner Classes for Game Entities ---
    class _Player:
        def __init__(self, pos, angle):
            self.pos = pos
            self.vel = pygame.Vector2(0, 0)
            self.angle = angle
            self.health = 100
            self.boost = GameEnv.BOOST_CAPACITY
            self.is_boosting = False
            self.laps = 1
            self.current_waypoint_index = 0
            self.used_portal_this_step = False
            self.portal_cooldown = 0

        def update(self, movement, boost_held, portal_held, portals, track_waypoints):
            self.used_portal_this_step = False
            
            # Steering
            if movement == 3: # Left
                self.angle -= GameEnv.PLAYER_TURN_SPEED * (self.vel.length() / GameEnv.PLAYER_MAX_SPEED)
            if movement == 4: # Right
                self.angle += GameEnv.PLAYER_TURN_SPEED * (self.vel.length() / GameEnv.PLAYER_MAX_SPEED)

            # Acceleration/Braking
            thrust = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
            if movement == 1: # Accelerate
                self.vel += thrust * GameEnv.PLAYER_ACCEL
            if movement == 2: # Brake
                self.vel *= (1.0 - GameEnv.PLAYER_BRAKE / GameEnv.FPS)

            # Boost
            self.is_boosting = boost_held and self.boost > 0
            if self.is_boosting:
                self.vel += thrust * GameEnv.PLAYER_ACCEL * GameEnv.BOOST_MULTIPLIER
                self.boost -= GameEnv.BOOST_COST
            else:
                self.boost = min(GameEnv.BOOST_CAPACITY, self.boost + GameEnv.BOOST_REGEN)

            # Physics
            max_speed = GameEnv.PLAYER_MAX_SPEED * (GameEnv.BOOST_MULTIPLIER if self.is_boosting else 1)
            if self.vel.length() > max_speed:
                self.vel.scale_to_length(max_speed)
            
            self.pos += self.vel
            self.vel *= GameEnv.FRICTION

            # Portals
            self.portal_cooldown = max(0, self.portal_cooldown - 1)
            if portal_held and self.portal_cooldown == 0:
                for portal in portals:
                    if portal['type'] == 'forward' and self.pos.distance_to(portal['pos']) < 30:
                        self.used_portal_this_step = True
                        self.portal_cooldown = 30 # 1 second cooldown
                        target_waypoint_idx = (self.current_waypoint_index + 5) % len(track_waypoints)
                        self.pos = pygame.Vector2(track_waypoints[target_waypoint_idx])
                        self.current_waypoint_index = target_waypoint_idx
                        break

        def draw(self, surface, camera_offset):
            # Car body
            screen_pos = self.pos - camera_offset
            points = [
                pygame.Vector2(10, 0), pygame.Vector2(-7, 7),
                pygame.Vector2(-7, -7)
            ]
            rotated_points = [p.rotate(math.degrees(self.angle)) + screen_pos for p in points]
            
            # Glow effect
            for i in range(5, 0, -1):
                glow_color = list(GameEnv.COLOR_PLAYER) + [50 - i*10]
                pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in rotated_points], glow_color)
            
            pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in rotated_points], GameEnv.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in rotated_points], GameEnv.COLOR_PLAYER)

        def handle_collision(self, other_pos):
            self.health -= GameEnv.COLLISION_DAMAGE
            self.vel *= GameEnv.COLLISION_SPEED_PENALTY
            # Knockback
            knockback = (self.pos - other_pos).normalize() * 5
            self.vel += knockback

        def check_waypoints(self, waypoints):
            next_waypoint_idx = (self.current_waypoint_index + 1) % len(waypoints)
            if self.pos.distance_to(waypoints[next_waypoint_idx]) < 60:
                self.current_waypoint_index = next_waypoint_idx
        
        def get_progress(self):
            return self.laps * 1000 + self.current_waypoint_index

    class _Opponent:
        def __init__(self, pos, angle, color, waypoints, speed):
            self.pos = pos
            self.vel = pygame.Vector2(0, 0)
            self.angle = angle
            self.color = color
            self.waypoints = waypoints
            self.current_waypoint_index = 0
            self.max_speed = speed

        def update(self):
            target = self.waypoints[self.current_waypoint_index]
            dist = self.pos.distance_to(target)
            
            if dist < 80:
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                target = self.waypoints[self.current_waypoint_index]

            target_angle = math.atan2(target.y - self.pos.y, target.x - self.pos.x)
            
            # Smooth steering
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += np.clip(angle_diff, -0.1, 0.1)

            thrust = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))
            self.vel += thrust * (GameEnv.PLAYER_ACCEL * 0.8)

            if self.vel.length() > self.max_speed:
                self.vel.scale_to_length(self.max_speed)
            
            self.pos += self.vel
            self.vel *= GameEnv.FRICTION

        def draw(self, surface, camera_offset):
            screen_pos = self.pos - camera_offset
            points = [pygame.Vector2(10, 0), pygame.Vector2(-7, 5), pygame.Vector2(-7, -5)]
            rotated_points = [p.rotate(math.degrees(self.angle)) + screen_pos for p in points]
            
            # Glow effect
            for i in range(4, 0, -1):
                glow_color = list(self.color) + [40 - i*10]
                pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in rotated_points], glow_color)

            pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in rotated_points], self.color)

        def handle_collision(self, other_pos):
            self.vel *= 0.8
            knockback = (self.pos - other_pos).normalize() * 2
            self.vel += knockback
        
        def get_progress(self):
            # A rough estimation of progress for ranking
            dist_to_next = self.pos.distance_to(self.waypoints[self.current_waypoint_index])
            return self.current_waypoint_index - dist_to_next * 0.001

    class _Particle:
        def __init__(self, pos, color, lifetime, speed, np_random):
            self.pos = pygame.Vector2(pos)
            vel_x = np_random.uniform(-1, 1)
            vel_y = np_random.uniform(-1, 1)
            self.vel = pygame.Vector2(vel_x, vel_y).normalize() * speed
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime

        def update(self):
            self.pos += self.vel
            self.lifetime -= 1
            return self.lifetime > 0

        def draw(self, surface, camera_offset):
            screen_pos = self.pos - camera_offset
            radius = int(max(0, (self.lifetime / self.max_lifetime) * 5))
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), radius, self.color)


if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # Use a display window if running standalone
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyberpunk Racer")
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Manual control mapping
    # W/A/S/D for movement, SPACE for boost, LSHIFT for portal
    while not done:
        movement = 0 # None
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    pygame.quit()