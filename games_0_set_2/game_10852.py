import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T12:03:00.632429
# Source Brief: brief_00852.md
# Brief Index: 852
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for vector operations
class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2D(self.x * scalar, self.y * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vec2D(0, 0)
        return Vec2D(self.x / mag, self.y / mag)

    def to_tuple(self):
        return (self.x, self.y)

# Helper class for particles
class Particle:
    def __init__(self, x, y, color, size, life, vel_x, vel_y):
        self.pos = Vec2D(x, y)
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.vel = Vec2D(vel_x, vel_y)

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size = max(0, self.size * (self.life / self.max_life))

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        color_with_alpha = self.color + (alpha,)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), color_with_alpha)

# Helper class for Kart entities
class Kart:
    def __init__(self, x, y, color, is_player=False):
        self.pos = Vec2D(x, y)
        self.vel = Vec2D(0, 0)
        self.angle = -math.pi / 2  # Start facing up
        self.color = color
        self.is_player = is_player
        self.lap = 1
        self.next_waypoint_idx = 1
        self.finished_rank = None
        self.vulnerable_timer = 0
        self.trail = []

    def get_progress(self, num_waypoints):
        return (self.lap - 1) * num_waypoints + self.next_waypoint_idx

    def update_trail(self):
        self.trail.append(self.pos.to_tuple())
        if len(self.trail) > 20:
            self.trail.pop(0)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A futuristic rhythm-based racing game. Time your actions with the beat indicator to steer and "
        "accelerate through procedurally generated tracks."
    )
    user_guide = (
        "Controls: Use ←→ to steer and ↑ to accelerate. Time your actions with the beat indicator at "
        "the bottom for a speed boost and to avoid penalties."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased for 3 laps
        self.NUM_LAPS = 3
        self.NUM_OPPONENTS = 3

        # Colors (Vibrant Neon)
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_OPPONENT_A = (255, 50, 50)
        self.COLOR_OPPONENT_B = (50, 150, 255)
        self.COLOR_PORTAL = (180, 0, 255)
        self.COLOR_TRACK = (40, 80, 150)
        self.COLOR_OBSTACLE = (255, 128, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_VULNERABLE = (255, 200, 0)
        
        # EXACT spaces:
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player = None
        self.opponents = []
        self.portals = []
        self.obstacles = []
        self.particles = []
        self.track_waypoints = []
        self.track_polygon = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Rhythm Game Mechanics
        self.BPM = 120
        self.beat_period = (self.FPS * 60) / self.BPM
        self.beat_timer = 0
        self.on_beat_window = 3 # frames before/after the beat
        
        self.np_random = None

    def _generate_track(self):
        self.track_waypoints = []
        self.obstacles = []
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        radius_x, radius_y = self.WIDTH * 0.4, self.HEIGHT * 0.35
        num_points = 16
        
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            rand_x = self.np_random.uniform(0.8, 1.2)
            rand_y = self.np_random.uniform(0.8, 1.2)
            x = center_x + math.cos(angle) * radius_x * rand_x
            y = center_y + math.sin(angle) * radius_y * rand_y
            points.append(Vec2D(x, y))
        
        # Ensure it's a loop
        points.append(points[0])
        self.track_waypoints = points
        
        # Generate track polygon for collision and rendering
        track_width = 40
        left_border, right_border = [], []
        for i in range(len(self.track_waypoints) - 1):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[i+1]
            
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            normal = Vec2D(-math.sin(angle), math.cos(angle))
            
            left_border.append((p1 + normal * track_width).to_tuple())
            right_border.append((p1 - normal * track_width).to_tuple())
        
        self.track_polygon = left_border + right_border[::-1]

        # Generate obstacles
        for i in range(1, len(self.track_waypoints) - 1, 3):
             p1 = self.track_waypoints[i]
             angle = math.atan2(self.track_waypoints[i+1].y - p1.y, self.track_waypoints[i+1].x - p1.x)
             normal = Vec2D(-math.sin(angle), math.cos(angle))
             offset = self.np_random.uniform(15, 25) * self.np_random.choice([-1, 1])
             pos = p1 + normal * offset
             self.obstacles.append(pos)

    def _generate_portals(self):
        self.portals = []
        for i in range(2, len(self.track_waypoints) - 1, 5):
            pos = self.track_waypoints[i]
            # Teleports 2 waypoints ahead
            target_idx = (i + 2) % (len(self.track_waypoints) -1)
            self.portals.append({'pos': pos, 'target_idx': target_idx, 'radius': 15})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_track()
        self._generate_portals()

        start_pos = self.track_waypoints[0]
        self.player = Kart(start_pos.x, start_pos.y + 10, self.COLOR_PLAYER, is_player=True)
        
        self.opponents = []
        for i in range(self.NUM_OPPONENTS):
            offset = (i - 1) * 15
            color = self.COLOR_OPPONENT_A if i % 2 == 0 else self.COLOR_OPPONENT_B
            self.opponents.append(Kart(start_pos.x + offset, start_pos.y - 10, color))

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beat_timer = 0
        self.particles = []
        self.opponent_base_speed = 1.5
        
        return self._get_observation(), self._get_info()

    def _is_on_beat(self):
        return self.beat_timer % self.beat_period < self.on_beat_window or \
               self.beat_period - (self.beat_timer % self.beat_period) < self.on_beat_window

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.beat_timer += 1
        reward = 0

        # -- Action Handling --
        movement = action[0]
        action_taken = movement in [1, 2, 3]

        prev_progress = self.player.get_progress(len(self.track_waypoints))

        if action_taken:
            if self._is_on_beat():
                # On-beat action
                if movement == 1: # Steer Left
                    self.player.angle -= 0.15
                elif movement == 2: # Steer Right
                    self.player.angle += 0.15
                elif movement == 3: # Accelerate
                    accel = Vec2D(math.cos(self.player.angle), math.sin(self.player.angle)) * 0.5
                    self.player.vel += accel
                # sfx: positive_beat.wav
                for _ in range(5):
                    self.particles.append(Particle(self.player.pos.x, self.player.pos.y, self.COLOR_PLAYER, 5, 15, self.np_random.uniform(-2,2), self.np_random.uniform(-2,2)))
            else:
                # Off-beat action
                self.player.vulnerable_timer = 60 # 2 seconds
                self.player.vel *= 0.95 # Slow down
                reward -= 0.5
                # sfx: negative_beat.wav
                for _ in range(5):
                    self.particles.append(Particle(self.player.pos.x, self.player.pos.y, self.COLOR_VULNERABLE, 5, 15, self.np_random.uniform(-1,1), self.np_random.uniform(-1,1)))

        # -- Physics and Game Logic Update --
        # Player update
        self.player.vel *= 0.97  # Friction
        if self.player.vel.magnitude() > 5: self.player.vel = self.player.vel.normalize() * 5
        self.player.pos += self.player.vel
        self.player.update_trail()
        if self.player.vulnerable_timer > 0: self.player.vulnerable_timer -= 1
        
        # Opponent AI update
        for opp in self.opponents:
            if opp.finished_rank is not None: continue
            
            target_waypoint = self.track_waypoints[opp.next_waypoint_idx]
            dist_to_target = (target_waypoint - opp.pos).magnitude()

            if dist_to_target < 30:
                opp.next_waypoint_idx += 1
                if opp.next_waypoint_idx >= len(self.track_waypoints) - 1:
                    opp.next_waypoint_idx = 0
                    opp.lap += 1

            target_angle = math.atan2(target_waypoint.y - opp.pos.y, target_waypoint.x - opp.pos.x)
            # Smooth steering
            angle_diff = (target_angle - opp.angle + math.pi) % (2 * math.pi) - math.pi
            opp.angle += np.clip(angle_diff, -0.1, 0.1)

            speed = self.opponent_base_speed + (opp.lap - 1) * 0.05
            opp.vel = Vec2D(math.cos(opp.angle), math.sin(opp.angle)) * speed
            opp.pos += opp.vel
            opp.update_trail()

            # Opponent attacks if player is vulnerable
            if self.player.vulnerable_timer > 0 and (opp.pos - self.player.pos).magnitude() < 50 and self.np_random.random() < 0.05:
                reward -= 2.0
                self.player.vel *= 0.5 # Hit effect
                # sfx: opponent_hit.wav
                for _ in range(10):
                    self.particles.append(Particle(self.player.pos.x, self.player.pos.y, opp.color, 7, 20, self.np_random.uniform(-3,3), self.np_random.uniform(-3,3)))


        # -- Collision and State Checks --
        terminated = False
        
        # Player waypoint and lap check
        player_target_waypoint = self.track_waypoints[self.player.next_waypoint_idx]
        if (player_target_waypoint - self.player.pos).magnitude() < 30:
            if self.player.next_waypoint_idx == 0: # Crossed finish line
                self.player.lap += 1
                self.opponent_base_speed += 0.05 # Opponents get faster
                self._generate_track() # New track layout
                self._generate_portals()
                # sfx: lap_complete.wav
                if self.player.lap > self.NUM_LAPS:
                    self.game_over = True
            self.player.next_waypoint_idx = (self.player.next_waypoint_idx + 1) % (len(self.track_waypoints) - 1)

        # Progress reward
        current_progress = self.player.get_progress(len(self.track_waypoints))
        if current_progress > prev_progress:
            reward += 0.1
        elif current_progress < prev_progress: # Moving backwards (unlikely but possible)
            reward -= 0.1

        # Portal collision
        for portal in self.portals:
            if (portal['pos'] - self.player.pos).magnitude() < portal['radius']:
                self.player.next_waypoint_idx = portal['target_idx']
                self.player.pos = self.track_waypoints[portal['target_idx']]
                reward += 5.0
                # sfx: portal_whoosh.wav
                for _ in range(30):
                    self.particles.append(Particle(self.player.pos.x, self.player.pos.y, self.COLOR_PORTAL, 10, 30, self.np_random.uniform(-4,4), self.np_random.uniform(-4,4)))
                break

        # Obstacle collision
        for obs in self.obstacles:
            if (obs - self.player.pos).magnitude() < 10:
                reward = -100
                terminated = True
                self.game_over = True
                # sfx: crash.wav
                break
        
        # Out of bounds check
        if not self._is_inside_track(self.player.pos):
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: fall_off.wav

        # Termination conditions
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if self.game_over:
            self._calculate_final_ranks()
            if self.player.finished_rank == 1: reward += 100
            elif self.player.finished_rank == 2: reward += 50
            elif self.player.finished_rank == 3: reward += 25
            elif self.player.finished_rank == 4: reward += 10

        self.score += reward
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_inside_track(self, point):
        # Ray casting algorithm
        x, y = point.to_tuple()
        n = len(self.track_polygon)
        inside = False
        p1x, p1y = self.track_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.track_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _calculate_final_ranks(self):
        all_karts = [self.player] + self.opponents
        # Sort by progress (higher is better)
        all_karts.sort(key=lambda k: k.get_progress(len(self.track_waypoints)), reverse=True)
        for i, kart in enumerate(all_karts):
            if kart.finished_rank is None:
                kart.finished_rank = i + 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

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
            "lap": self.player.lap,
            "rank": self.player.finished_rank or self._get_current_rank()
        }

    def _get_current_rank(self):
        all_karts = [self.player] + self.opponents
        all_karts.sort(key=lambda k: k.get_progress(len(self.track_waypoints)), reverse=True)
        try:
            return all_karts.index(self.player) + 1
        except ValueError:
            return len(all_karts)

    def _render_game(self):
        # Background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Track
        pygame.gfxdraw.aapolygon(self.screen, self.track_polygon, self.COLOR_TRACK)
        pygame.gfxdraw.filled_polygon(self.screen, self.track_polygon, self.COLOR_TRACK)

        # Portals
        for portal in self.portals:
            radius = portal['radius'] * (1 + 0.1 * math.sin(self.steps * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, int(portal['pos'].x), int(portal['pos'].y), int(radius), self.COLOR_PORTAL + (100,))
            pygame.gfxdraw.aacircle(self.screen, int(portal['pos'].x), int(portal['pos'].y), int(radius), self.COLOR_PORTAL)

        # Obstacles
        for obs in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(obs.x), int(obs.y), 5, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(obs.x), int(obs.y), 5, self.COLOR_OBSTACLE)

        # Karts (render opponents first)
        all_karts = self.opponents + [self.player]
        for kart in all_karts:
            # Trail
            if len(kart.trail) > 1:
                alpha_trail_color = kart.color + (50,)
                pygame.draw.lines(self.screen, alpha_trail_color, False, kart.trail, width=5)

            # Kart Body
            p1 = (kart.pos.x + math.cos(kart.angle) * 12, kart.pos.y + math.sin(kart.angle) * 12)
            p2 = (kart.pos.x + math.cos(kart.angle + 2.5) * 8, kart.pos.y + math.sin(kart.angle + 2.5) * 8)
            p3 = (kart.pos.x + math.cos(kart.angle - 2.5) * 8, kart.pos.y + math.sin(kart.angle - 2.5) * 8)
            points = [p1, p2, p3]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            
            # Glow effect
            glow_color = kart.color + (60,)
            pygame.gfxdraw.filled_trigon(self.screen, int_points[0][0], int_points[0][1], int_points[1][0], int_points[1][1], int_points[2][0], int_points[2][1], glow_color)
            
            # Main body
            pygame.gfxdraw.aapolygon(self.screen, int_points, kart.color)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, kart.color)

            # Vulnerable indicator
            if kart.is_player and kart.vulnerable_timer > 0:
                blink = (self.steps // 5) % 2 == 0
                if blink:
                    pygame.gfxdraw.aacircle(self.screen, int(kart.pos.x), int(kart.pos.y), 15, self.COLOR_VULNERABLE)

        # Particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Lap Counter
        lap_text = f"LAP: {min(self.player.lap, self.NUM_LAPS)} / {self.NUM_LAPS}"
        lap_surf = self.font_main.render(lap_text, True, self.COLOR_TEXT)
        self.screen.blit(lap_surf, (10, 10))

        # Rank
        rank = self.player.finished_rank or self._get_current_rank()
        rank_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(rank, f"{rank}th")
        rank_text = f"POS: {rank_str}"
        rank_surf = self.font_main.render(rank_text, True, self.COLOR_TEXT)
        self.screen.blit(rank_surf, (self.WIDTH - rank_surf.get_width() - 10, 10))

        # Beat Indicator
        beat_bar_width = 200
        beat_bar_height = 10
        beat_bar_x = (self.WIDTH - beat_bar_width) / 2
        beat_bar_y = self.HEIGHT - 30
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (beat_bar_x, beat_bar_y, beat_bar_width, beat_bar_height), border_radius=5)
        
        progress = (self.beat_timer % self.beat_period) / self.beat_period
        indicator_pos = beat_bar_x + progress * beat_bar_width
        
        if self._is_on_beat():
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (beat_bar_x, beat_bar_y, beat_bar_width, beat_bar_height), border_radius=5)
        
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (indicator_pos - 2, beat_bar_y - 2, 4, beat_bar_height + 4), border_radius=2)
        
        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            finish_text = f"RACE FINISHED! You placed {rank_str}"
            finish_surf = self.font_main.render(finish_text, True, self.COLOR_TEXT)
            self.screen.blit(finish_surf, (self.WIDTH/2 - finish_surf.get_width()/2, self.HEIGHT/2 - finish_surf.get_height()/2))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to re-enable the display driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Racer")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement_action = 0 # no-op
        if keys[pygame.K_UP]:
            movement_action = 3 # accelerate
        elif keys[pygame.K_LEFT]:
            movement_action = 1 # steer left
        elif keys[pygame.K_RIGHT]:
            movement_action = 2 # steer right
        
        action = [movement_action, 0, 0] # space/shift not used for manual play
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Rank: {info['rank']}, Steps: {info['steps']}")
            # Wait for a moment before resetting to show the final screen
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        clock.tick(env.FPS)
        
    env.close()