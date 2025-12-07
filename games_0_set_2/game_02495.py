
# Generated: 2025-08-28T05:01:14.676807
# Source Brief: brief_02495.md
# Brief Index: 2495

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper Snail class
class Snail:
    def __init__(self, x, y, angle, color, shell_color):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle
        self.color = color
        self.shell_color = shell_color
        self.size = 15
        self.next_waypoint_idx = 1
        self.lap = 0
        self.is_finished = False
        self.finish_time = float('inf')

# Helper Particle class
class Particle:
    def __init__(self, x, y, angle, speed, color, lifetime):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(speed, 0).rotate(-angle)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to use a speed boost."
    )

    game_description = (
        "Fast-paced arcade snail racer. Drift through procedurally generated corners, "
        "grab boosts, and race against AI opponents to finish first."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        try:
            self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 60)


        # Colors
        self.COLOR_BG = (50, 50, 60)
        self.COLOR_TRACK = (70, 80, 70)
        self.COLOR_WALL = (120, 150, 120)
        self.COLOR_FINISH_1 = (200, 200, 200)
        self.COLOR_FINISH_2 = (30, 30, 30)

        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_SHELL = (255, 150, 150)
        self.COLOR_AI1 = (80, 80, 255)
        self.COLOR_AI1_SHELL = (150, 150, 255)
        self.COLOR_AI2 = (255, 255, 80)
        self.COLOR_AI2_SHELL = (255, 255, 150)

        self.COLOR_BOOST_PARTICLE = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # Game parameters
        self.FPS = 30
        self.MAX_TIME = 60 * self.FPS
        self.TOTAL_LAPS = 3

        # Physics
        self.ACCELERATION = 0.1
        self.BRAKING = 0.2
        self.FRICTION = 0.98
        self.MAX_SPEED = 3.0
        self.TURN_SPEED = 3.0
        self.BOOST_SPEED = 6.0
        self.BOOST_DURATION = 15 # frames
        self.MAX_BOOSTS = 3
        self.WALL_PENALTY = 0.5

        # AI parameters
        self.AI_BASE_SPEED = 2.0
        self.AI_SPEED_LAP_INCREASE = 0.15

        # State variables
        self.player = None
        self.opponents = []
        self.track_center = []
        self.track_inner_wall = []
        self.track_outer_wall = []
        self.track_width = 0
        self.particles = []
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.boost_cooldown = 0
        self.boosts_left = 0
        self.rng = None

        self.reset()
        
    def _generate_track(self):
        self.track_center.clear()
        self.track_inner_wall.clear()
        self.track_outer_wall.clear()
        self.track_width = 80

        num_points = 12
        map_width, map_height = 1600, 1200
        center_x, center_y = map_width / 2, map_height / 2
        radius_x, radius_y = 600, 400
        
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            rand_factor_x = self.rng.uniform(0.8, 1.2)
            rand_factor_y = self.rng.uniform(0.8, 1.2)
            x = center_x + math.cos(angle) * radius_x * rand_factor_x
            y = center_y + math.sin(angle) * radius_y * rand_factor_y
            points.append(pygame.math.Vector2(x, y))

        for i in range(num_points):
            p0 = points[(i - 1 + num_points) % num_points]
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p3 = points[(i + 2) % num_points]

            for t_val in np.linspace(0, 1, 15, endpoint=False):
                t2, t3 = t_val * t_val, t_val * t_val * t_val
                tx = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t_val + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3)
                ty = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t_val + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
                self.track_center.append(pygame.math.Vector2(tx, ty))

        for i in range(len(self.track_center)):
            p1 = self.track_center[i]
            p2 = self.track_center[(i + 1) % len(self.track_center)]
            
            direction = (p2 - p1)
            if direction.length() > 0:
                normal = direction.normalize().rotate(90)
                self.track_inner_wall.append(p1 - normal * self.track_width / 2)
                self.track_outer_wall.append(p1 + normal * self.track_width / 2)
            elif len(self.track_inner_wall) > 0:
                self.track_inner_wall.append(self.track_inner_wall[-1])
                self.track_outer_wall.append(self.track_outer_wall[-1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self._generate_track()

        start_pos = self.track_center[0]
        start_angle_vec = self.track_center[1] - self.track_center[0]
        start_angle = -start_angle_vec.angle_to(pygame.math.Vector2(1, 0))

        self.player = Snail(start_pos.x, start_pos.y, start_angle, self.COLOR_PLAYER, self.COLOR_PLAYER_SHELL)
        self.opponents = [
            Snail(start_pos.x - 20, start_pos.y, start_angle, self.COLOR_AI1, self.COLOR_AI1_SHELL),
            Snail(start_pos.x + 20, start_pos.y, start_angle, self.COLOR_AI2, self.COLOR_AI2_SHELL),
        ]
        
        self.particles.clear()
        self.time_remaining = self.MAX_TIME
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.boost_cooldown = 0
        self.boosts_left = self.MAX_BOOSTS
        
        for snail in [self.player] + self.opponents:
            snail.lap = 0
            snail.next_waypoint_idx = 1
            snail.is_finished = False
            snail.finish_time = float('inf')

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        player_direction = pygame.math.Vector2(1, 0).rotate(-self.player.angle)
        prev_dist_to_waypoint = self.player.pos.distance_to(self.track_center[self.player.next_waypoint_idx])

        if movement == 1: self.player.vel += player_direction * self.ACCELERATION
        elif movement == 2: self.player.vel -= player_direction * self.BRAKING
        if movement == 3: self.player.angle += self.TURN_SPEED
        elif movement == 4: self.player.angle -= self.TURN_SPEED

        if space_held and self.boosts_left > 0 and self.boost_cooldown == 0:
            self.boost_cooldown = self.BOOST_DURATION
            self.boosts_left -= 1
            reward += 0.5  # Reward for using boost
            # Sound: Boost activate

        if self.boost_cooldown > 0:
            self.player.vel = player_direction * self.BOOST_SPEED
            self.boost_cooldown -= 1
            for _ in range(3):
                p_angle = self.player.angle + 180 + self.rng.uniform(-20, 20)
                p_speed = self.rng.uniform(1, 3)
                p_pos = self.player.pos - player_direction * self.player.size
                self.particles.append(Particle(p_pos.x, p_pos.y, p_angle, p_speed, self.COLOR_BOOST_PARTICLE, 20))

        self.player.vel *= self.FRICTION
        if self.player.vel.length() > self.MAX_SPEED and self.boost_cooldown == 0:
            self.player.vel.scale_to_length(self.MAX_SPEED)
        self.player.pos += self.player.vel

        for ai in self.opponents:
            if ai.is_finished: continue
            target_waypoint = self.track_center[ai.next_waypoint_idx]
            ai_dir = target_waypoint - ai.pos
            if ai_dir.length() < self.track_width * 0.75:
                ai.next_waypoint_idx = (ai.next_waypoint_idx + 1) % len(self.track_center)
            target_angle = -ai_dir.angle_to(pygame.math.Vector2(1,0))
            angle_diff = (target_angle - ai.angle + 540) % 360 - 180
            ai.angle += np.clip(angle_diff, -self.TURN_SPEED * 1.2, self.TURN_SPEED * 1.2)
            ai_forward = pygame.math.Vector2(1, 0).rotate(-ai.angle)
            current_ai_speed = self.AI_BASE_SPEED + (self.player.lap * self.AI_SPEED_LAP_INCREASE)
            ai.vel = ai_forward * (current_ai_speed + self.rng.uniform(-0.1, 0.1))
            ai.pos += ai.vel

        all_snails = [self.player] + self.opponents
        for snail in all_snails:
            if snail.is_finished: continue
            
            for i in range(len(self.track_center)):
                p1_in, p2_in = self.track_inner_wall[i], self.track_inner_wall[(i + 1) % len(self.track_inner_wall)]
                p1_out, p2_out = self.track_outer_wall[i], self.track_outer_wall[(i + 1) % len(self.track_outer_wall)]
                if self._point_segment_distance(snail.pos, p1_in, p2_in) < snail.size or \
                   self._point_segment_distance(snail.pos, p1_out, p2_out) < snail.size:
                    snail.vel *= self.WALL_PENALTY
                    if snail == self.player:
                        reward -= 0.2 # Penalty for hitting wall
                        # Sound: Collision thump
                    break
            
            if snail.next_waypoint_idx > len(self.track_center) * 0.9:
                p1, p2 = self.track_center[0], self.track_center[1]
                finish_center = (p1 + p2) / 2
                if self.player.pos.distance_to(finish_center) < self.track_width:
                    finish_line_start = self.track_inner_wall[0]
                    finish_line_end = self.track_outer_wall[0]
                    if self._line_segment_intersection(snail.pos - snail.vel, snail.pos, finish_line_start, finish_line_end):
                        snail.lap += 1
                        snail.next_waypoint_idx = 1
                        if snail == self.player:
                            reward += 10.0 # Lap completion reward
                            # Sound: Lap complete chime
                        if snail.lap >= self.TOTAL_LAPS:
                            snail.is_finished = True
                            snail.finish_time = self.steps
        
        self.time_remaining -= 1
        self.steps += 1
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles: p.pos += p.vel; p.lifetime -= 1
        
        current_dist_to_waypoint = self.player.pos.distance_to(self.track_center[self.player.next_waypoint_idx])
        reward += (prev_dist_to_waypoint - current_dist_to_waypoint) * 0.1

        if self.player.is_finished:
            self.game_over = True
            self.game_won = True
            finish_times = sorted([s.finish_time for s in all_snails])
            player_rank = finish_times.index(self.player.finish_time)
            reward += [100.0, 50.0, 25.0][player_rank]
        
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            self.game_won = False
        
        self.score += reward
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        camera_offset = pygame.math.Vector2(self.WIDTH/2, self.HEIGHT/2) - self.player.pos
        self._render_track(camera_offset)
        self._render_snails(camera_offset)
        self._render_particles(camera_offset)
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self, offset):
        track_poly_points = [p + offset for p in self.track_outer_wall] + [p + offset for p in reversed(self.track_inner_wall)]
        if len(track_poly_points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, track_poly_points, self.COLOR_TRACK)
        
        pygame.draw.aalines(self.screen, self.COLOR_WALL, True, [p + offset for p in self.track_inner_wall], 1)
        pygame.draw.aalines(self.screen, self.COLOR_WALL, True, [p + offset for p in self.track_outer_wall], 1)
        
        finish_p1, finish_p2 = self.track_inner_wall[0] + offset, self.track_outer_wall[0] + offset
        for i in range(10):
            start = finish_p1.lerp(finish_p2, i / 10)
            end = finish_p1.lerp(finish_p2, (i + 1) / 10)
            color = self.COLOR_FINISH_1 if i % 2 == 0 else self.COLOR_FINISH_2
            pygame.draw.line(self.screen, color, start, end, 5)

    def _render_snails(self, offset):
        for snail in self.opponents:
            if not snail.is_finished:
                self._draw_snail(snail.pos + offset, snail.angle, snail.size, snail.color, snail.shell_color)
        player_screen_pos = pygame.math.Vector2(self.WIDTH/2, self.HEIGHT/2)
        self._draw_snail(player_screen_pos, self.player.angle, self.player.size, self.player.color, self.player.shell_color)
    
    def _draw_snail(self, pos, angle, size, color, shell_color):
        body_poly = [pos + pygame.math.Vector2(size * 0.8, 0).rotate(-angle + 180 + (i - 2) * 20) for i in range(5)]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in body_poly], color)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in body_poly], color)

        shell_pos = pos - pygame.math.Vector2(size * 0.2, 0).rotate(-angle)
        pygame.gfxdraw.filled_circle(self.screen, int(shell_pos.x), int(shell_pos.y), int(size), shell_color)
        pygame.gfxdraw.aacircle(self.screen, int(shell_pos.x), int(shell_pos.y), int(size), shell_color)
        for i in range(3):
            pygame.gfxdraw.aacircle(self.screen, int(shell_pos.x), int(shell_pos.y), int(size * (1 - i*0.3)), color)
        
        eye_base = pos + pygame.math.Vector2(size * 0.7, 0).rotate(-angle)
        eye1_pos = eye_base + pygame.math.Vector2(0, size * 0.4).rotate(-angle)
        eye2_pos = eye_base - pygame.math.Vector2(0, size * 0.4).rotate(-angle)
        for eye_pos in [eye1_pos, eye2_pos]:
            pygame.draw.circle(self.screen, (255,255,255), (int(eye_pos.x), int(eye_pos.y)), 3)
            pygame.draw.circle(self.screen, (0,0,0), (int(eye_pos.x), int(eye_pos.y)), 1)

    def _render_particles(self, offset):
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            pos = p.pos + offset
            size = max(1, int(3 * (p.lifetime / p.max_lifetime)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (p.color[0], p.color[1], p.color[2], alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(pos.x - size), int(pos.y - size)))

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        lap_text = f"LAP: {min(self.player.lap + 1, self.TOTAL_LAPS)}/{self.TOTAL_LAPS}"
        lap_surf = self.font_small.render(lap_text, True, self.COLOR_TEXT)
        self.screen.blit(lap_surf, (10, 10))
        
        boost_text = f"BOOSTS: {'● ' * self.boosts_left}{'○ ' * (self.MAX_BOOSTS - self.boosts_left)}"
        boost_surf = self.font_small.render(boost_text, True, self.COLOR_TEXT)
        self.screen.blit(boost_surf, (10, 30))

        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "TIME'S UP!"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lap": self.player.lap,
                "time_remaining": self.time_remaining, "boosts_left": self.boosts_left}

    def _point_segment_distance(self, p, a, b):
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        return p.distance_to(a + t * (b - a))
        
    def _line_segment_intersection(self, p1, p2, p3, p4):
        den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        if den == 0: return False
        t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / den
        u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / den
        return 0 < t < 1 and 0 < u < 1
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snail Racer")
    clock = pygame.time.Clock()
    action = [0, 0, 0]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        action = [0, 0, 0]
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        if keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        if keys[pygame.K_r]: obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()