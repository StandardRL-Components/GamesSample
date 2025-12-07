
# Generated: 2025-08-28T02:47:31.776457
# Source Brief: brief_01811.md
# Brief Index: 1811

        
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


# Helper class for a 2D vector
class Vec2D:
    def __init__(self, x, y=None):
        if y is None:
            self.x, self.y = x[0], x[1]
        else:
            self.x, self.y = x, y

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vec2D(0,0)
        return Vec2D(self.x / scalar, self.y / scalar)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec2D(0, 0)
        return Vec2D(self.x / l, self.y / l)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def rotate(self, angle_rad):
        x = self.x * math.cos(angle_rad) - self.y * math.sin(angle_rad)
        y = self.x * math.sin(angle_rad) + self.y * math.cos(angle_rad)
        return Vec2D(x, y)
    
    @property
    def tuple(self):
        return (int(self.x), int(self.y))

# Helper for particles
class Particle:
    def __init__(self, pos, vel, life, color, radius_start, radius_end):
        self.pos = Vec2D(pos.x, pos.y)
        self.vel = Vec2D(vel.x, vel.y)
        self.life = life
        self.max_life = max(1, life)
        self.color = color
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos += self.vel
        self.life -= 1

    def draw(self, surface, camera_offset):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            radius = self.radius_start * life_ratio + self.radius_end * (1 - life_ratio)
            draw_pos = self.pos - camera_offset
            pygame.draw.circle(surface, self.color, draw_pos.tuple, int(radius))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
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
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_TRACK = (80, 80, 90)
        self.COLOR_LINES = (200, 200, 220)
        self.COLOR_START = (0, 200, 50)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_OPPONENTS = [(50, 100, 255), (50, 255, 100), (255, 200, 50)]
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_BOOST_BAR = (0, 150, 255)
        self.COLOR_BOOST_PARTICLE = (255, 255, 255)
        self.COLOR_CRASH_PARTICLE = (255, 100, 0)

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game constants
        self.NUM_OPPONENTS = 3
        self.TOTAL_LAPS = 3
        self.MAX_STEPS = 10000
        self.TRACK_WIDTH = 60
        self.NUM_CHECKPOINTS = 16
        
        # Initialize state variables
        self.player_pos = Vec2D(0, 0)
        self.player_vel = Vec2D(0, 0)
        self.player_angle = 0.0
        self.player_lives = 0
        self.player_lap = 0
        self.player_checkpoint = 0
        self.player_boost = 0.0
        
        self.opponents = []
        
        self.track_centerline = []
        self.track_inner_bound = []
        self.track_outer_bound = []
        self.checkpoints = []

        self.particles = []
        self.game_over_message = ""
        self.start_countdown = 0
        
        self.reset()
        
        # self.validate_implementation() # For internal validation

    def _generate_track(self):
        center = Vec2D(self.WIDTH * self.np_random.uniform(0.8, 1.2), self.HEIGHT * self.np_random.uniform(0.8, 1.2))
        radius = min(self.WIDTH, self.HEIGHT) * 1.5
        points = []
        for i in range(self.NUM_CHECKPOINTS):
            angle = (i / self.NUM_CHECKPOINTS) * 2 * math.pi
            r = radius * (0.4 + self.np_random.uniform(-0.15, 0.15))
            x = center.x + r * math.cos(angle)
            y = center.y + r * math.sin(angle)
            points.append(Vec2D(x, y))

        self.track_centerline = []
        for i in range(len(points)):
            p0 = points[(i - 1 + len(points)) % len(points)]
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            p3 = points[(i + 2) % len(points)]
            for t in np.linspace(0, 1, 20, endpoint=False):
                x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t**2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t**3)
                y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t**2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t**3)
                self.track_centerline.append(Vec2D(x, y))
        
        self.track_inner_bound, self.track_outer_bound = [], []
        for i, p in enumerate(self.track_centerline):
            prev_p = self.track_centerline[i - 1]
            tangent = (p - prev_p).normalize()
            normal = Vec2D(-tangent.y, tangent.x)
            self.track_inner_bound.append(p - normal * self.TRACK_WIDTH)
            self.track_outer_bound.append(p + normal * self.TRACK_WIDTH)

        self.checkpoints = []
        segment_len = len(self.track_centerline) // self.NUM_CHECKPOINTS
        for i in range(self.NUM_CHECKPOINTS):
            self.checkpoints.append(self.track_centerline[i * segment_len])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_track()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.start_countdown = 90 # 3 seconds at 30fps

        start_pos = self.track_centerline[0]
        start_dir = (self.track_centerline[1] - self.track_centerline[0]).normalize()

        self.player_pos = Vec2D(start_pos.x, start_pos.y)
        self.player_vel = Vec2D(0, 0)
        self.player_angle = math.atan2(start_dir.y, start_dir.x)
        self.player_lives = 3
        self.player_lap = 0
        self.player_checkpoint = 0
        self.player_boost = 100.0
        
        self.opponents = []
        for i in range(self.NUM_OPPONENTS):
            offset = (i + 1) * 20
            path_idx = (len(self.track_centerline) - offset) % len(self.track_centerline)
            pos = self.track_centerline[path_idx]
            self.opponents.append({
                "pos": pos,
                "angle": self.player_angle,
                "speed": 2.5 + self.np_random.uniform(-0.2, 0.2),
                "lap": 0,
                "checkpoint": 0,
                "path_index": path_idx
            })
        
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.start_countdown > 0:
            self.start_countdown -= 1
            action = np.array([0,0,0])

        if self.game_over:
            # Keep particles fading out but no other logic
            self.particles = [p for p in self.particles if p.life > 0]
            for p in self.particles: p.update()
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward += self._update_player(movement, space_held, shift_held)
        self._update_opponents()
        
        lap_completed, crashed, overtook_opponent = self._update_game_state()
        
        if lap_completed: reward += 5
        if crashed: reward -= 2
        if overtook_opponent: reward += 2
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over: # Max steps reached
            self.game_over = True
            self.game_over_message = "Game Over: Time Limit"

        if self.game_over:
            if "Win" in self.game_over_message:
                reward += 100
            else:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held, shift_held):
        ACCEL, BRAKE, TURN_SPEED = 0.2, 0.3, 0.05
        FRICTION, DRIFT_FRICTION, DRIFT_TURN_MULT = 0.96, 0.92, 1.5
        MAX_SPEED, BOOST_SPEED = 5, 8
        
        forward_vec = Vec2D(math.cos(self.player_angle), math.sin(self.player_angle))
        
        speed = self.player_vel.length()
        turn_factor = min(1, speed / 3.0)
        if movement == 3: self.player_angle -= TURN_SPEED * turn_factor * (DRIFT_TURN_MULT if shift_held else 1)
        if movement == 4: self.player_angle += TURN_SPEED * turn_factor * (DRIFT_TURN_MULT if shift_held else 1)
        
        if movement == 1: self.player_vel += forward_vec * ACCEL
        if movement == 2: self.player_vel -= self.player_vel.normalize() * BRAKE if speed > BRAKE else self.player_vel

        if shift_held and speed > 2.0:
            if self.steps % 2 == 0:
                p_vel = self.player_vel * -0.1
                self.particles.append(Particle(self.player_pos, p_vel, 20, (100,100,100), 3, 0))

        if space_held and self.player_boost > 0:
            self.player_boost -= 2.0
            max_speed = BOOST_SPEED
            if self.steps % 2 == 0:
                p_pos = self.player_pos - forward_vec * 10
                p_vel = (self.player_vel * -0.2) + Vec2D(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
                self.particles.append(Particle(p_pos, p_vel, 30, self.COLOR_BOOST_PARTICLE, 4, 0))
        else:
            self.player_boost = min(100, self.player_boost + 0.3)
            max_speed = MAX_SPEED
        
        self.player_vel *= DRIFT_FRICTION if shift_held and speed > 2.0 else FRICTION
        if self.player_vel.length() > max_speed:
            self.player_vel = self.player_vel.normalize() * max_speed
        
        self.player_pos += self.player_vel
        
        forward_movement = self.player_vel.dot(forward_vec)
        return max(0, forward_movement * 0.02) + 0.1 if movement == 1 else 0

    def _update_opponents(self):
        base_speed = 2.5 + self.player_lap * 0.05
        for opp in self.opponents:
            target_index = (opp["path_index"] + 5) % len(self.track_centerline)
            target_pos = self.track_centerline[target_index]
            
            direction = (target_pos - opp["pos"]).normalize()
            opp["pos"] += direction * opp["speed"]
            opp["angle"] = math.atan2(direction.y, direction.x)

            if (target_pos - opp["pos"]).length() < 10:
                opp["path_index"] = target_index
            
            next_checkpoint_idx = (opp["checkpoint"] + 1) % self.NUM_CHECKPOINTS
            if (self.checkpoints[next_checkpoint_idx] - opp["pos"]).length() < self.TRACK_WIDTH:
                opp["checkpoint"] = next_checkpoint_idx
                if opp["checkpoint"] == 0:
                    opp["lap"] += 1
                    opp["speed"] = base_speed + self.np_random.uniform(-0.2, 0.2)

    def _update_game_state(self):
        lap_completed, crashed, overtook = False, False, False
        
        min_dist_sq = float('inf')
        for p in self.track_centerline:
            dist_sq = (self.player_pos.x - p.x)**2 + (self.player_pos.y - p.y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        if min_dist_sq > self.TRACK_WIDTH**2:
            crashed = True
            self.player_lives -= 1
            self.player_vel = Vec2D(0, 0)
            safe_idx = (self.player_checkpoint * (len(self.track_centerline) // self.NUM_CHECKPOINTS)) % len(self.track_centerline)
            self.player_pos = Vec2D(self.track_centerline[safe_idx].x, self.track_centerline[safe_idx].y)
            for _ in range(30):
                angle = self.np_random.uniform(0, 2*math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = Vec2D(math.cos(angle) * speed, math.sin(angle) * speed)
                self.particles.append(Particle(self.player_pos, vel, 40, self.COLOR_CRASH_PARTICLE, 5, 0))

        prev_rank = self._get_player_rank()
        next_checkpoint_idx = (self.player_checkpoint + 1) % self.NUM_CHECKPOINTS
        if (self.checkpoints[next_checkpoint_idx] - self.player_pos).length() < self.TRACK_WIDTH * 1.5:
            self.player_checkpoint = next_checkpoint_idx
            if self.player_checkpoint == 0:
                self.player_lap += 1
                lap_completed = True

        if self._get_player_rank() < prev_rank: overtook = True

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()
            
        return lap_completed, crashed, overtook

    def _get_player_rank(self):
        player_progress = self.player_lap + self.player_checkpoint / self.NUM_CHECKPOINTS
        rank = 1
        for opp in self.opponents:
            opp_progress = opp["lap"] + opp["checkpoint"] / self.NUM_CHECKPOINTS
            if opp_progress > player_progress:
                rank += 1
        return rank

    def _check_termination(self):
        if self.player_lap >= self.TOTAL_LAPS:
            self.game_over = True
            self.game_over_message = f"You Win! Rank: {self._get_player_rank()}"
        elif self.player_lives <= 0:
            self.game_over = True
            self.game_over_message = "Game Over: Crashed!"
        elif self.steps >= self.MAX_STEPS:
            return True # Terminate but let step() set the message
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        camera_offset = self.player_pos - Vec2D(self.WIDTH / 2, self.HEIGHT / 2)
        self._render_game(camera_offset)
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, camera_offset):
        track_pts_screen = [(p - camera_offset).tuple for p in self.track_centerline]
        if len(track_pts_screen) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRACK, True, track_pts_screen, self.TRACK_WIDTH * 2)

        outer_b = [(p - camera_offset).tuple for p in self.track_outer_bound]
        inner_b = [(p - camera_offset).tuple for p in self.track_inner_bound]
        if len(outer_b) > 1 and len(inner_b) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_LINES, True, outer_b)
            pygame.draw.aalines(self.screen, self.COLOR_LINES, True, inner_b)

        start_p1, start_p2 = (self.track_inner_bound[0] - camera_offset), (self.track_outer_bound[0] - camera_offset)
        pygame.draw.line(self.screen, self.COLOR_START, start_p1.tuple, start_p2.tuple, 5)

        for p in self.particles: p.draw(self.screen, camera_offset)
        for i, opp in enumerate(self.opponents): self._render_kart(opp["pos"] - camera_offset, opp["angle"], self.COLOR_OPPONENTS[i])
        self._render_kart(Vec2D(self.WIDTH/2, self.HEIGHT/2), self.player_angle, self.COLOR_PLAYER, is_player=True)

    def _render_kart(self, pos, angle, color, is_player=False):
        w, h = 10, 20
        points = [Vec2D(-w, -h), Vec2D(w, -h), Vec2D(w, h), Vec2D(-w, h)]
        rotated_points = [p.rotate(angle - math.pi/2) + pos for p in points]
        
        if is_player:
            glow_points = [p.rotate(angle - math.pi/2) + pos for p in [
                Vec2D(-w-3, -h-3), Vec2D(w+3, -h-3), Vec2D(w+3, h+3), Vec2D(-w-3, h+3)]]
            pygame.gfxdraw.filled_polygon(self.screen, [p.tuple for p in glow_points], (*color, 60))

        pygame.gfxdraw.filled_polygon(self.screen, [p.tuple for p in rotated_points], color)
        pygame.gfxdraw.aapolygon(self.screen, [p.tuple for p in rotated_points], color)

    def _render_ui(self):
        lap_text = self.font_main.render(f"Lap: {min(self.player_lap + 1, self.TOTAL_LAPS)}/{self.TOTAL_LAPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))

        rank = self._get_player_rank()
        rank_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(rank, f"{rank}th")
        pos_text = self.font_main.render(f"Pos: {rank_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(pos_text, (self.WIDTH - pos_text.get_width() - 10, 10))

        lives_text = self.font_main.render(f"Lives: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (10, 40))

        bar_x, bar_y, bar_w, bar_h = self.WIDTH - 160, self.HEIGHT - 30, 150, 20
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        fill_w = bar_w * (self.player_boost / 100.0)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 2)
        boost_text = self.font_main.render("Boost", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (bar_x - boost_text.get_width() - 10, bar_y - 2))

        if self.game_over:
            msg_surf = self.font_big.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)
        elif self.start_countdown > 0:
            count = math.ceil(self.start_countdown / 30)
            msg = str(count) if count > 0 else "GO!"
            msg_surf = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    running = True
    terminated = False
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")

    while running:
        action = np.array([0, 0, 0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

        if terminated:
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()