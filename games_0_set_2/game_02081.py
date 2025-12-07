
# Generated: 2025-08-28T03:37:38.290889
# Source Brief: brief_02081.md
# Brief Index: 2081

        
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


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a line rider-style game.

    The agent places track segments one by one, and a physics-based rider attempts
    to navigate the created track to reach a finish line. The game is turn-based;
    each action places one segment and simulates the rider's movement along it.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use actions to place track segments: ↑/↓ for curves, Space for a boost. Default is a straight segment."
    )

    game_description = (
        "Draw a track for your rider to reach the finish line! Place straight, curved, or boost segments to guide them safely and quickly before time runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH = 640
        self.HEIGHT = 400
        self.FINISH_X = 2000

        # Colors
        self.COLOR_BG_SKY = (135, 206, 235)
        self.COLOR_BG_GROUND = (139, 69, 19)
        self.COLOR_TRACK = (20, 20, 20)
        self.COLOR_BOOST_TRACK = (255, 215, 0)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_START = (0, 255, 0, 150)
        self.COLOR_FINISH = (255, 0, 0, 150)
        self.COLOR_TEXT = (20, 20, 20)
        self.COLOR_TEXT_SHADOW = (150, 150, 150)

        # Game parameters
        self.GRAVITY = pygame.math.Vector2(0, 0.08)
        self.SEGMENT_LENGTH = 60
        self.CURVE_ANGLE = 20  # degrees
        self.FALL_THRESHOLD = 25
        self.MAX_SIM_STEPS_PER_SEGMENT = 100
        self.MAX_EPISODE_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 30
        self.FPS = 30  # Simulation FPS

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.track_nodes = []
        self.track_segments = []
        self.camera_offset_x = 0.0
        self.particles = []
        self.current_segment_type = "STRAIGHT"

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS

        # Initial track setup
        start_pos = pygame.math.Vector2(100, self.HEIGHT / 2)
        self.track_nodes = [
            start_pos - pygame.math.Vector2(self.SEGMENT_LENGTH, 0),
            start_pos
        ]
        self.track_segments = [
            {'type': 'straight', 'color': self.COLOR_TRACK}
        ]

        # Rider setup
        self.rider_pos = start_pos.copy()
        self.rider_vel = pygame.math.Vector2(2.5, 0)

        # Camera and UI
        self.camera_offset_x = self.rider_pos.x - self.WIDTH / 4
        self.particles = []
        self.current_segment_type = "STRAIGHT"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Decode Action
        movement, space_held, _ = action
        segment_type = "STRAIGHT"
        if space_held == 1:
            segment_type = "BOOST"
        elif movement == 1:  # Up
            segment_type = "CURVE_UP"
        elif movement == 2:  # Down
            segment_type = "CURVE_DOWN"
        self.current_segment_type = segment_type.replace("_", " ")

        # 2. Add new track segment
        last_node = self.track_nodes[-1]
        prev_node = self.track_nodes[-2]
        
        vec_prev_to_last = last_node - prev_node
        last_angle = math.degrees(math.atan2(-vec_prev_to_last.y, vec_prev_to_last.x)) if vec_prev_to_last.length_squared() > 0 else 0

        new_angle = last_angle
        if segment_type == "CURVE_UP":
            new_angle += self.CURVE_ANGLE
        elif segment_type == "CURVE_DOWN":
            new_angle -= self.CURVE_ANGLE
        
        new_angle = max(-75, min(75, new_angle))

        move_vec = pygame.math.Vector2(self.SEGMENT_LENGTH, 0).rotate(new_angle)
        new_node = last_node + move_vec
        
        self.track_nodes.append(new_node)
        segment_info = {
            'type': segment_type,
            'color': self.COLOR_BOOST_TRACK if segment_type == "BOOST" else self.COLOR_TRACK
        }
        self.track_segments.append(segment_info)

        # 3. Simulate rider on the new segment
        fell_off, sim_duration = self._simulate_rider_on_segment(last_node, new_node, segment_type)
        self.time_remaining -= sim_duration
        
        # 4. Calculate reward and termination
        reward = 0
        terminated = False
        
        if fell_off:
            reward = -5
            terminated = True
            # Sound: rider_fall.wav
        else:
            reward = 0.1  # Survived the segment
            self.rider_pos = new_node.copy() # Snap rider to the end
        
        if self.rider_pos.x >= self.FINISH_X:
            terminated = True
            reward += 10  # Base reward for finishing
            if self.time_remaining > (self.TIME_LIMIT_SECONDS / 2) * self.FPS:
                reward += 50  # Bonus for finishing fast
            # Sound: win_fanfare.wav
        elif self.time_remaining <= 0 and not terminated:
            terminated = True
            reward = -2
        elif self.steps >= self.MAX_EPISODE_STEPS - 1 and not terminated:
            terminated = True
            reward = -1

        self.game_over = terminated
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _simulate_rider_on_segment(self, start_node, end_node, segment_type):
        sim_steps = 0
        segment_vec = end_node - start_node
        segment_dir = segment_vec.normalize() if segment_vec.length() > 0 else pygame.math.Vector2(1, 0)

        for i in range(self.MAX_SIM_STEPS_PER_SEGMENT):
            sim_steps += 1
            
            self.rider_vel += self.GRAVITY
            self.rider_pos += self.rider_vel

            self.rider_vel.rotate_ip(self.rider_vel.angle_to(segment_dir) * 0.2)
            
            if segment_type == "BOOST":
                self.rider_vel *= 1.02
                # Sound: boost.wav
                if i % 3 == 0:
                    self.particles.append(Particle(self.rider_pos, self.COLOR_BOOST_TRACK, life=10))

            dist_to_segment = self._point_segment_distance(self.rider_pos, start_node, end_node)
            if dist_to_segment > self.FALL_THRESHOLD:
                for _ in range(10):
                    self.particles.append(Particle(self.rider_pos, self.COLOR_RIDER, life=20, gravity=True))
                return True, sim_steps

            proj = (self.rider_pos - start_node).dot(segment_dir)
            if proj >= segment_vec.length():
                return False, sim_steps

        return False, sim_steps

    def _point_segment_distance(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_len_sq = ab.length_squared()
        if ab_len_sq == 0:
            return ap.length()
        
        dot = ap.dot(ab)
        t = max(0, min(1, dot / ab_len_sq))
        projection = a + t * ab
        return (p - projection).length()

    def _get_observation(self):
        target_cam_x = self.rider_pos.x - self.WIDTH / 4
        self.camera_offset_x += (target_cam_x - self.camera_offset_x) * 0.1
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_SKY[0] * (1 - interp) + self.COLOR_BG_GROUND[0] * interp),
                int(self.COLOR_BG_SKY[1] * (1 - interp) + self.COLOR_BG_GROUND[1] * interp),
                int(self.COLOR_BG_SKY[2] * (1 - interp) + self.COLOR_BG_GROUND[2] * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        start_x_on_screen = 100 - self.camera_offset_x
        pygame.draw.line(self.screen, self.COLOR_START, (start_x_on_screen, 0), (start_x_on_screen, self.HEIGHT), 3)
        finish_x_on_screen = self.FINISH_X - self.camera_offset_x
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x_on_screen, 0), (finish_x_on_screen, self.HEIGHT), 3)

        if len(self.track_nodes) > 1:
            for i in range(len(self.track_segments)):
                p1 = self.track_nodes[i]
                p2 = self.track_nodes[i+1]
                color = self.track_segments[i]['color']
                start_pos = (int(p1.x - self.camera_offset_x), int(p1.y))
                end_pos = (int(p2.x - self.camera_offset_x), int(p2.y))
                pygame.draw.aaline(self.screen, color, start_pos, end_pos, True)
                pygame.draw.aaline(self.screen, color, (start_pos[0], start_pos[1]+1), (end_pos[0], end_pos[1]+1), True)

        for p in self.particles[:]:
            p.update()
            if not p.is_alive():
                self.particles.remove(p)
            else:
                p.draw(self.screen, self.camera_offset_x)

        rider_screen_pos = (int(self.rider_pos.x - self.camera_offset_x), int(self.rider_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], 8, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], 8, self.COLOR_RIDER)

    def _render_ui(self):
        time_sec = max(0, self.time_remaining // self.FPS)
        timer_text = f"TIME: {time_sec:02d}"
        self._draw_text(timer_text, (self.WIDTH - 10, 10), self.font_large, self.COLOR_TEXT, "topright")

        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (10, 10), self.font_large, self.COLOR_TEXT, "topleft")

        segment_text = f"SEGMENT: {self.current_segment_type}"
        self._draw_text(segment_text, (10, self.HEIGHT - 10), self.font_small, self.COLOR_TEXT, "bottomleft")

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(**{align: pos})
        
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        shadow_rect = shadow_surface.get_rect(**{align: (pos[0] + 1, pos[1] + 1)})
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "rider_x_pos": self.rider_pos.x,
        }

    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, pos, color, life, gravity=False):
        self.pos = pos.copy()
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-2, 0))
        self.color = color
        self.life = life
        self.max_life = life
        self.use_gravity = gravity

    def update(self):
        if self.use_gravity:
            self.vel.y += 0.1
        self.pos += self.vel
        self.life -= 1

    def is_alive(self):
        return self.life > 0

    def draw(self, surface, camera_offset_x):
        alpha = int(255 * (self.life / self.max_life))
        size = max(1, int(4 * (self.life / self.max_life)))
        
        draw_pos = (int(self.pos.x - camera_offset_x), int(self.pos.y))
        
        temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        temp_surf.fill((*self.color, alpha))
        surface.blit(temp_surf, (draw_pos[0] - size//2, draw_pos[1] - size//2))

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym Environment")
    
    running = True
    action_to_take = None

    print("\n" + "="*30)
    print("Line Rider Gym - Manual Control")
    print(env.user_guide)
    print("Press 'R' to reset.")
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    action_to_take = None
                if not done:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                        action_to_take = action
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                        action_to_take = action
                    elif event.key == pygame.K_SPACE:
                        action[1] = 1
                        action_to_take = action
                    elif event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN]:
                        action[0] = 0
                        action_to_take = action

        if action_to_take and not done:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            done = terminated
            action_to_take = None
        
        frame_to_show = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display, frame_to_show)
        pygame.display.flip()

        env.clock.tick(30)

    env.close()