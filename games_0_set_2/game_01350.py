import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# A simple 2D vector class to make vector math easier
class Vec2:
    def __init__(self, x, y=None):
        if isinstance(x, (tuple, list)):
            self.x, self.y = x[0], x[1]
        else:
            self.x, self.y = x, y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vec2(0, 0)
        return Vec2(self.x / scalar, self.y / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def length_sq(self):
        return self.x * self.x + self.y * self.y

    def length(self):
        len_sq = self.length_sq()
        return math.sqrt(len_sq) if len_sq > 0 else 0

    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        return self / l

    def to_tuple(self):
        return (self.x, self.y)

    def to_int_tuple(self):
        return (int(self.x), int(self.y))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    user_guide = (
        "Controls: Use arrow keys to draw lines. Guide the rider to the red finish line."
    )

    game_description = (
        "A physics-based puzzle game where you draw lines to create a path for a sled rider."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.W, self.H = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        # --- Constants ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (30, 35, 40)
        self.COLOR_TERRAIN = (100, 110, 120)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_LINE = (0, 150, 255)
        self.COLOR_START = (0, 255, 150)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)

        self.NUM_STAGES = 3
        self.MAX_STEPS = 2500
        self.NUM_PHYSICS_SUBSTEPS = 8
        self.LINE_SEGMENT_LENGTH = 20
        self.MAX_DRAWN_LINES = 150
        self.FINISH_LINE_X = self.W - 40

        # --- Game State (persistent across episodes for stage progression) ---
        self.current_stage = 0
        self.stage_just_completed = False

        # This is here to allow instantiation before the first reset
        self.terrain = []
        self.lines = []
        self.particles = []
        self.rider = {}
        self.last_draw_pos = Vec2(0,0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.stage_just_completed:
            self.current_stage = (self.current_stage + 1) % self.NUM_STAGES
            self.stage_just_completed = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.terrain = self._generate_terrain(self.current_stage)
        self.lines = []
        self.particles = []

        start_pos = Vec2(self.terrain[0][0])
        self.rider = {
            'pos': Vec2(start_pos.x, start_pos.y - 10),
            'vel': Vec2(1, 0),
            'radius': 7,
            'last_pos_x': start_pos.x
        }
        self.last_draw_pos = Vec2(start_pos.x, start_pos.y + 20)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._handle_drawing(movement)

        reward = 0
        for _ in range(self.NUM_PHYSICS_SUBSTEPS):
            if not self.game_over:
                reward += self._physics_update()
        
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Win or timeout
            if self.rider['pos'].x >= self.FINISH_LINE_X:
                win_reward = 10 + 100
                reward += win_reward
                self.score += win_reward
                self.stage_just_completed = True
                self.win_message = f"STAGE {self.current_stage + 1} COMPLETE!"
            else: # Timeout
                self.win_message = "OUT OF TIME"
            self.game_over = True
        elif self.game_over: # Crash
            crash_penalty = -10
            reward += crash_penalty
            self.score += crash_penalty
            self.stage_just_completed = False
            self.win_message = "CRASHED!"

        self.score = max(0, self.score)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_drawing(self, movement):
        if movement == 0: # No-op
            return

        angles = {
            1: -math.pi / 4,    # Up-Right
            2: math.pi / 4,     # Down-Right
            3: math.pi * 3 / 4, # Down-Left
            4: -math.pi * 3 / 4 # Up-Left
        }
        angle = angles.get(movement)
        if angle is None: return

        start_pos = self.last_draw_pos
        end_pos = start_pos + Vec2(math.cos(angle), math.sin(angle)) * self.LINE_SEGMENT_LENGTH
        
        end_pos.x = np.clip(end_pos.x, 0, self.W)
        end_pos.y = np.clip(end_pos.y, 0, self.H)

        self.lines.append((start_pos, end_pos))
        self.last_draw_pos = end_pos

        if len(self.lines) > self.MAX_DRAWN_LINES:
            self.lines.pop(0)

    def _physics_update(self):
        GRAVITY = 0.02
        FRICTION = 0.995
        RESTITUTION = 0.2

        self.rider['vel'].y += GRAVITY

        all_lines = self.terrain + self.lines
        for p1, p2 in all_lines:
            p1_vec = p1 if isinstance(p1, Vec2) else Vec2(p1)
            p2_vec = p2 if isinstance(p2, Vec2) else Vec2(p2)
            line_vec = p2_vec - p1_vec
            if line_vec.length_sq() == 0: continue

            point_vec = self.rider['pos'] - p1_vec
            t = line_vec.dot(point_vec) / line_vec.length_sq()
            t = np.clip(t, 0, 1)
            
            closest_point = p1_vec + line_vec * t
            dist_vec = self.rider['pos'] - closest_point
            
            if dist_vec.length_sq() < self.rider['radius'] ** 2:
                dist = dist_vec.length()
                penetration = self.rider['radius'] - dist
                normal = dist_vec.normalize()
                
                self.rider['pos'] += normal * penetration
                
                vn = self.rider['vel'].dot(normal)
                if vn < 0:
                    vt_vec = self.rider['vel'] - normal * vn
                    vn_vec = normal * (-vn * RESTITUTION)
                    self.rider['vel'] = (vt_vec * FRICTION) + vn_vec
                
                if self.np_random.random() < 0.5:
                    self._create_particles(self.rider['pos'], 1, normal * -0.5)

        self.rider['pos'] += self.rider['vel']

        if not (0 < self.rider['pos'].y < self.H):
            self.game_over = True
            self._create_particles(self.rider['pos'], 20, Vec2(0,0), (200,200,200))
            return 0

        delta_x = self.rider['pos'].x - self.rider['last_pos_x']
        reward = max(0, delta_x) * 0.1
        self.rider['last_pos_x'] = self.rider['pos'].x
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.rider['pos'].x >= self.FINISH_LINE_X

    def _generate_terrain(self, stage):
        points = []
        if stage == 0: # Gentle slope
            for i in range(0, self.W + 20, 20):
                points.append((i, 100 + i * 0.2))
        elif stage == 1: # A gap
            for i in range(0, 250, 20):
                points.append((i, 150 + i * 0.1))
            for i in range(400, self.W + 20, 20):
                points.append((i, 200 + i * 0.15))
        elif stage == 2: # Hills and valleys
            for i in range(0, self.W + 20, 15):
                y = 200 + 80 * math.sin(i / 100) + 30 * math.cos(i/40)
                points.append((i, y))
        
        return [(points[i], points[i+1]) for i in range(len(points)-1)]

    def _create_particles(self, pos, count, base_vel, color=None):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 1.5 + 0.5
            vel = Vec2(math.cos(angle), math.sin(angle)) * speed + base_vel
            self.particles.append({
                'pos': Vec2(pos.x, pos.y),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'size': self.np_random.random() * 2 + 1,
                'color': color or self.COLOR_RIDER
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(0, self.W, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.H))
        for i in range(0, self.H, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.W, i))

        if self.terrain:
            pygame.draw.line(self.screen, self.COLOR_START, (self.terrain[0][0][0], 0), (self.terrain[0][0][0], self.H), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.H), 3)

        for p1, p2 in self.terrain:
            pygame.draw.aaline(self.screen, self.COLOR_TERRAIN, p1, p2, 2)

        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, p1.to_tuple(), p2.to_tuple(), 2)
        
        if not self.game_over:
            pygame.gfxdraw.filled_circle(self.screen, int(self.last_draw_pos.x), int(self.last_draw_pos.y), 3, (*self.COLOR_LINE, 150))

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 25))))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)

        if self.rider:
            r_pos = self.rider['pos'].to_int_tuple()
            r_rad = self.rider['radius']
            angle = math.atan2(self.rider['vel'].y, self.rider['vel'].x)
            p1 = self.rider['pos'] + Vec2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * (r_rad + 2)
            p2 = self.rider['pos'] + Vec2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * (r_rad + 2)
            pygame.draw.aaline(self.screen, self.COLOR_RIDER, p1.to_int_tuple(), p2.to_int_tuple())
            pygame.gfxdraw.filled_circle(self.screen, r_pos[0], r_pos[1], r_rad, self.COLOR_RIDER)
            pygame.gfxdraw.aacircle(self.screen, r_pos[0], r_pos[1], r_rad, self.COLOR_RIDER)
            head_pos = (r_pos[0], r_pos[1] - r_rad)
            pygame.gfxdraw.filled_circle(self.screen, head_pos[0], head_pos[1], r_rad // 2, self.COLOR_RIDER)

    def _render_ui(self):
        rider_x = self.rider['pos'].x if self.rider else 0
        dist_text = self.font_ui.render(f"DIST: {int(rider_x):04d}m", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_TEXT)
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage + 1}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        steps_text = self.font_ui.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)

        self.screen.blit(dist_text, (10, 10))
        self.screen.blit(stage_text, (10, 30))
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))
        self.screen.blit(steps_text, (self.W - steps_text.get_width() - 10, 30))
        
        if self.game_over and self.win_message:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_FINISH if self.stage_just_completed else self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "rider_x": self.rider['pos'].x if self.rider else 0,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("MANUAL CONTROL".center(30))
    print("="*30)
    print("W: Draw Up-Left   | E: Draw Up-Right")
    print("S: Draw Down-Left | D: Draw Down-Right")
    print("R: Reset Environment")
    print("Q: Quit")
    print("Any other key: No-Op (advance physics)")
    print("="*30 + "\n")
    
    pygame.display.set_caption("Line Rider Gym")
    display_screen = pygame.display.set_mode((env.W, env.H))

    while True:
        action = [0, 0, 0]
        
        event_processed = False
        while not event_processed:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    exit()
                if event.key == pygame.K_r:
                    done = True
                
                key_map = {
                    pygame.K_e: 1, # Up-Right
                    pygame.K_d: 2, # Down-Right
                    pygame.K_s: 3, # Down-Left
                    pygame.K_w: 4, # Up-Left
                }
                if event.key in key_map:
                    action[0] = key_map[event.key]
                
                event_processed = True

        if done:
            print(f"Episode finished. Final Score: {info['score']}. Resetting...")
            obs, info = env.reset()
            done = False

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # The observation is (H, W, C). Pygame surface wants (W, H).
        # We need to transpose it back.
        obs_for_display = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_for_display)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()