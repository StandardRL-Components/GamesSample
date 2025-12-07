
# Generated: 2025-08-27T16:13:57.446984
# Source Brief: brief_01160.md
# Brief Index: 1160

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to draw lines from the rider. "
        "Space draws a long ramp down. Shift draws a short ramp up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Draw lines to guide the sledder "
        "from the start to the finish, collecting checkpoints along the way."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (44, 62, 80)        # Dark Slate Blue
    COLOR_GRID = (52, 73, 94)      # Wet Asphalt
    COLOR_LINE = (236, 240, 241)   # Clouds (almost white)
    COLOR_RIDER = (255, 255, 255)  # White
    COLOR_SLED = (231, 76, 60)     # Alizarin Red
    COLOR_START = (46, 204, 113)   # Emerald Green
    COLOR_FINISH = (192, 57, 43)   # Pomegranate Red
    COLOR_CHECKPOINT = (52, 152, 219) # Peter River Blue
    COLOR_UI_TEXT = (236, 240, 241) # Clouds

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 1000
    TIME_LIMIT_SECONDS = 120.0
    TIME_PER_STEP = 0.5 # Seconds added to timer per step
    NUM_CHECKPOINTS = 10

    # Rider Physics
    GRAVITY = 0.1
    FRICTION = 0.01
    RIDER_RADIUS = 6
    PHYSICS_SUBSTEPS = 10

    # Line Drawing
    LINE_LENGTH_DIR = 40
    LINE_LENGTH_SPACE = 80
    LINE_LENGTH_SHIFT = 30


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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 40, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        # State variables are initialized in reset()
        self.rider_pos = None
        self.rider_vel = None
        self.lines = None
        self.particles = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.start_pos = None
        self.finish_pos = None
        self.checkpoints = None
        self.visited_checkpoints = None
        self.termination_reason = ""

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.start_pos = np.array([50.0, 100.0])
        self.finish_pos = np.array([self.SCREEN_WIDTH - 50.0, self.SCREEN_HEIGHT - 50.0])

        self.rider_pos = self.start_pos.copy()
        self.rider_vel = np.array([1.0, 0.0]) # Small initial push

        # Initial platform
        self.lines = [
            (np.array([self.start_pos[0] - 20, self.start_pos[1] + self.RIDER_RADIUS]),
             np.array([self.start_pos[0] + 20, self.start_pos[1] + self.RIDER_RADIUS]))
        ]
        self.particles = []

        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.termination_reason = ""
        
        self._setup_checkpoints()

        return self._get_observation(), self._get_info()

    def _setup_checkpoints(self):
        self.checkpoints = []
        self.visited_checkpoints = set()
        for i in range(1, self.NUM_CHECKPOINTS + 1):
            fraction = i / (self.NUM_CHECKPOINTS + 1)
            pos = self.start_pos + (self.finish_pos - self.start_pos) * fraction
            pos[1] += math.sin(fraction * math.pi * 2) * 50
            self.checkpoints.append(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        self.steps += 1
        self.timer += self.TIME_PER_STEP
        reward = -0.01  # Step penalty

        self._handle_action(movement, space_held, shift_held)
        
        old_rider_x = self.rider_pos[0]
        self._simulate_physics()
        
        # Reward for forward progress
        progress = self.rider_pos[0] - old_rider_x
        reward += progress * 0.1

        # Reward for checkpoints
        for i, cp_pos in enumerate(self.checkpoints):
            if i not in self.visited_checkpoints:
                if np.linalg.norm(self.rider_pos - cp_pos) < 20:
                    self.visited_checkpoints.add(i)
                    reward += 10.0
                    self.score += 10
                    # Sound placeholder: checkpoint_get.wav

        terminated = self._check_termination()
        if terminated:
            if "FINISH" in self.termination_reason:
                reward += 100.0
                self.score += 100
                # Sound placeholder: victory.wav
            elif "FELL" in self.termination_reason:
                reward -= 50.0
                # Sound placeholder: fall.wav
            elif "TIME" in self.termination_reason:
                reward -= 25.0
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.game_over: # Already terminated in a previous check
            return True
            
        if np.linalg.norm(self.rider_pos - self.finish_pos) < 20:
            self.game_over = True
            self.termination_reason = "FINISH!"
            return True
        elif self.rider_pos[1] > self.SCREEN_HEIGHT + self.RIDER_RADIUS * 2 or self.rider_pos[0] < -20 or self.rider_pos[0] > self.SCREEN_WIDTH + 20:
            self.game_over = True
            self.termination_reason = "FELL OFF"
            return True
        elif self.timer >= self.TIME_LIMIT_SECONDS:
            self.game_over = True
            self.termination_reason = "TIME UP"
            return True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.termination_reason = "MAX STEPS"
            return True
        return False

    def _handle_action(self, movement, space_held, shift_held):
        start_point = self.rider_pos.copy()
        end_point = start_point.copy()

        if space_held:
            end_point += np.array([self.LINE_LENGTH_SPACE * 0.9, self.LINE_LENGTH_SPACE * 0.45])
            # Sound placeholder: draw_long.wav
        elif shift_held:
            end_point += np.array([self.LINE_LENGTH_SHIFT * 0.9, -self.LINE_LENGTH_SHIFT * 0.45])
            # Sound placeholder: draw_short.wav
        else:
            if movement == 1: end_point[1] -= self.LINE_LENGTH_DIR
            elif movement == 2: end_point[1] += self.LINE_LENGTH_DIR
            elif movement == 3: end_point[0] -= self.LINE_LENGTH_DIR
            elif movement == 4: end_point[0] += self.LINE_LENGTH_DIR
            # Sound placeholder: draw_normal.wav

        end_point[0] = np.clip(end_point[0], 0, self.SCREEN_WIDTH)
        end_point[1] = np.clip(end_point[1], 0, self.SCREEN_HEIGHT)
        
        if np.linalg.norm(start_point - end_point) > 1:
            self.lines.append((start_point, end_point))

    def _simulate_physics(self):
        for _ in range(self.PHYSICS_SUBSTEPS):
            vel_before_collision = self.rider_vel.copy()
            self.rider_vel[1] += self.GRAVITY / self.PHYSICS_SUBSTEPS
            self.rider_pos += self.rider_vel / self.PHYSICS_SUBSTEPS
            collided = False
            for p1, p2 in self.lines:
                d_line = p2 - p1
                len_sq = np.dot(d_line, d_line)
                if len_sq == 0: continue
                t = max(0, min(1, np.dot(self.rider_pos - p1, d_line) / len_sq))
                closest_point = p1 + t * d_line
                dist_vec = self.rider_pos - closest_point
                dist = np.linalg.norm(dist_vec)
                if dist < self.RIDER_RADIUS:
                    collided = True
                    penetration = self.RIDER_RADIUS - dist
                    self.rider_pos += (dist_vec / dist) * penetration
                    line_vec = d_line / np.sqrt(len_sq)
                    normal = np.array([-line_vec[1], line_vec[0]])
                    if np.dot(normal, dist_vec) < 0: normal *= -1
                    v_n_scalar = np.dot(self.rider_vel, normal)
                    v_n = v_n_scalar * normal
                    v_t = self.rider_vel - v_n
                    if v_n_scalar < 0:
                        self.rider_vel = v_t * (1.0 - self.FRICTION)
                        # Sound placeholder: slide_loop.wav
                    break
            if collided:
                vel_change = np.linalg.norm(self.rider_vel - vel_before_collision)
                if vel_change > 0.1:
                    for _ in range(2):
                        particle_vel = (vel_before_collision - self.rider_vel) * self.np_random.uniform(0.5, 1.5)
                        particle_vel += self.np_random.uniform(-0.5, 0.5, size=2)
                        self.particles.append({"pos": self.rider_pos.copy(), "vel": particle_vel, "life": self.np_random.integers(15, 30), "size": self.np_random.uniform(2, 4)})
        self._update_particles()

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] > 0 and p['size'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos[0], 0), (self.start_pos[0], self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_pos[0], 0), (self.finish_pos[0], self.SCREEN_HEIGHT), 3)
        for i, cp_pos in enumerate(self.checkpoints):
            color = self.COLOR_START if i in self.visited_checkpoints else self.COLOR_CHECKPOINT
            pygame.gfxdraw.filled_circle(self.screen, int(cp_pos[0]), int(cp_pos[1]), 8, color)
            pygame.gfxdraw.aacircle(self.screen, int(cp_pos[0]), int(cp_pos[1]), 8, color)
        for p1, p2 in self.lines: pygame.draw.line(self.screen, self.COLOR_LINE, p1.astype(int), p2.astype(int), 4)
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*self.COLOR_RIDER, alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, p['pos'] - p['size'])
        pos_i = self.rider_pos.astype(int)
        sled_points = [(pos_i[0] - 8, pos_i[1] + 3), (pos_i[0] + 8, pos_i[1] + 3), (pos_i[0], pos_i[1] - 8)]
        pygame.draw.polygon(self.screen, self.COLOR_SLED, sled_points)
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1]-2, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1]-2, self.RIDER_RADIUS, self.COLOR_RIDER)
        
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        time_left = max(0, self.TIME_LIMIT_SECONDS - self.timer)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        if self.game_over:
            msg_text = self.font_msg.render(self.termination_reason, True, self.COLOR_UI_TEXT)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "rider_pos": self.rider_pos.tolist(),
            "rider_vel": self.rider_vel.tolist(),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()