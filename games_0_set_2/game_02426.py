import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Space to draw a line. "
        "Release Space to place it. Hold Shift to give the rider a forward boost."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines in real-time to guide the sledder from the "
        "green start line to the red finish line, passing through checkpoints."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000

    # Colors
    COLOR_BG = (220, 220, 230)
    COLOR_GRID = (200, 200, 210)
    COLOR_RIDER = (255, 255, 255)
    COLOR_SLED = (200, 30, 30)
    COLOR_TRACK = (20, 20, 80)
    COLOR_TRACK_PREVIEW = (100, 100, 150, 150)
    COLOR_START = (0, 200, 0)
    COLOR_FINISH = (255, 0, 0)
    COLOR_CHECKPOINT = (255, 200, 0)
    COLOR_TEXT = (10, 10, 40)
    COLOR_CURSOR = (255, 80, 0)

    # Physics
    GRAVITY = 0.3
    FRICTION = 0.99
    BOOST_FORCE = 0.2
    RIDER_RADIUS = 7
    CURSOR_SPEED = 8
    MIN_LINE_LENGTH = 5
    STUCK_VEL_THRESHOLD = 0.1
    STUCK_FRAMES_LIMIT = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        # This will call reset() and initialize np_random
        # No need to call it explicitly here, as it's called by __init__
        # self.reset()
        # self.validate_implementation() is called after the first reset in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        start_y = self.HEIGHT * 0.2
        self.start_pos = [self.WIDTH * 0.05, start_y]
        self.finish_x = self.WIDTH * 0.95

        self.rider_pos = np.array(self.start_pos, dtype=float)
        self.rider_vel = np.array([0.0, 0.0], dtype=float)
        self.rider_angle = 0.0
        self.on_ground = False

        self.lines = [([0, start_y + 20], [self.start_pos[0] + 30, start_y + 20])]
        self.checkpoints = []
        num_checkpoints = 3
        for i in range(num_checkpoints):
            x = self.WIDTH * (0.2 + 0.7 * (i + 1) / (num_checkpoints + 1))
            self.checkpoints.append({"x": x, "passed": False})

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.is_drawing = False
        self.line_start_pos = None
        
        self.prev_space_held = False
        self.stuck_frames = 0
        self.last_rider_x = self.rider_pos[0]
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)
        self._update_physics(shift_held)

        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
             self.score -= 10 # Penalty for running out of time
             terminated = True
        
        self.steps += 1
        
        self.prev_space_held = space_held
        self.last_rider_x = self.rider_pos[0]

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Line drawing
        space_pressed = space_held and not self.prev_space_held
        space_released = not space_held and self.prev_space_held

        if space_pressed and not self.is_drawing:
            self.is_drawing = True
            self.line_start_pos = self.cursor_pos.copy()
        
        if space_released and self.is_drawing:
            p1 = self.line_start_pos
            p2 = self.cursor_pos
            if np.linalg.norm(p2 - p1) > self.MIN_LINE_LENGTH:
                # Ensure lines are within a reasonable drawing area
                if min(p1[1], p2[1]) > 20 and max(p1[1], p2[1]) < self.HEIGHT - 20:
                    self.lines.append((list(p1), list(p2)))
                    # sfx: line_placed.wav
            self.is_drawing = False

    def _update_physics(self, shift_held):
        if self.game_over:
            return

        # 1. Apply forces
        self.rider_vel[1] += self.GRAVITY
        if shift_held and self.on_ground:
            # sfx: boost.wav
            boost_vec = np.array([math.cos(self.rider_angle), math.sin(self.rider_angle)])
            self.rider_vel += boost_vec * self.BOOST_FORCE

        # 2. Update position
        self.rider_pos += self.rider_vel
        self.on_ground = False

        # 3. Collision detection and response
        for p1, p2 in self.lines:
            p1 = np.array(p1)
            p2 = np.array(p2)
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq == 0: continue

            rider_to_p1 = self.rider_pos - p1
            t = np.dot(rider_to_p1, line_vec) / line_len_sq
            t_clamped = np.clip(t, 0, 1)

            closest_point = p1 + t_clamped * line_vec
            dist_vec = self.rider_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.RIDER_RADIUS**2:
                self.on_ground = True
                dist = math.sqrt(dist_sq)
                penetration = self.RIDER_RADIUS - dist
                
                # Resolve penetration
                if dist > 0:
                    self.rider_pos += (dist_vec / dist) * penetration

                # Calculate surface normal and apply collision response
                line_normal = np.array([-line_vec[1], line_vec[0]])
                line_normal /= np.linalg.norm(line_normal)
                
                # Ensure normal points towards the rider
                if np.dot(line_normal, rider_to_p1) < 0:
                    line_normal *= -1
                
                # Reflect velocity
                vel_dot_normal = np.dot(self.rider_vel, line_normal)
                if vel_dot_normal < 0:
                    self.rider_vel -= 2 * vel_dot_normal * line_normal
                
                self.rider_vel *= self.FRICTION
                self.rider_angle = math.atan2(line_vec[1], line_vec[0])
                
                # Add particles
                if np.linalg.norm(self.rider_vel) > 1.0:
                    self._create_particles(closest_point, self.rider_vel)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, rider_vel):
        for _ in range(2):
            angle = math.atan2(-rider_vel[1], -rider_vel[0]) + (self.np_random.random() - 0.5) * 0.5
            speed = self.np_random.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(10, 21)})

    def _calculate_reward(self):
        if self.game_over:
            return 0
        
        reward = 0
        
        # Reward for forward progress
        progress = self.rider_pos[0] - self.last_rider_x
        reward += progress * 0.1

        # Checkpoint reward
        for cp in self.checkpoints:
            if not cp["passed"] and self.last_rider_x < cp["x"] <= self.rider_pos[0]:
                cp["passed"] = True
                reward += 10
                # sfx: checkpoint.wav
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        if self.rider_pos[0] >= self.finish_x:
            all_checkpoints_passed = all(cp["passed"] for cp in self.checkpoints)
            if all_checkpoints_passed:
                self.score += 100
                self.game_over = True
                self.win = True
                # sfx: win.wav
                return True

        # Crash condition (out of bounds)
        if not (0 < self.rider_pos[0] < self.WIDTH and -50 < self.rider_pos[1] < self.HEIGHT + 50):
            self.score -= 10
            self.game_over = True
            # sfx: crash.wav
            return True
        
        # Stuck condition
        if np.linalg.norm(self.rider_vel) < self.STUCK_VEL_THRESHOLD:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
        
        if self.stuck_frames > self.STUCK_FRAMES_LIMIT:
            self.score -= 10 # Penalty for getting stuck
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw start/finish/checkpoints
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos[0], 0), (self.start_pos[0], self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.HEIGHT), 3)
        for cp in self.checkpoints:
            color = self.COLOR_START if cp["passed"] else self.COLOR_CHECKPOINT
            pygame.draw.line(self.screen, color, (cp["x"], 0), (cp["x"], self.HEIGHT), 2)
        
        # Draw placed lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1[0], p1[1]+1), (p2[0], p2[1]+1))

        # Draw particles
        for p in self.particles:
            size = int(p['life'] / 5)
            if size > 0:
                pygame.draw.circle(self.screen, self.COLOR_RIDER, (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw rider
        rx, ry = int(self.rider_pos[0]), int(self.rider_pos[1])
        sled_angle = self.rider_angle
        if not self.on_ground:
            sled_angle = math.atan2(self.rider_vel[1], self.rider_vel[0])

        s_cos = math.cos(sled_angle)
        s_sin = math.sin(sled_angle)
        
        sled_points = [
            (-10, 2), (10, 2), (5, -3), (-8, -3)
        ]
        transformed_sled = [
            (rx + p[0]*s_cos - p[1]*s_sin, ry + p[0]*s_sin + p[1]*s_cos)
            for p in sled_points
        ]
        pygame.gfxdraw.aapolygon(self.screen, transformed_sled, self.COLOR_SLED)
        pygame.gfxdraw.filled_polygon(self.screen, transformed_sled, self.COLOR_SLED)

        pygame.gfxdraw.aacircle(self.screen, rx, ry - 8, self.RIDER_RADIUS - 2, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rx, ry - 8, self.RIDER_RADIUS - 2, self.COLOR_RIDER)

        # Draw drawing preview
        if self.is_drawing:
            start = self.line_start_pos
            end = self.cursor_pos
            pygame.draw.line(self.screen, self.COLOR_TRACK_PREVIEW, start, end, 2)
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, (int(start[0]), int(start[1])), 4)

        # Draw cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 8, cy), (cx + 8, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 8), (cx, cy + 8), 2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.MAX_STEPS - self.steps)
        steps_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        speed_val = np.linalg.norm(self.rider_vel) * 5 # Arbitrary scale for display
        speed_text = self.font_ui.render(f"SPEED: {speed_val:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (10, 30))

        if self.game_over:
            message = "FINISH!" if self.win else "CRASHED"
            end_text = self.font_big.render(message, True, self.COLOR_FINISH if self.win else self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": self.rider_pos.tolist(),
            "rider_vel": self.rider_vel.tolist(),
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for internal validation during development and can be removed.
        print("Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play ---
    # To play manually, you need a Pygame window.
    # The environment is designed for headless rendering (rgb_array),
    # but we can display the frames to play.
    
    try:
        window = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Line Rider Gym Environment")
    except pygame.error:
        print("\nCould not create Pygame display. Running headless test.")
        window = None

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    print("\n" + GameEnv.user_guide)

    while True:
        if window:
            # Display the frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            window.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Resetting environment...")
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            if window:
                pygame.time.wait(2000)

        # Map keyboard keys to actions for manual play
        keys = pygame.key.get_pressed() if window else {}
        
        movement = 0 # none
        if keys.get(pygame.K_UP): movement = 1
        elif keys.get(pygame.K_DOWN): movement = 2
        elif keys.get(pygame.K_LEFT): movement = 3
        elif keys.get(pygame.K_RIGHT): movement = 4
        
        space_held = 1 if keys.get(pygame.K_SPACE) else 0
        shift_held = 1 if keys.get(pygame.K_LSHIFT) or keys.get(pygame.K_RSHIFT) else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    quit()
        
        env.clock.tick(GameEnv.FPS)