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

    user_guide = (
        "Controls: Arrows to move cursor. Hold Space to draw a track. Hold Shift to erase."
    )

    game_description = (
        "Draw a track in real-time for your sled to ride on. Navigate the terrain, "
        "hit checkpoints, and reach the finish line before time runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BG_ACCENT = (30, 45, 60)
        self.COLOR_TERRAIN = (60, 70, 80)
        self.COLOR_SLED = (50, 150, 255)
        self.COLOR_SLED_ACCENT = (200, 220, 255)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_START = (80, 200, 80)
        self.COLOR_FINISH = (200, 80, 80)
        self.COLOR_CHECKPOINT = (255, 200, 0)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_ERASER = (255, 80, 80, 100)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Game Constants ---
        self.GRAVITY = 0.35
        self.SLED_FRICTION = 0.995
        self.SLED_SIZE = 12
        self.CURSOR_SPEED = 6
        self.ERASER_RADIUS = 25
        self.MAX_TRACK_SEGMENTS = 200
        self.TIME_LIMIT_SECONDS = 60
        self.CHECKPOINT_RADIUS = 15

        # Will be initialized in reset
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_condition = None
        self.level = None
        self.timer = None
        self.sled_pos = None
        self.sled_vel = None
        self.sled_angle = None
        self.on_surface = None
        self.cursor_pos = None
        self.last_draw_point = None
        self.track_segments = None
        self.terrain = None
        self.start_pos = None
        self.finish_line_x = None
        self.checkpoints = None
        self.particles = None
        self.bg_elements = None
        self.last_sled_x = None

        self.reset()
        # self.validate_implementation() # Validation is good for dev, but not needed in final class

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = options.get("level", 1) if options else 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        
        self._generate_level()

        self.sled_pos = list(self.start_pos)
        self.sled_vel = [0.0, 0.0]
        self.sled_angle = 0.0
        self.on_surface = False
        self.last_sled_x = self.sled_pos[0]

        self.cursor_pos = [self.start_pos[0] + 50, self.start_pos[1]]
        self.last_draw_point = None
        self.track_segments = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if not self.game_over:
            drawn_length = self._handle_input(movement, space_held, shift_held)
            self._update_sled_physics()
            self._update_particles()
            
            self.timer -= 1
            reward = self._calculate_reward(drawn_length)
            self.score += reward

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            terminal_reward = 100.0 if self.win_condition else -100.0
            reward += terminal_reward
            self.score += terminal_reward

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.terrain = []
        self.checkpoints = []
        self.bg_elements = [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(10, 50)) for _ in range(20)]

        num_platforms = 4 + self.level
        max_gap = 20 + self.level * 5
        y_variance = 40

        last_x = 0
        current_y = self.HEIGHT * 0.75
        self.start_pos = (60, current_y - 30)

        for i in range(num_platforms):
            gap = self.np_random.integers(10, max_gap)
            start_x = last_x + gap
            length = self.np_random.integers(80, 200)
            end_x = start_x + length
            
            if end_x > self.WIDTH - 50:
                end_x = self.WIDTH
                self.terrain.append(((start_x, current_y), (end_x, current_y)))
                break

            self.terrain.append(((start_x, current_y), (end_x, current_y)))

            # Add checkpoints on platforms
            if i > 0 and i % 2 == 0:
                cp_x = start_x + length / 2
                cp_y = current_y - self.CHECKPOINT_RADIUS * 2
                self.checkpoints.append({"pos": (cp_x, cp_y), "active": True})

            last_x = end_x
            current_y += self.np_random.integers(-y_variance, y_variance)
            current_y = np.clip(current_y, self.HEIGHT * 0.4, self.HEIGHT - 20)

        self.finish_line_x = last_x

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        drawn_length = 0
        if shift_held:
            # sfx: eraser_sound
            new_tracks = []
            for p1, p2 in self.track_segments:
                mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                if math.hypot(mid_point[0] - self.cursor_pos[0], mid_point[1] - self.cursor_pos[1]) > self.ERASER_RADIUS:
                    new_tracks.append((p1, p2))
            self.track_segments = new_tracks
            self.last_draw_point = None

        elif space_held:
            # sfx: draw_line
            if self.last_draw_point is None:
                self.last_draw_point = tuple(self.cursor_pos)
            
            current_point = tuple(self.cursor_pos)
            if self.last_draw_point != current_point:
                segment = (self.last_draw_point, current_point)
                drawn_length = math.hypot(segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
                self.track_segments.append(segment)
                self.last_draw_point = current_point
                if len(self.track_segments) > self.MAX_TRACK_SEGMENTS:
                    self.track_segments.pop(0)
        else:
            self.last_draw_point = None
            
        return drawn_length

    def _update_sled_physics(self):
        surfaces = self.track_segments + self.terrain
        self.on_surface = False
        
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY

        best_surface = None
        min_dist = float('inf')
        surface_y = self.HEIGHT + 100

        for p1, p2 in surfaces:
            # Bounding box check for efficiency
            if not (min(p1[0], p2[0]) - self.SLED_SIZE <= self.sled_pos[0] <= max(p1[0], p2[0]) + self.SLED_SIZE):
                continue

            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if dx == 0 and dy == 0: continue
            
            line_mag_sq = dx*dx + dy*dy
            t = ((self.sled_pos[0] - p1[0]) * dx + (self.sled_pos[1] - p1[1]) * dy) / line_mag_sq
            t = np.clip(t, 0, 1)
            
            closest_point = (p1[0] + t * dx, p1[1] + t * dy)
            dist_sq = (self.sled_pos[0] - closest_point[0])**2 + (self.sled_pos[1] - closest_point[1])**2

            if dist_sq < min_dist:
                min_dist = dist_sq
                best_surface = (p1, p2)
                surface_y = closest_point[1]

        if best_surface and self.sled_pos[1] + self.SLED_SIZE / 2 >= surface_y and min_dist < (self.SLED_SIZE / 2)**2:
            self.on_surface = True
            # sfx: sled_scrape
            p1, p2 = best_surface
            self.sled_pos[1] = surface_y - self.SLED_SIZE / 2

            surface_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            self.sled_angle = surface_angle

            # Project velocity onto surface
            dot_product = self.sled_vel[0] * math.cos(surface_angle) + self.sled_vel[1] * math.sin(surface_angle)
            self.sled_vel[0] = dot_product * math.cos(surface_angle)
            self.sled_vel[1] = dot_product * math.sin(surface_angle)

            # Add gravity component along the slope
            gravity_effect = self.GRAVITY * math.sin(surface_angle)
            self.sled_vel[0] += gravity_effect
            
            # Dampen vertical velocity to prevent bouncing
            self.sled_vel[1] *= 0.5 
        
        self.sled_vel[0] *= self.SLED_FRICTION
        self.sled_pos[0] += self.sled_vel[0]
        self.sled_pos[1] += self.sled_vel[1]
        
        # Emit particles
        if self.on_surface and math.hypot(*self.sled_vel) > 1:
            for _ in range(2):
                p_vel = [self.np_random.uniform(-1, 1) - self.sled_vel[0]*0.1, self.np_random.uniform(-0.5, 0)]
                self.particles.append({'pos': list(self.sled_pos), 'vel': p_vel, 'life': 20, 'size': self.np_random.uniform(2, 5)})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _calculate_reward(self, drawn_length):
        reward = 0.0
        
        # Reward for forward progress
        progress = self.sled_pos[0] - self.last_sled_x
        reward += progress * 0.1
        self.last_sled_x = self.sled_pos[0]

        # Penalty for drawing
        reward -= drawn_length * 0.01

        # Reward for checkpoints
        for cp in self.checkpoints:
            if cp["active"]:
                dist = math.hypot(self.sled_pos[0] - cp["pos"][0], self.sled_pos[1] - cp["pos"][1])
                if dist < self.CHECKPOINT_RADIUS + self.SLED_SIZE / 2:
                    cp["active"] = False
                    reward += 5.0
                    self.timer += 10 * self.FPS # Add 10 seconds
                    # sfx: checkpoint_get

        return reward

    def _check_termination(self):
        # Crash condition
        if self.sled_pos[1] > self.HEIGHT + self.SLED_SIZE or self.sled_pos[0] < -self.SLED_SIZE:
            return True
        # Win condition
        if self.sled_pos[0] > self.finish_line_x:
            self.win_condition = True
            return True
        # Time out
        if self.timer <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_terrain()
        self._render_track()
        self._render_checkpoints_and_finish()
        self._render_particles()
        self._render_sled()
        self._render_cursor()

    def _render_background(self):
        for x, y, size in self.bg_elements:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), self.COLOR_BG_ACCENT)

    def _render_terrain(self):
        for p1, p2 in self.terrain:
            pygame.draw.line(self.screen, self.COLOR_TERRAIN, p1, p2, 10)

    def _render_track(self):
        if len(self.track_segments) > 1:
            all_points = [self.track_segments[0][0]] + [seg[1] for seg in self.track_segments]
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, all_points, 1)
        elif self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, self.track_segments[0][0], self.track_segments[0][1], 1)


    def _render_checkpoints_and_finish(self):
        # Start Line
        start_y = self.start_pos[1] + self.SLED_SIZE
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos[0], start_y - 20), (self.start_pos[0], start_y + 20), 3)

        # Finish Line
        finish_y = self.HEIGHT
        for p1, p2 in self.terrain:
            if p1[0] <= self.finish_line_x <= p2[0]:
                finish_y = p1[1]
                break
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, finish_y - 20), (self.finish_line_x, finish_y + 20), 3)

        # Checkpoints
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if cp["active"] else self.COLOR_TERRAIN
            pos = (int(cp["pos"][0]), int(cp["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CHECKPOINT_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CHECKPOINT_RADIUS, self.COLOR_TEXT)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_SLED_ACCENT, alpha)
            size = int(p['size'] * (p['life'] / 20))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_sled(self):
        x, y = int(self.sled_pos[0]), int(self.sled_pos[1])
        angle = self.sled_angle if self.on_surface else self.sled_vel[0] * 0.05
        
        points = [
            (-self.SLED_SIZE, -self.SLED_SIZE / 4),
            (self.SLED_SIZE, -self.SLED_SIZE / 4),
            (self.SLED_SIZE * 0.8, self.SLED_SIZE / 2),
            (-self.SLED_SIZE * 0.8, self.SLED_SIZE / 2),
        ]
        
        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle) - py * math.sin(angle) + x
            ry = px * math.sin(angle) + py * math.cos(angle) + y
            rotated_points.append((rx, ry))
        
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED_ACCENT)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        _, shift_held = self.action_space.sample()[1:] # A bit of a hack to check shift state for rendering
        
        if shift_held:
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ERASER_RADIUS, self.COLOR_ERASER)
        else:
            pygame.gfxdraw.filled_circle(self.screen, x, y, 5, self.COLOR_CURSOR)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.timer / self.FPS)
        time_color = self.COLOR_FINISH if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_ui.render(f"Time: {time_left:.1f}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, 10))

        if self.game_over:
            msg = "FINISH!" if self.win_condition else "GAME OVER"
            color = self.COLOR_START if self.win_condition else self.COLOR_FINISH
            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_left": max(0, self.timer / self.FPS),
            "win": self.win_condition,
        }

    def validate_implementation(self):
        # This is a helper function for development and is not required by the problem.
        print("Validating implementation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually.
    # It will create a window and map keyboard keys to actions.
    
    # Un-set the dummy video driver to allow a window to be created.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Sled Drawer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while not done:
        action = [0, 0, 0] # no-op, released, released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation on the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    pygame.quit()
    print(f"Game Over! Final Info: {info}")