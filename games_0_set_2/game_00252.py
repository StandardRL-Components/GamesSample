
# Generated: 2025-08-27T13:05:17.863141
# Source Brief: brief_00252.md
# Brief Index: 252

        
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


class Particle:
    """A simple particle class for visual effects like snow spray."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifetime -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys to move the drawing cursor. Hold Shift to set the start of a line, "
        "and press Space to set the end point and draw the line. The rider will then "
        "simulate on the new track."
    )

    game_description = (
        "A physics-based puzzle game inspired by Line Rider. Draw tracks for a sledder "
        "to navigate from the start to the finish line. Be creative, but be quick - "
        "the clock is ticking!"
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_SIM_STEPS_PER_ACTION = 120 # Max simulation time after drawing a line

    COLOR_BG = (240, 240, 245)
    COLOR_TRACK = (20, 20, 20)
    COLOR_RIDER = (0, 120, 255)
    COLOR_RIDER_SLED = (150, 75, 0)
    COLOR_START = (0, 200, 100)
    COLOR_FINISH = (220, 50, 50)
    COLOR_CURSOR = (255, 100, 0)
    COLOR_DRAFT_LINE = (150, 150, 150)
    COLOR_UI_TEXT = (10, 10, 10)
    
    GRAVITY = 0.15
    FRICTION = 0.995
    BOUNCINESS = 0.3
    RIDER_RADIUS = 5
    CURSOR_SPEED = 8
    MIN_LINE_LENGTH_FOR_REWARD = 20
    
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
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_game_over = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_time = 60.0 # Total game time in seconds
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.total_time
        
        self.start_pos = pygame.math.Vector2(50, 100)
        self.finish_x = self.WIDTH - 50

        self.rider_pos = pygame.math.Vector2(self.start_pos)
        self.rider_vel = pygame.math.Vector2(1, 0)
        self.rider_on_ground = False
        self.last_rider_angle = 0

        self.lines = [
            ((self.start_pos.x - 20, self.start_pos.y + 20), (self.start_pos.x + 30, self.start_pos.y + 20))
        ]
        
        self.drawing_cursor_pos = pygame.math.Vector2(self.start_pos.x + 50, self.start_pos.y + 20)
        self.current_line_start = None
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        # --- Handle Drawing Input ---
        self._move_cursor(movement)

        if shift_pressed:
            self.current_line_start = pygame.math.Vector2(self.drawing_cursor_pos)
            # sfx: UI_select.wav

        line_drawn = False
        if space_pressed and self.current_line_start is not None:
            line_end = self.drawing_cursor_pos
            if self.current_line_start.distance_to(line_end) > 1: # Avoid zero-length lines
                self.lines.append((tuple(self.current_line_start), tuple(line_end)))
                line_drawn = True
                # sfx: draw_line.wav
            
            line_length = self.current_line_start.distance_to(line_end)
            if line_length < self.MIN_LINE_LENGTH_FOR_REWARD:
                reward -= 1.0 # Penalty for drawing tiny, useless lines
            
            self.current_line_start = None

        # --- Run Simulation if a line was drawn ---
        if line_drawn:
            sim_steps = 0
            while sim_steps < self.MAX_SIM_STEPS_PER_ACTION and not self.game_over:
                sim_reward = self._update_physics()
                reward += sim_reward
                self._update_particles()
                
                self.time_remaining -= 1 / self.FPS
                if self.time_remaining <= 0:
                    self.game_over = True
                    reward -= 10 # Timeout penalty
                    break
                sim_steps += 1
        else:
            # Small penalty for inaction to encourage drawing
            reward -= 0.1
            self.time_remaining -= 1 / self.FPS
            if self.time_remaining <= 0:
                self.game_over = True
                reward -= 10

        dist_to_finish = (self.finish_x - self.rider_pos.x) / self.WIDTH
        reward -= dist_to_finish * 0.01

        # --- Check for termination ---
        terminated = self.game_over or self.steps > 1000

        if self.rider_pos.x > self.finish_x and not self.game_over:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
            # sfx: victory_fanfare.wav
        
        if (self.rider_pos.y > self.HEIGHT + 20 or self.rider_pos.x < -20 or self.rider_pos.x > self.WIDTH + 20) and not self.game_over:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: fall_scream.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: self.drawing_cursor_pos.y -= self.CURSOR_SPEED # Up
        elif movement == 2: self.drawing_cursor_pos.y += self.CURSOR_SPEED # Down
        elif movement == 3: self.drawing_cursor_pos.x -= self.CURSOR_SPEED # Left
        elif movement == 4: self.drawing_cursor_pos.x += self.CURSOR_SPEED # Right
        self.drawing_cursor_pos.x = np.clip(self.drawing_cursor_pos.x, 0, self.WIDTH)
        self.drawing_cursor_pos.y = np.clip(self.drawing_cursor_pos.y, 0, self.HEIGHT)

    def _update_physics(self):
        step_reward = 0
        
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        self.rider_pos += self.rider_vel

        self.rider_on_ground = False
        closest_dist = float('inf')
        collision_line = None
        
        for line in self.lines:
            p1 = pygame.math.Vector2(line[0])
            p2 = pygame.math.Vector2(line[1])
            
            # Find closest point on line segment
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            t = ((self.rider_pos - p1).dot(line_vec)) / line_vec.length_squared()
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec
            
            dist = self.rider_pos.distance_to(closest_point)

            if dist < closest_dist:
                closest_dist = dist
                collision_line = (p1, p2, closest_point)

        if collision_line and closest_dist < self.RIDER_RADIUS:
            self.rider_on_ground = True
            step_reward += 0.1 # Reward for being on a track
            
            p1, p2, closest_point = collision_line
            
            # Resolve penetration
            penetration = self.RIDER_RADIUS - closest_dist
            normal = (self.rider_pos - closest_point).normalize() if (self.rider_pos - closest_point).length() > 0 else pygame.math.Vector2(0, -1)
            self.rider_pos += normal * penetration

            # Calculate line normal (pointing "upwards")
            line_normal = (p2 - p1).rotate(90).normalize()
            if line_normal.y > 0:
                line_normal *= -1
            
            # Decompose velocity
            v_perp = line_normal * self.rider_vel.dot(line_normal)
            v_para = self.rider_vel - v_perp
            
            # Apply friction and bounce
            self.rider_vel = (v_para * self.FRICTION) - (v_perp * self.BOUNCINESS)
            # sfx: sled_scrape.wav (continuous)
            
            # Check for "risky" maneuver (sharp turn)
            current_angle = math.atan2(self.rider_vel.y, self.rider_vel.x)
            angle_change = abs(math.degrees(current_angle - self.last_rider_angle))
            if angle_change > 15: # Sharp turn
                step_reward += 5
                # sfx: whoosh_turn.wav
                for _ in range(5):
                    spray_vel = -self.rider_vel.normalize().rotate(random.uniform(-30, 30)) * random.uniform(1, 3)
                    self.particles.append(Particle(self.rider_pos, spray_vel, random.uniform(2, 4), (200, 200, 220), 20))
            self.last_rider_angle = current_angle

        return step_reward
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start and finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos.x, 0), (self.start_pos.x, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.HEIGHT), 3)
        
        # Draw all tracks
        for line in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, line[0], line[1], 3)
            
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw the rider
        self._draw_rider()
        
        # Draw drawing UI
        if not self.game_over:
            self._draw_cursor()

    def _draw_rider(self):
        p = self.rider_pos
        v = self.rider_vel
        angle = math.atan2(v.y, v.x)
        
        # Body
        body_start = p + pygame.math.Vector2(8, 0).rotate_rad(-angle + math.pi/2)
        body_end = p - pygame.math.Vector2(12, 0).rotate_rad(-angle + math.pi/2)
        pygame.draw.line(self.screen, self.COLOR_RIDER, body_start, body_end, 3)

        # Sled
        sled_p1 = p + pygame.math.Vector2(15, -3).rotate_rad(-angle)
        sled_p2 = p + pygame.math.Vector2(-15, -3).rotate_rad(-angle)
        pygame.draw.line(self.screen, self.COLOR_RIDER_SLED, sled_p1, sled_p2, 4)

        # Head
        head_pos = body_start + pygame.math.Vector2(0, -6).rotate_rad(-angle + math.pi/2)
        pygame.draw.circle(self.screen, self.COLOR_RIDER, (int(head_pos.x), int(head_pos.y)), 5)
        pygame.gfxdraw.aacircle(self.screen, int(head_pos.x), int(head_pos.y), 5, self.COLOR_RIDER)

    def _draw_cursor(self):
        # Draw cursor
        x, y = int(self.drawing_cursor_pos.x), int(self.drawing_cursor_pos.y)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 5, y), (x + 5, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 5), (x, y + 5), 2)
        
        # Draw line being drafted
        if self.current_line_start:
            start_pos = (int(self.current_line_start.x), int(self.current_line_start.y))
            end_pos = (x, y)
            # Draw dashed line
            dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
            dist = math.hypot(dx, dy)
            if dist == 0: return
            dashes = int(dist / 10)
            for i in range(dashes):
                start = (start_pos[0] + dx * i / dashes, start_pos[1] + dy * i / dashes)
                end = (start_pos[0] + dx * (i + 0.5) / dashes, start_pos[1] + dy * (i + 0.5) / dashes)
                if i % 2 == 0:
                    pygame.draw.line(self.screen, self.COLOR_DRAFT_LINE, start, end, 2)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_text = self.font_ui.render(f"Time: {max(0, self.time_remaining):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            if self.rider_pos.x > self.finish_x:
                msg = "GOAL!"
                color = self.COLOR_START
            else:
                msg = "TRY AGAIN"
                color = self.COLOR_FINISH
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
        }

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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Control the human-playable frame rate
        
    env.close()