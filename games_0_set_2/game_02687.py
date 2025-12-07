
# Generated: 2025-08-28T05:37:06.814538
# Source Brief: brief_02687.md
# Brief Index: 2687

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to set line start. Shift to draw line. Guide the sled to the flag."
    )

    game_description = (
        "A physics-based puzzle game inspired by Line Rider. Draw tracks to guide a sled from the start to the finish."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    CURSOR_SPEED = 5.0
    GRAVITY = 0.15
    FRICTION = 0.998
    CONTACT_DISTANCE = 8.0

    # --- Colors ---
    COLOR_BG = (210, 220, 230)
    COLOR_TRACK = (20, 20, 20)
    COLOR_SLED = (220, 50, 50)
    COLOR_RIDER = (30, 30, 30)
    COLOR_START = (60, 180, 60)
    COLOR_END = (200, 60, 60)
    COLOR_CURSOR = (50, 50, 200, 150)
    COLOR_GHOST_LINE = (100, 100, 100, 150)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (10, 10, 10)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.sled_angle = 0.0
        self.on_track = False

        self.track_segments = []
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.line_start_pos = None

        self.particles = deque(maxlen=100)
        
        self.start_zone = pygame.Rect(30, self.SCREEN_HEIGHT - 100, 50, 50)
        self.end_zone = pygame.Rect(self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 100, 50, 50)

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.sled_pos = pygame.math.Vector2(self.start_zone.centerx, self.start_zone.top - 20)
        self.sled_vel = pygame.math.Vector2(0.5, 0)
        self.sled_angle = 0.0
        self.on_track = False

        initial_ground = (
            pygame.math.Vector2(self.start_zone.left - 20, self.start_zone.top + 20),
            pygame.math.Vector2(self.start_zone.right + 20, self.start_zone.top + 20)
        )
        self.track_segments = [initial_ground]

        self.cursor_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.line_start_pos = None
        
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_drawing(movement, space_held, shift_held)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_drawing(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        if space_held:
            self.line_start_pos = pygame.math.Vector2(self.cursor_pos)
            # SFX: Line start point placed sound

        if shift_held and self.line_start_pos is not None:
            new_line = (self.line_start_pos, self.cursor_pos.copy())
            if new_line[0].distance_to(new_line[1]) > 2: # Avoid zero-length lines
                self.track_segments.append(new_line)
            self.line_start_pos = None
            # SFX: Line drawn sound

    def _update_physics(self):
        closest_line, dist, on_segment = self._find_closest_line()

        if closest_line and dist < self.CONTACT_DISTANCE:
            self.on_track = True
            p1, p2 = closest_line
            line_vec = p2 - p1
            line_angle = math.atan2(line_vec.y, line_vec.x)
            
            # Snap position to the line
            line_normal = line_vec.rotate(90).normalize()
            self.sled_pos -= line_normal * (dist - self.CONTACT_DISTANCE / 2.0)
            
            # Apply gravity along the slope
            gravity_force = self.GRAVITY * math.sin(line_angle)
            self.sled_vel.x += gravity_force * math.cos(line_angle)
            self.sled_vel.y += gravity_force * math.sin(line_angle)
            
            # Project velocity onto the line and apply friction
            current_speed = self.sled_vel.length()
            self.sled_vel = line_vec.normalize() * current_speed * self.FRICTION
            self.sled_angle = line_angle
        else:
            self.on_track = False
            # Standard projectile motion (in air)
            self.sled_vel.y += self.GRAVITY
            self.sled_angle = math.atan2(self.sled_vel.y, self.sled_vel.x)

        self.sled_pos += self.sled_vel
        
    def _find_closest_line(self):
        min_dist = float('inf')
        closest_line = None
        on_segment = False

        for line in self.track_segments:
            p1, p2 = line
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
                
            point_vec = self.sled_pos - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            
            if 0 <= t <= 1:
                closest_point_on_line = p1 + t * line_vec
                dist = self.sled_pos.distance_to(closest_point_on_line)
                is_on_segment = True
            else:
                dist1 = self.sled_pos.distance_to(p1)
                dist2 = self.sled_pos.distance_to(p2)
                dist = min(dist1, dist2)
                is_on_segment = False

            if dist < min_dist:
                min_dist = dist
                closest_line = line
                on_segment = is_on_segment
        
        return closest_line, min_dist, on_segment

    def _update_particles(self):
        if self.on_track and self.sled_vel.length() > 1.0:
            for _ in range(2):
                particle = {
                    "pos": self.sled_pos.copy() + (random.uniform(-3, 3), random.uniform(-3, 3)),
                    "vel": self.sled_vel.rotate(random.uniform(-30, 30)) * -0.1,
                    "life": random.randint(20, 40),
                    "size": random.uniform(2, 5)
                }
                self.particles.append(particle)

        for p in list(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] -= 0.1
            if p["life"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        reward = 0.0
        
        # Reward for moving towards the goal
        goal_dir = (self.end_zone.centerx - self.sled_pos.x)
        if goal_dir != 0:
            vel_alignment = self.sled_vel.x / abs(goal_dir) if self.sled_vel.x * goal_dir > 0 else -abs(self.sled_vel.x)
            reward += 0.1 * vel_alignment

        # Penalty for moving away from goal
        if self.sled_vel.x < -0.1 and self.sled_pos.x > self.start_zone.centerx:
             reward -= 0.1

        # Check win/loss
        if self.end_zone.collidepoint(self.sled_pos):
            reward += 100.0  # Big win reward
        
        if not (0 < self.sled_pos.x < self.SCREEN_WIDTH and 0 < self.sled_pos.y < self.SCREEN_HEIGHT):
            reward -= 10.0 # Crash penalty

        return reward

    def _check_termination(self):
        if self.end_zone.collidepoint(self.sled_pos):
            # SFX: Win fanfare
            return True
        if not (0 < self.sled_pos.x < self.SCREEN_WIDTH and 0 < self.sled_pos.y < self.SCREEN_HEIGHT):
            # SFX: Crash sound
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start and end zones
        pygame.draw.rect(self.screen, self.COLOR_START, self.start_zone, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_END, self.end_zone, border_radius=5)
        
        # Draw flag
        flag_pole = (self.end_zone.centerx - 15, self.end_zone.bottom)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, flag_pole, (flag_pole[0], flag_pole[1] - 40), 3)
        pygame.draw.polygon(self.screen, self.COLOR_END, [
            (flag_pole[0], flag_pole[1] - 40),
            (flag_pole[0] + 20, flag_pole[1] - 30),
            (flag_pole[0], flag_pole[1] - 20)
        ])

        # Draw track segments
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

        # Draw ghost line
        if self.line_start_pos:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.aaline(s, self.COLOR_GHOST_LINE, self.line_start_pos, self.cursor_pos, 2)
            pygame.gfxdraw.filled_circle(s, int(self.line_start_pos.x), int(self.line_start_pos.y), 4, self.COLOR_GHOST_LINE)
            self.screen.blit(s, (0, 0))

        # Draw cursor
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, int(self.cursor_pos.x), int(self.cursor_pos.y), 8, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(s, int(self.cursor_pos.x), int(self.cursor_pos.y), 8, (255,255,255))
        self.screen.blit(s, (0, 0))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p["pos"].x), int(p["pos"].y)), int(p["size"]))

        # Draw sled
        sled_width, sled_height = 20, 6
        sled_surf = pygame.Surface((sled_width, sled_height + 8), pygame.SRCALPHA)
        # Sled body
        pygame.draw.rect(sled_surf, self.COLOR_SLED, (0, 8, sled_width, sled_height), border_radius=2)
        # Rider
        pygame.draw.circle(sled_surf, self.COLOR_RIDER, (sled_width/2, 5), 5)

        rotated_sled = pygame.transform.rotate(sled_surf, -math.degrees(self.sled_angle))
        sled_rect = rotated_sled.get_rect(center=self.sled_pos)
        self.screen.blit(rotated_sled, sled_rect.topleft)
        
    def _render_ui(self):
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        vel_text = self.font_small.render(f"VEL: {self.sled_vel.length():.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(vel_text, (self.SCREEN_WIDTH - vel_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.end_zone.collidepoint(self.sled_pos):
                end_text = self.font_large.render("FINISH!", True, self.COLOR_START)
            else:
                end_text = self.font_large.render("CRASHED", True, self.COLOR_END)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for visualization
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a simple mapping from keyboard to MultiDiscrete action
    key_action_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame setup for rendering
    pygame.display.set_caption("Line Rider Gym Env")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        for key, move_action in key_action_map.items():
            if keys[key]:
                action[0] = move_action
                break # only one move at a time
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()