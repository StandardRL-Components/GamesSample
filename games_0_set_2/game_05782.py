
# Generated: 2025-08-28T06:04:46.504377
# Source Brief: brief_05782.md
# Brief Index: 5782

        
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
        "Controls: Arrow keys to set the angle of the next track segment. "
        "Hold Space for a longer segment. Hold Shift to skip drawing and let the sled move."
    )

    game_description = (
        "Draw a track for a sled to ride on, aiming for a fast and stylish completion. "
        "Reach the finish line on the right, but be careful! Three crashes and the run is over."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    MAX_CRASHES = 3

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_SLED = (255, 60, 60)
    COLOR_SLED_GLOW = (255, 120, 120)
    COLOR_TRACK = (240, 240, 240)
    COLOR_START_FINISH = (80, 200, 120)
    COLOR_CRASH = (255, 200, 0)
    COLOR_SPEED_LINE = (50, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    # Physics
    GRAVITY = 0.15
    SLED_SIZE = 12
    FRICTION = 0.995
    RESTITUTION = 0.6
    SEGMENT_LENGTH_NORMAL = 30
    SEGMENT_LENGTH_LONG = 50
    FINISH_LINE_X = SCREEN_WIDTH - 40

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.track_segments = []
        self.last_draw_pos = np.array([0.0, 0.0])
        self.crashes = 0
        self.steps = 0
        self.timer = 0.0
        self.game_over = False
        self.win = False
        self.particles = []
        self.speed_trails = deque(maxlen=20)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_pos = np.array([50.0, 100.0])
        self.sled_pos = start_pos.copy()
        self.sled_vel = np.array([2.0, 0.0])

        self.track_segments = []
        start_segment = (np.array([20.0, 100.0]), start_pos + np.array([10.0, 0.0]))
        self.track_segments.append(start_segment)
        self.last_draw_pos = start_segment[1].copy()

        self.crashes = 0
        self.steps = 0
        self.timer = 0.0
        self.game_over = False
        self.win = False

        self.particles = []
        self.speed_trails.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.steps += 1
        self.timer += 1 / 30.0  # Assuming 30 FPS for timer

        # 1. Handle Action: Draw Track
        if not shift_held:
            self._draw_track_segment(movement, space_held)

        # 2. Update Physics
        self._update_physics()

        # 3. Update Game State & Check for Events
        reward = self._update_game_state()

        # 4. Check for Termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.win: # Punish for timeout
             reward -= 10

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _draw_track_segment(self, movement_action, is_long):
        length = self.SEGMENT_LENGTH_LONG if is_long else self.SEGMENT_LENGTH_NORMAL
        
        # Angle based on action. 0 is horizontal right.
        angles = {
            0: 0,           # None: horizontal forward
            1: -math.pi / 3,  # Up: sharp up-forward
            2: math.pi / 3,   # Down: sharp down-forward
            3: 2 * math.pi / 3, # Left: down-backward
            4: -2 * math.pi / 3, # Right: up-backward
        }
        angle = angles[movement_action]

        start_point = self.last_draw_pos
        end_point = start_point + np.array([math.cos(angle) * length, math.sin(angle) * length])
        
        # Prevent drawing off-screen
        end_point[0] = np.clip(end_point[0], 0, self.SCREEN_WIDTH)
        end_point[1] = np.clip(end_point[1], 0, self.SCREEN_HEIGHT)

        new_segment = (start_point, end_point)
        self.track_segments.append(new_segment)
        self.last_draw_pos = end_point.copy()

    def _update_physics(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        self.sled_vel *= self.FRICTION

        # Update position
        self.sled_pos += self.sled_vel
        self.speed_trails.append(self.sled_pos.copy())

        # Collision detection and response
        collided = False
        for p1, p2 in self.track_segments:
            closest_point, on_segment = self._get_closest_point_on_segment(self.sled_pos, p1, p2)
            if not on_segment:
                continue

            vec_to_sled = self.sled_pos - closest_point
            dist = np.linalg.norm(vec_to_sled)

            if dist < self.SLED_SIZE / 2:
                collided = True
                # Positional correction
                overlap = self.SLED_SIZE / 2 - dist
                self.sled_pos += (vec_to_sled / dist) * overlap

                # Collision response
                segment_vec = p2 - p1
                if np.linalg.norm(segment_vec) == 0: continue
                
                # Normal vector of the segment surface
                normal = np.array([-segment_vec[1], segment_vec[0]])
                normal /= np.linalg.norm(normal)

                # Ensure normal points towards the sled
                if np.dot(normal, vec_to_sled) < 0:
                    normal *= -1

                vel_dot_normal = np.dot(self.sled_vel, normal)
                
                # Reflect velocity
                if vel_dot_normal < 0: # Moving towards the surface
                    self.sled_vel -= (1 + self.RESTITUTION) * vel_dot_normal * normal
                    # Sound: *thump*
                
                break # Handle one collision per frame

    def _update_game_state(self):
        reward = 0
        
        # Reward for forward progress
        reward += self.sled_vel[0] * 0.05
        
        # Check for crash
        if not (0 < self.sled_pos[1] < self.SCREEN_HEIGHT and 0 < self.sled_pos[0]):
            self.crashes += 1
            reward = -5.0
            self._create_particles(self.sled_pos, 30)
            # Sound: *crash_explosion*
            if self.crashes >= self.MAX_CRASHES:
                self.game_over = True
            else:
                self._reset_sled()
        
        # Check for win
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            self.win = True
            self.game_over = True
            time_bonus = 50 / max(1.0, self.timer)
            reward = 50 + time_bonus
            # Sound: *victory_fanfare*
            self._create_particles(self.sled_pos, 50, self.COLOR_START_FINISH)

        return reward

    def _reset_sled(self):
        self.sled_pos = self.last_draw_pos.copy() + np.array([0, -self.SLED_SIZE])
        self.sled_vel = np.array([0.0, 0.0])
        self.speed_trails.clear()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Start/Finish Lines
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (20, 0), (20, self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_START_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.SCREEN_HEIGHT), 2)

        # Speed Trails
        for i, pos in enumerate(self.speed_trails):
            alpha = int(255 * (i / len(self.speed_trails)))
            color = (*self.COLOR_SPEED_LINE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, color)

        # Track
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)

        # Sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE, self.SLED_SIZE)
        sled_rect.center = (int(self.sled_pos[0]), int(self.sled_pos[1]))
        
        # Glow effect
        glow_radius = int(self.SLED_SIZE * 0.8)
        glow_center = sled_rect.center
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, (*self.COLOR_SLED_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], int(glow_radius*0.7), (*self.COLOR_SLED_GLOW, 70))

        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect, border_radius=2)

        # Particles
        self._update_and_draw_particles()

    def _render_ui(self):
        crash_text = self.font_ui.render(f"Crashes: {self.crashes}/{self.MAX_CRASHES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crash_text, (10, 10))

        timer_text = self.font_ui.render(f"Time: {self.timer:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "FINISH!" if self.win else "GAME OVER"
            status_color = self.COLOR_START_FINISH if self.win else self.COLOR_SLED
            status_text = self.font_game_over.render(status_text_str, True, status_color)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "time": self.timer,
            "crashes": self.crashes,
            "steps": self.steps,
            "win": self.win,
        }

    def _get_closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            return a, True
        
        t = np.dot(ap, ab) / ab_squared
        t = np.clip(t, 0, 1) # Clamp to segment
        
        closest = a + t * ab
        on_segment = (0 <= t <= 1)
        return closest, on_segment

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_CRASH
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(20, 40),
                "radius": random.uniform(2, 5),
                "color": color,
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"][1] += self.GRAVITY * 0.1 # Particles have slight gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = int(255 * (p["life"] / 40))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color)
            
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Sled Drawer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        # For human play, we want to simulate continuously if no action is taken
        # So we use the "shift" action (skip drawing) as the default
        if not any(keys):
            action = [0, 0, 1] # Simulate without drawing

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

        if terminated:
            print(f"Game Over. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()