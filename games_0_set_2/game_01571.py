
# Generated: 2025-08-28T02:02:48.692261
# Source Brief: brief_01571.md
# Brief Index: 1571

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to set draw direction. No action draws right. Space for longer segments, Shift for shorter."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in a 2D physics environment and guide your sled to the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.W, self.H = 640, 400
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        
        # Visuals & Constants
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_SLED = (230, 50, 50)
        self.COLOR_FINISH = (50, 230, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        self.FONT_S = pygame.font.SysFont("Arial", 16)
        self.FONT_L = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Game mechanics
        self.MAX_STEPS = 2000
        self.FINISH_X = self.W - 50
        self.SLED_SIZE = (20, 10)
        self.GRAVITY = 0.15
        self.FRICTION = 0.995
        self.SEGMENT_LENGTH = 20
        self.MAX_TRACK_POINTS = 200
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.sled_pos = None
        self.sled_vel = None
        self.track_points = None
        self.particles = None
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.particles = []
        
        # Initialize sled
        self.sled_pos = pygame.Vector2(60, 100)
        self.sled_vel = pygame.Vector2(2, 0)
        
        # Initialize track with a starting platform
        self.track_points = [
            pygame.Vector2(-10, 150),
            pygame.Vector2(120, 160)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # 1. Add new track segment based on action
        self._add_track_segment(movement, space_held, shift_held)

        # 2. Update game logic
        self._update_sled_physics()
        self._update_particles()
        self.steps += 1
        
        # 3. Calculate reward and check termination
        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _add_track_segment(self, movement, space_held, shift_held):
        length = self.SEGMENT_LENGTH
        if space_held: length *= 1.5
        if shift_held: length *= 0.5
        
        direction = pygame.Vector2()
        if movement == 0: direction.x = 1    # No-op -> right
        elif movement == 1: direction.y = -1 # Up
        elif movement == 2: direction.y = 1  # Down
        elif movement == 3: direction.x = -1 # Left
        elif movement == 4: direction.x = 1  # Right
        
        if direction.length() == 0:
            direction.x = 1 # Default to right if no direction
        
        direction.normalize_ip()
        
        last_point = self.track_points[-1]
        new_point = last_point + direction * length
        self.track_points.append(new_point)

        # Prune old track points
        if len(self.track_points) > self.MAX_TRACK_POINTS:
            self.track_points.pop(0)

    def _update_sled_physics(self):
        self.sled_vel.y += self.GRAVITY
        
        on_track = False
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            
            # Broad phase check
            if not (min(p1.x, p2.x) - self.SLED_SIZE[0] <= self.sled_pos.x <= max(p1.x, p2.x) + self.SLED_SIZE[0]):
                continue

            # Narrow phase check
            if p1.x == p2.x: continue # Avoid division by zero on vertical lines
            
            # Check if sled is horizontally above this segment
            if min(p1.x, p2.x) <= self.sled_pos.x <= max(p1.x, p2.x):
                t = (self.sled_pos.x - p1.x) / (p2.x - p1.x)
                track_y = p1.y + t * (p2.y - p1.y)
                
                sled_bottom = self.sled_pos.y + self.SLED_SIZE[1] / 2
                
                # Check for collision (sled bottom is near or just passed the track line)
                if sled_bottom >= track_y and sled_bottom - self.sled_vel.y < track_y + 5: # Small tolerance
                    on_track = True
                    
                    # Snap to track
                    self.sled_pos.y = track_y - self.SLED_SIZE[1] / 2
                    
                    track_vec = (p2 - p1).normalize()
                    normal_vec = pygame.Vector2(-track_vec.y, track_vec.x)
                    if normal_vec.y > 0: normal_vec *= -1
                    
                    # Inelastic collision response
                    perp_vel = self.sled_vel.dot(normal_vec)
                    if perp_vel > 0:
                        self.sled_vel -= perp_vel * normal_vec
                    
                    # Apply friction
                    self.sled_vel *= self.FRICTION
                    
                    # Spawn particles
                    # sfx: Sled scraping on snow
                    self._spawn_particles(self.sled_pos + pygame.Vector2(0, self.SLED_SIZE[1]/2), normal_vec)
                    break
        
        self.sled_pos += self.sled_vel

    def _spawn_particles(self, pos, surface_normal):
        num_particles = random.randint(1, 3)
        for _ in range(num_particles):
            vel = surface_normal.rotate(random.uniform(-60, 60)) * random.uniform(0.5, 2)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.uniform(1, 3),
                "lifetime": random.randint(15, 30)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["radius"] -= 0.05
        self.particles = [p for p in self.particles if p["lifetime"] > 0 and p["radius"] > 0]

    def _calculate_reward_and_termination(self):
        terminated = False
        reward = 0
        
        if self.sled_pos.x >= self.FINISH_X:
            reward = 100.0
            terminated = True
            self.game_over_message = "FINISH!"
            # sfx: Win jingle
        elif not (0 <= self.sled_pos.x < self.W and 0 <= self.sled_pos.y < self.H):
            reward = -10.0
            terminated = True
            self.game_over_message = "CRASHED!"
            # sfx: Crash sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "TIME UP"
        else:
            reward = 0.1 # Survival reward
            
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_finish_line()
        self._render_track()
        self._render_particles()
        self._render_sled()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for x in range(0, self.W, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.W, y))

    def _render_finish_line(self):
        line_width = 10
        for i in range(0, self.H, line_width * 2):
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.FINISH_X, i, line_width, line_width))
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.FINISH_X + line_width, i + line_width, line_width, line_width))

    def _render_track(self):
        if len(self.track_points) >= 2:
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, self.track_points, 3)

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*self.COLOR_TRACK, 150)
            )

    def _render_sled(self):
        # Create a surface for the sled
        sled_surf = pygame.Surface(self.SLED_SIZE, pygame.SRCALPHA)
        sled_surf.fill(self.COLOR_SLED)
        
        # Rotate the sled to match its velocity vector
        angle = self.sled_vel.angle_to(pygame.Vector2(1, 0))
        rotated_surf = pygame.transform.rotate(sled_surf, -angle)
        
        # Get the new rect for blitting, centered on the sled's position
        rotated_rect = rotated_surf.get_rect(center=self.sled_pos)
        self.screen.blit(rotated_surf, rotated_rect)

    def _render_ui(self):
        steps_text = self.FONT_S.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))
        
        score_text = self.FONT_S.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.W - 10, 10))
        self.screen.blit(score_text, score_rect)

        speed_text = self.FONT_S.render(f"SPEED: {self.sled_vel.length():.2f}", True, self.COLOR_TEXT)
        speed_rect = speed_text.get_rect(topright=(self.W - 10, 30))
        self.screen.blit(speed_text, speed_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))
        
        message_text = self.FONT_L.render(self.game_over_message, True, self.COLOR_TEXT)
        message_rect = message_text.get_rect(center=(self.W / 2, self.H / 2))
        self.screen.blit(message_text, message_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")