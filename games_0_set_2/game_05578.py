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


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows draw angled track segments. Space draws a long horizontal platform. Shift draws a short one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time to guide the rider to the finish line. Cross checkpoints for points and finish fast for a high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    MAX_TRACK_SEGMENTS = 75

    # Colors
    COLOR_BG = (28, 30, 36)
    COLOR_GRID = (40, 42, 50)
    COLOR_TRACK = (0, 160, 255)
    COLOR_RIDER = (255, 255, 255)
    COLOR_RIDER_GLOW = (255, 255, 255, 30)
    COLOR_START = (0, 255, 128)
    COLOR_FINISH = (255, 64, 64)
    COLOR_CHECKPOINT = (255, 215, 0)
    COLOR_UI_TEXT = (240, 240, 240)

    # Physics
    GRAVITY = pygame.math.Vector2(0, 0.25)
    RIDER_RADIUS = 8
    COLLISION_DAMPING = 0.85
    
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
        
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        self.render_mode = render_mode
        self.track_segments = []
        self.particles = []
        self.checkpoints = []
        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.last_reward_pos = pygame.math.Vector2(0, 0)
        
        # This is called to initialize attributes, even though we call it again in the public reset
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_reward_msg = ""
        self.last_reward_timer = 0

        self.rider_pos = pygame.math.Vector2(80, self.HEIGHT / 2)
        self.rider_vel = pygame.math.Vector2(2.5, 0)

        self.track_segments = []
        self.particles = []

        # Initial flat track to start the rider
        self._add_track_segment(pygame.math.Vector2(0, self.HEIGHT / 2 + 50), pygame.math.Vector2(150, self.HEIGHT / 2 + 50))

        # Checkpoints
        self.checkpoints = [
            {"pos": pygame.math.Vector2(250, 150), "triggered": False},
            {"pos": pygame.math.Vector2(450, 250), "triggered": False},
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_physics()
        
        # Base reward for survival
        reward += 0.01

        # Checkpoint collision
        for cp in self.checkpoints:
            if not cp["triggered"] and (cp["pos"] - self.rider_pos).length() < 20 + self.RIDER_RADIUS:
                cp["triggered"] = True
                self.score += 25
                reward += 5
                self._show_reward_msg("+25", cp["pos"])
                # sfx: checkpoint_get.wav

        self._handle_track_collisions()

        terminated, term_reward = self._check_termination()
        self.game_over = terminated
        reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            reward += -10 # Penalty for timeout
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        draw_point = self.rider_pos + pygame.math.Vector2(40, -10)
        
        # Priority: Space > Shift > Arrows
        if space_held:
            # Long horizontal segment
            start = draw_point + pygame.math.Vector2(-10, 0)
            end = draw_point + pygame.math.Vector2(90, 0)
            self._add_track_segment(start, end)
        elif shift_held:
            # Short horizontal segment
            start = draw_point
            end = draw_point + pygame.math.Vector2(50, 0)
            self._add_track_segment(start, end)
        elif movement > 0:
            angle = 0
            length = 45
            if movement == 1: angle = -45   # Up-Right
            elif movement == 2: angle = 45  # Down-Right
            elif movement == 3: angle = -135 # Up-Left
            elif movement == 4: angle = 135 # Down-Left
            
            end_offset = pygame.math.Vector2(length, 0).rotate(angle)
            self._add_track_segment(draw_point, draw_point + end_offset)

    def _add_track_segment(self, start, end):
        # Clip segments to be within bounds
        start.x = max(0, min(self.WIDTH, start.x))
        start.y = max(0, min(self.HEIGHT, start.y))
        end.x = max(0, min(self.WIDTH, end.x))
        end.y = max(0, min(self.HEIGHT, end.y))

        # Don't add zero-length segments
        if (start - end).length() < 1:
            return

        self.track_segments.append((start, end))
        if len(self.track_segments) > self.MAX_TRACK_SEGMENTS:
            self.track_segments.pop(0)
        # sfx: track_draw.wav

    def _update_physics(self):
        # Update Rider
        self.rider_vel += self.GRAVITY
        self.rider_pos += self.rider_vel

        # Update Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _handle_track_collisions(self):
        collided = False
        for start, end in self.track_segments:
            line_vec = end - start
            if line_vec.length_squared() == 0: continue

            point_vec = self.rider_pos - start
            t = line_vec.dot(point_vec) / line_vec.length_squared()
            
            closest_point = None
            if t < 0: closest_point = start
            elif t > 1: closest_point = end
            else: closest_point = start + t * line_vec

            dist_vec = self.rider_pos - closest_point
            if dist_vec.length() < self.RIDER_RADIUS:
                collided = True
                # Resolve penetration
                penetration_depth = self.RIDER_RADIUS - dist_vec.length()
                if dist_vec.length() > 0:
                    self.rider_pos += dist_vec.normalize() * penetration_depth
                
                # Calculate normal and reflect velocity
                normal = dist_vec.normalize() if dist_vec.length() > 0 else pygame.math.Vector2(0, -1)
                self.rider_vel = self.rider_vel.reflect(normal) * self.COLLISION_DAMPING
                
                # sfx: bounce.wav
                self._create_collision_particles(self.rider_pos, normal)
                break # Handle one collision per frame for stability

    def _check_termination(self):
        # Win condition
        if self.rider_pos.x > self.WIDTH - 40:
            self.win = True
            time_bonus = max(0, 50 - (self.steps / self.FPS))
            self.score += 100 + time_bonus
            reward = 50 + time_bonus
            self._show_reward_msg(f"FINISH! +{int(100+time_bonus)}", self.rider_pos)
            # sfx: win.wav
            return True, reward

        # Loss condition (out of bounds)
        if not (0 < self.rider_pos.y < self.HEIGHT and 0 < self.rider_pos.x < self.WIDTH):
            # sfx: crash.wav
            return True, -50

        return False, 0
    
    def _create_collision_particles(self, pos, normal):
        for _ in range(5):
            angle_offset = self.np_random.uniform(-45, 45)
            vel = normal.rotate(angle_offset) * self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': pygame.math.Vector2(pos), # FIX: Use constructor to copy
                'vel': vel,
                'lifetime': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(2, 4),
                'color': random.choice([self.COLOR_TRACK, self.COLOR_RIDER, (100, 200, 255)])
            })

    def _show_reward_msg(self, text, pos):
        self.last_reward_msg = text
        self.last_reward_pos = pygame.math.Vector2(pos) # FIX: Use constructor to copy
        self.last_reward_timer = self.FPS  # Show for 1 second

    def _get_observation(self):
        self._render_scene()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, channels), we need (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y)
        }

    def _render_scene(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # Start/Finish Lines
        pygame.draw.rect(self.screen, self.COLOR_START, (0, 0, 20, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.WIDTH - 20, 0, 20, self.HEIGHT))

        # Checkpoints
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if not cp["triggered"] else (80, 70, 0)
            pos_int = (int(cp["pos"].x), int(cp["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 20, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 20, color)

        # Track
        for start, end in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, start, end, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(start.x), int(start.y), 3, self.COLOR_TRACK)
            pygame.gfxdraw.filled_circle(self.screen, int(end.x), int(end.y), 3, self.COLOR_TRACK)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 20))
            # Create a temporary surface for alpha blending
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (p['pos'].x - p['radius'], p['pos'].y - p['radius']))

        # Rider
        rider_x, rider_y = int(self.rider_pos.x), int(self.rider_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS + 4, self.COLOR_RIDER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # UI
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (30, 10))

        time_elapsed = self.steps / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_elapsed:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 30, 10))

        # Game Over / Win Message
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "CRASHED!"
            msg_color = self.COLOR_START if self.win else self.COLOR_FINISH
            msg_render = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_render, msg_rect)

        # Floating reward text
        if self.last_reward_timer > 0:
            alpha = int(255 * (self.last_reward_timer / self.FPS))
            # Create a copy of the color tuple to add alpha
            color_with_alpha = (*self.COLOR_CHECKPOINT[:3], alpha)
            reward_render = self.font_ui.render(self.last_reward_msg, True, self.COLOR_UI_TEXT)
            reward_render.set_alpha(alpha)
            reward_rect = reward_render.get_rect(center=(self.last_reward_pos.x, self.last_reward_pos.y - 30))
            self.screen.blit(reward_render, reward_rect)
            self.last_reward_timer -= 1
            self.last_reward_pos.y -= 0.5
    
    def close(self):
        pygame.quit()

# --- Example Usage (for testing with a window) ---
if __name__ == "__main__":
    # To run with a window, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Track Rider")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Action mapping for human input ---
        keys = pygame.key.get_pressed()
        move = 0
        # Pygame's y-axis is inverted, so down arrow increases y
        if keys[pygame.K_UP]: move = 1     # Up-Right
        elif keys[pygame.K_DOWN]: move = 2 # Down-Right
        
        # Using left/right for other angles is not intuitive, so we'll map them differently for human play
        # This part is just for the human player example, the agent uses 1-4
        if keys[pygame.K_q]: move = 3 # Up-Left
        if keys[pygame.K_a]: move = 4 # Down-Left

        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is (H, W, C), but pygame surface needs (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        clock.tick(GameEnv.FPS)
        
    env.close()