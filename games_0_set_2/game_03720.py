# Generated: 2025-08-28T00:11:49.981927
# Source Brief: brief_03720.md
# Brief Index: 3720

        
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
    """
    An arcade physics game where the player draws tracks for a sled to ride on.

    The goal is to guide a sled from a starting platform to a finish line by
    drawing a path for it. The sled is subject to gravity and will slide along
    the drawn tracks. Crashing into the procedurally generated terrain or falling
    off the screen ends the episode. The game rewards speed and reaching the
    finish line.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to draw a track point."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sled down a procedurally generated track. Draw lines for it to slide on and reach the finish line as fast as possible."
    )

    # Frames auto-advance for real-time physics simulation.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 1000  # The conceptual width of the level

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (240, 248, 255)  # Alice Blue
        self.COLOR_SLED = (220, 20, 60)  # Crimson
        self.COLOR_SLED_HIGHLIGHT = (255, 105, 180) # Hot Pink
        self.COLOR_TERRAIN = (139, 69, 19)  # Saddle Brown
        self.COLOR_TRACK = (25, 25, 25)  # Dark Grey
        self.COLOR_CURSOR = (0, 0, 0, 100) # Semi-transparent black
        self.COLOR_START = (60, 179, 113)  # Medium Sea Green
        self.COLOR_FINISH = (255, 69, 0)  # Orange Red
        self.COLOR_TEXT = (10, 10, 10)
        self.COLOR_PARTICLE = (255, 255, 255)

        # Game constants
        self.GRAVITY = 0.15
        self.FRICTION = 0.995
        self.MAX_STEPS = 1500
        self.START_X = 50
        self.FINISH_X = self.WORLD_WIDTH - 50
        self.CURSOR_SPEED = 5

        # Initialize state variables
        self.sled_pos = None
        self.sled_vel = None
        self.terrain_points = None
        self.player_track = None
        self.cursor_pos = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.last_sled_x = 0.0
        self.prev_space_held = False
        self.terrain_roughness = 0.0

        # self.reset() is called by the API, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.terrain_roughness = 0.05

        start_y = self.HEIGHT * 0.3
        self.sled_pos = np.array([float(self.START_X), start_y - 20.0])
        self.sled_vel = np.array([2.0, 0.0])
        self.last_sled_x = self.sled_pos[0]

        self.cursor_pos = np.array([self.WIDTH // 2, self.HEIGHT // 2])
        self.player_track = [np.array([float(self.START_X - 20), start_y]), np.array([float(self.START_X + 20), start_y])]
        self.particles = []
        self.prev_space_held = False
        
        self._generate_terrain()

        return self._get_observation(), self._get_info()

    def _generate_terrain(self):
        self.terrain_points = []
        y = self.HEIGHT * 0.8
        for x in range(0, self.WORLD_WIDTH + 100, 20):
            self.terrain_points.append((x, y))
            y += self.np_random.uniform(-15, 15) * self.terrain_roughness
            y = np.clip(y, self.HEIGHT * 0.5, self.HEIGHT)
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
        
        self.time_elapsed += self.clock.get_time() / 1000.0
        
        if not self.game_over:
            self._handle_input(action)
            self._update_sled()
            self._update_particles()
        
        self.steps += 1
        
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 500 == 0:
            self.terrain_roughness += 0.05

        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT - 1)

        # Place track point on space press (rising edge)
        if space_held and not self.prev_space_held:
            # Convert screen-space cursor to world-space
            new_point = self.cursor_pos + np.array([self.sled_pos[0] - self.WIDTH/2, 0])
            
            # Add point if it's reasonably close to the last one
            if len(self.player_track) > 0:
                dist = np.linalg.norm(new_point - self.player_track[-1])
                if dist < 200: # Max draw distance
                    self.player_track.append(new_point)
            else:
                 self.player_track.append(new_point)
        
        self.prev_space_held = space_held

    def _update_sled(self):
        # Apply gravity
        self.sled_vel[1] += self.GRAVITY
        self.sled_vel *= self.FRICTION

        # Predict next position
        next_pos = self.sled_pos + self.sled_vel
        
        on_surface = False
        
        # Combine player track and terrain for collision detection.
        # Player track is checked first for precedence.
        all_tracks_points = [self.player_track, [np.array(p) for p in self.terrain_points]]

        for track_points in all_tracks_points:
            if on_surface: # If we found a collision on the player track, we're done.
                break
            
            for i in range(len(track_points) - 1):
                p1 = track_points[i]
                p2 = track_points[i+1]

                # Broad phase check
                if not (min(p1[0], p2[0]) - 10 < next_pos[0] < max(p1[0], p2[0]) + 10):
                    continue

                # Line-point distance calculation
                d = p2 - p1
                line_len_sq = d[0]**2 + d[1]**2
                if line_len_sq == 0: continue

                t = ((next_pos[0] - p1[0]) * d[0] + (next_pos[1] - p1[1]) * d[1]) / line_len_sq
                t = np.clip(t, 0, 1)
                
                closest_point = p1 + t * d
                dist_vec = next_pos - closest_point
                dist = np.linalg.norm(dist_vec)

                # If colliding
                if dist < 5: # Sled half-height
                    on_surface = True
                    
                    # Correct position
                    if dist > 1e-9:
                        self.sled_pos = closest_point + (dist_vec / dist) * 5
                    else:
                        self.sled_pos = closest_point

                    # Adjust velocity
                    normal = np.array([-d[1], d[0]])
                    normal /= np.linalg.norm(normal)
                    
                    # Ensure normal points upwards
                    if normal[1] > 0:
                        normal *= -1
                    
                    vel_dot_normal = self.sled_vel @ normal
                    
                    # If moving into the surface, apply response
                    if vel_dot_normal > 0:
                        self.sled_vel -= vel_dot_normal * normal * 1.5 # Bounce factor
                    
                    # Apply friction and slide
                    tangent = d / np.linalg.norm(d)
                    vel_dot_tangent = self.sled_vel @ tangent
                    self.sled_vel = tangent * vel_dot_tangent
                    
                    # Add particles
                    if np.linalg.norm(self.sled_vel) > 1.0:
                        self._add_particles(5)
                    break

        if not on_surface:
            self.sled_pos = next_pos
            
        # Clean up old track points that are far behind the sled
        self.player_track = [p for p in self.player_track if p[0] > self.sled_pos[0] - self.WIDTH]


    def _add_particles(self, count):
        for _ in range(count):
            vel = self.np_random.standard_normal(size=2) * 0.5
            self.particles.append({
                "pos": self.sled_pos.copy(),
                "vel": vel - self.sled_vel * 0.1,
                "life": self.np_random.integers(10, 20),
                "size": self.np_random.uniform(1, 3)
            })
            
    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _calculate_reward_and_termination(self):
        terminated = False
        reward = 0

        # Reward for horizontal progress
        progress = self.sled_pos[0] - self.last_sled_x
        reward += progress * 0.1
        self.last_sled_x = self.sled_pos[0]
        
        # Check for win condition
        if self.sled_pos[0] >= self.FINISH_X:
            reward += 100
            terminated = True
            return reward, terminated

        # Check for crash (out of bounds)
        if not (0 < self.sled_pos[1] < self.HEIGHT):
            reward -= 10
            terminated = True
            return reward, terminated
        
        # The terrain collision check is removed as the sled now rides on the terrain.
        
        return reward, terminated
        
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Render a gradient from top to bottom
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Camera offset to keep sled in the middle of the screen
        cam_x = self.sled_pos[0] - self.WIDTH / 2
        
        # Draw terrain
        if len(self.terrain_points) > 1:
            screen_points = [(p[0] - cam_x, p[1]) for p in self.terrain_points]
            pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, screen_points, 2)
            
        # Draw player track
        if len(self.player_track) > 1:
            screen_points = [(p[0] - cam_x, p[1]) for p in self.player_track]
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, screen_points, 5)
            for p in screen_points:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 3, self.COLOR_TRACK)

        # Draw start and finish lines
        start_screen_x = self.START_X - cam_x
        finish_screen_x = self.FINISH_X - cam_x
        pygame.draw.line(self.screen, self.COLOR_START, (start_screen_x, 0), (start_screen_x, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 3)

        # Draw particles
        for p in self.particles:
            screen_pos = p['pos'] - np.array([cam_x, 0])
            alpha = int(255 * (p['life'] / 20.0))
            color = (*self.COLOR_PARTICLE, alpha)
            radius = int(p['size'] * (p['life'] / 20.0))
            if radius > 0:
                try:
                    # Use a temporary surface for alpha blending
                    target_rect = pygame.Rect(screen_pos[0]-radius, screen_pos[1]-radius, radius*2, radius*2)
                    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(s, radius, radius, radius, color)
                    self.screen.blit(s, target_rect)
                except (ValueError, TypeError):
                    pass # Ignore errors from particles going off-screen

        # Draw sled
        sled_screen_pos = self.sled_pos - np.array([cam_x, 0])
        sled_rect = pygame.Rect(0, 0, 15, 8)
        sled_rect.center = (int(sled_screen_pos[0]), int(sled_screen_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_SLED_HIGHLIGHT, sled_rect.inflate(-4,-4), border_radius=2)

        # Draw cursor
        pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 10, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (self.cursor_pos[0] - 5, self.cursor_pos[1]), (self.cursor_pos[0] + 5, self.cursor_pos[1]))
        pygame.draw.line(self.screen, self.COLOR_TEXT, (self.cursor_pos[0], self.cursor_pos[1] - 5), (self.cursor_pos[0], self.cursor_pos[1] + 5))
        
    def _render_ui(self):
        # Display speed
        speed = np.linalg.norm(self.sled_vel) * 10
        speed_text = self.font_small.render(f"Speed: {speed:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (10, 10))

        # Display time
        time_text = self.font_small.render(f"Time: {self.time_elapsed:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        if self.game_over:
            reason = ""
            if self.sled_pos[0] >= self.FINISH_X:
                reason = "FINISH!"
            elif self.steps >= self.MAX_STEPS:
                reason = "TIME UP"
            else:
                reason = "CRASHED"
            
            end_text = self.font_large.render(reason, True, self.COLOR_SLED)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_x": self.sled_pos[0],
            "time": self.time_elapsed,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # In this mode, we want to see the screen.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    running = True
    
    # Main game loop
    while running:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            
            # Map keys to actions
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Control human play speed

    env.close()