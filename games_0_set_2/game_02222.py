
# Generated: 2025-08-27T19:40:24.394809
# Source Brief: brief_02222.md
# Brief Index: 2222

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrows change the angle of the next track piece. "
        "Hold Space to draw a longer piece. Hold Shift to create a boost pad."
    )

    # User-facing description of the game
    game_description = (
        "Draw a track in real-time for a physics-based sled. "
        "Race to the finish line before time runs out, but be careful not to crash!"
    )

    # Frames advance on action receipt
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 600  # 20 steps/sec * 30 seconds

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (50, 55, 60)
    COLOR_TRACK = (220, 220, 230)
    COLOR_BOOST_TRACK = (255, 220, 0)
    COLOR_SLED = (255, 60, 60)
    COLOR_SLED_GLOW = (255, 60, 60, 50)
    COLOR_START = (0, 200, 80)
    COLOR_FINISH = (80, 80, 255)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics
    GRAVITY = 0.15
    FRICTION = 0.995
    BOOST_FORCE = 1.5
    NORMAL_FORCE_MULTIPLIER = 0.2
    SLED_WIDTH = 20
    SLED_HEIGHT = 8
    PHYSICS_SUBSTEPS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sled_pos = None
        self.sled_vel = None
        self.sled_angle = None
        self.track_points = None
        self.particles = None
        self.rng = None
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Use a default generator if no seed is provided
            if not hasattr(self, 'rng') or self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.sled_pos = pygame.Vector2(50, 100)
        self.sled_vel = pygame.Vector2(2, 0)
        self.sled_angle = 0
        
        # Initial flat track
        self.track_points = [
            {"pos": pygame.Vector2(0, 120), "type": "normal"},
            {"pos": pygame.Vector2(80, 120), "type": "normal"}
        ]
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Action ---
        self._handle_action(action)
        
        # --- 2. Update Physics ---
        prev_pos_x = self.sled_pos.x
        for _ in range(self.PHYSICS_SUBSTEPS):
            self._update_physics(1 / self.PHYSICS_SUBSTEPS)
        
        # --- 3. Update Particles & Game State ---
        self._update_particles()
        self.steps += 1
        
        # --- 4. Calculate Reward ---
        forward_velocity = self.sled_vel.x
        reward = 0.01 * forward_velocity
        self.score += reward

        # --- 5. Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.sled_pos.x >= self.SCREEN_WIDTH - 20: # Win
                reward += 100
                self.score += 100
            else: # Crash
                reward -= 10
                self.score -= 10
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        last_point = self.track_points[-1]["pos"]
        
        # Determine angle
        if movement == 0: angle_deg = 0  # None
        elif movement == 1: angle_deg = 30 # Up
        elif movement == 2: angle_deg = -30 # Down
        elif movement == 3: angle_deg = -15 # Gentle Down
        else: angle_deg = 15 # Gentle Up
        
        # Determine length
        length = 40 if space_held else 20
        
        # Determine type
        track_type = "boost" if shift_held else "normal"
        
        # Create new point
        angle_rad = math.radians(angle_deg)
        new_point_pos = last_point + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * length
        
        # Clamp to screen bounds to prevent drawing off-screen
        new_point_pos.x = max(0, min(self.SCREEN_WIDTH, new_point_pos.x))
        new_point_pos.y = max(0, min(self.SCREEN_HEIGHT, new_point_pos.y))

        self.track_points.append({"pos": new_point_pos, "type": track_type})

    def _update_physics(self, dt):
        # Apply gravity
        self.sled_vel.y += self.GRAVITY * dt * self.PHYSICS_SUBSTEPS

        # Find current track segment
        on_track = False
        current_segment_index = -1
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]["pos"]
            p2 = self.track_points[i + 1]["pos"]
            if p1.x <= self.sled_pos.x < p2.x or p2.x <= self.sled_pos.x < p1.x:
                current_segment_index = i
                break

        if current_segment_index != -1:
            p1 = self.track_points[current_segment_index]["pos"]
            p2 = self.track_points[current_segment_index + 1]["pos"]
            segment_vec = p2 - p1
            
            if segment_vec.length() > 0:
                # Interpolate track height at sled's x position
                t = (self.sled_pos.x - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
                track_y = p1.y + t * (p2.y - p1.y)

                # Collision detection and response
                if self.sled_pos.y > track_y - self.SLED_HEIGHT / 2:
                    on_track = True
                    self.sled_pos.y = track_y - self.SLED_HEIGHT / 2
                    
                    # Normal force
                    normal = segment_vec.rotate(90).normalize()
                    if normal.y > 0: normal = -normal # Ensure normal points up
                    
                    proj = self.sled_vel.dot(normal)
                    if proj > 0:
                        self.sled_vel -= normal * proj * self.NORMAL_FORCE_MULTIPLIER

                    # Friction
                    self.sled_vel *= (self.FRICTION ** (dt * self.PHYSICS_SUBSTEPS))

                    # Align sled to track
                    self.sled_angle = -math.degrees(math.atan2(segment_vec.y, segment_vec.x))
                    
                    # Handle boost pads
                    if self.track_points[current_segment_index]["type"] == "boost":
                        # sfx: boost_sound()
                        boost_dir = segment_vec.normalize()
                        self.sled_vel += boost_dir * self.BOOST_FORCE * dt * self.PHYSICS_SUBSTEPS

        # Update position
        self.sled_pos += self.sled_vel * dt * self.PHYSICS_SUBSTEPS

        # Emit particles if moving
        if self.sled_vel.length() > 1 and on_track:
            self._emit_particle()

    def _emit_particle(self):
        if len(self.particles) < 100:
            particle_pos = self.sled_pos.copy()
            particle_vel = pygame.Vector2(self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.5, 0.5)) - self.sled_vel * 0.1
            particle_life = self.rng.integers(20, 40)
            particle_radius = self.rng.uniform(2, 4)
            self.particles.append([particle_pos, particle_vel, particle_life, particle_radius])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[2] -= 1 # life -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _check_termination(self):
        return (
            self.sled_pos.y > self.SCREEN_HEIGHT + 20 or
            self.sled_pos.y < -20 or
            self.sled_pos.x < -20 or
            self.sled_pos.x >= self.SCREEN_WIDTH - 20 or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_start_finish()
        self._render_track()
        self._render_particles()
        self._render_sled()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "sled_speed": self.sled_vel.length(),
        }

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_start_finish(self):
        pygame.draw.line(self.screen, self.COLOR_START, (10, 0), (10, self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.SCREEN_WIDTH - 10, 0), (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT), 3)

    def _render_track(self):
        if len(self.track_points) < 2: return
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]["pos"]
            p2 = self.track_points[i+1]["pos"]
            color = self.COLOR_BOOST_TRACK if self.track_points[i+1]["type"] == 'boost' else self.COLOR_TRACK
            
            pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 4)
            pygame.gfxdraw.filled_circle(self.screen, int(p1.x), int(p1.y), 2, color)
        
        # Draw last point
        last_p = self.track_points[-1]["pos"]
        pygame.gfxdraw.filled_circle(self.screen, int(last_p.x), int(last_p.y), 2, self.COLOR_TRACK)


    def _render_sled(self):
        # Glow effect
        glow_radius = int(self.SLED_WIDTH * 0.8 + self.sled_vel.length() * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_SLED_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.sled_pos.x - glow_radius), int(self.sled_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Sled body
        sled_rect = pygame.Rect(0, 0, self.SLED_WIDTH, self.SLED_HEIGHT)
        sled_surface = pygame.Surface(sled_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(sled_surface, self.COLOR_SLED, sled_rect, border_radius=3)
        
        rotated_surface = pygame.transform.rotate(sled_surface, self.sled_angle)
        rotated_rect = rotated_surface.get_rect(center=(int(self.sled_pos.x), int(self.sled_pos.y)))
        self.screen.blit(rotated_surface, rotated_rect)

    def _render_particles(self):
        for pos, vel, life, radius in self.particles:
            alpha = int(255 * (life / 40))
            color = (*self.COLOR_SLED, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _render_ui(self):
        time_left = (self.MAX_STEPS - self.steps) / (self.MAX_STEPS / 30.0) # Assuming 20fps for 30s
        time_text = self.font_main.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (20, 10))
        
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        score_rect.top = 10
        self.screen.blit(score_text, score_rect)
        
        speed = self.sled_vel.length() * 10
        speed_text = self.font_main.render(f"SPEED: {int(speed)}", True, self.COLOR_TEXT)
        speed_rect = speed_text.get_rect(right=self.SCREEN_WIDTH - 20, top=10)
        self.screen.blit(speed_text, speed_rect)

        if self.game_over:
            status_text = "FINISH!" if self.sled_pos.x >= self.SCREEN_WIDTH - 20 else "CRASHED!"
            status_color = self.COLOR_FINISH if status_text == "FINISH!" else self.COLOR_SLED
            
            large_font = pygame.font.SysFont("Consolas", 60, bold=True)
            end_text_surf = large_font.render(status_text, True, status_color)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == "__main__":
    # To run with display, you need to unset the dummy video driver
    import os
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # None
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [move_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(20) # Control the speed of human play

    env.close()