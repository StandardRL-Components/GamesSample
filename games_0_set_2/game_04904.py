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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to draw a solid track. Hold Space + arrows to draw a boost track."
    )

    # User-facing description of the game
    game_description = (
        "Draw a track for your sled to reach the finish line. Use boost pads strategically to beat the clock!"
    )

    # Frames auto-advance for real-time physics
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000
    GAME_DURATION_SECONDS = 20

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_SLED = (255, 255, 255)
    COLOR_SLED_GLOW = (200, 200, 255, 60)
    COLOR_TRACK_SOLID = (50, 150, 255)
    COLOR_TRACK_BOOST = (255, 150, 50)
    COLOR_START_LINE = (100, 255, 100)
    COLOR_FINISH_LINE = (255, 100, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Physics
    GRAVITY = 0.15
    FRICTION = 0.995
    BOOST_FORCE = 0.5
    SLIDE_FORCE = 0.05
    TRACK_SEGMENT_LENGTH = 40
    DRAW_COOLDOWN = 5  # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Internal state variables
        self.sled_pos = None
        self.sled_vel = None
        self.sled_angle = None
        self.track_segments = None
        self.last_track_point = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.last_draw_step = -self.DRAW_COOLDOWN

        # These are defined in reset, but need to exist for __init__
        self.start_pos = pygame.math.Vector2(80, 100)
        self.finish_x = self.SCREEN_WIDTH - 80

        # Self-check to ensure API compliance - call after reset
        # self.validate_implementation() # This is better called outside after init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        # Initialize game state
        self.start_pos = pygame.math.Vector2(80, 100)
        self.finish_x = self.SCREEN_WIDTH - 80

        self.sled_pos = pygame.math.Vector2(self.start_pos)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.sled_angle = 0

        # Initial flat track segment
        initial_track_start = self.start_pos - (self.TRACK_SEGMENT_LENGTH / 2, 0)
        initial_track_end = self.start_pos + (self.TRACK_SEGMENT_LENGTH / 2, 0)
        self.track_segments = [{
            "start": initial_track_start,
            "end": initial_track_end,
            "type": "solid",
            "angle": 0,
            "normal": pygame.math.Vector2(0, -1)
        }]
        self.last_track_point = initial_track_end

        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        self.last_draw_step = -self.DRAW_COOLDOWN

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- 1. Handle Action ---
        movement = action[0]
        space_held = action[1] == 1
        
        can_draw = self.steps >= self.last_draw_step + self.DRAW_COOLDOWN

        if movement != 0 and can_draw:
            self.last_draw_step = self.steps
            
            # Map movement to angles: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
            # In pygame, positive y is down, so UP is -pi/2, DOWN is pi/2.
            angle_map = {1: -math.pi / 2, 2: math.pi / 2, 3: math.pi, 4: 0}
            angle_rad = angle_map[movement]
            
            new_point = self.last_track_point + pygame.math.Vector2(
                math.cos(angle_rad) * self.TRACK_SEGMENT_LENGTH,
                math.sin(angle_rad) * self.TRACK_SEGMENT_LENGTH,
            )
            
            # Clamp to screen bounds
            new_point.x = max(0, min(self.SCREEN_WIDTH, new_point.x))
            new_point.y = max(0, min(self.SCREEN_HEIGHT, new_point.y))

            track_type = "boost" if space_held else "solid"
            
            # Prevent creating zero-length segments
            if (new_point - self.last_track_point).length() > 1:
                segment_vec = new_point - self.last_track_point
                segment_angle = segment_vec.angle_to(pygame.math.Vector2(1, 0))
                segment_normal = segment_vec.rotate(90).normalize()

                self.track_segments.append({
                    "start": self.last_track_point,
                    "end": new_point,
                    "type": track_type,
                    "angle": math.radians(-segment_angle),
                    "normal": segment_normal,
                })
                self.last_track_point = new_point

                reward += 5 if track_type == "boost" else -1
        
        # --- 2. Update Physics & Game State ---
        self.steps += 1
        self.time_remaining -= 1

        # Sled physics
        self._update_sled_physics()
        if hasattr(self, 'on_ground') and self.on_ground:
            reward += 0.1
        
        # Particle physics
        self._update_particles()
        
        # --- 3. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.sled_pos.x >= self.finish_x:
                reward += 100 # Win
            else:
                reward -= 100 # Lose

        # Update total score
        self.score += reward
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_sled_physics(self):
        self.on_ground = False
        
        # Apply gravity
        self.sled_vel.y += self.GRAVITY

        # Check for track collision
        for segment in reversed(self.track_segments):
            p1 = segment["start"]
            p2 = segment["end"]

            # Bounding box check for performance
            if not (min(p1.x, p2.x) - 5 < self.sled_pos.x < max(p1.x, p2.x) + 5):
                continue

            # Project sled position onto the line defined by the segment
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            t = (self.sled_pos - p1).dot(line_vec) / line_vec.length_squared()

            if 0 <= t <= 1:
                closest_point = p1 + t * line_vec
                dist_vec = self.sled_pos - closest_point
                
                # If sled is on or just through the track surface
                if dist_vec.dot(segment["normal"]) < 5 and dist_vec.length() < 10:
                    self.on_ground = True
                    
                    # Snap position to track
                    self.sled_pos = closest_point
                    
                    # Align velocity with track
                    track_angle = segment["angle"]
                    
                    # Sliding force
                    slide_force = self.SLIDE_FORCE * math.sin(track_angle)
                    self.sled_vel.x += math.cos(track_angle) * slide_force
                    
                    # Normal force cancels gravity component
                    gravity_on_normal = self.GRAVITY * math.cos(track_angle)
                    self.sled_vel.y -= gravity_on_normal * math.cos(track_angle)
                    
                    # Apply friction
                    self.sled_vel *= self.FRICTION

                    # Handle boost
                    if segment["type"] == "boost":
                        boost_vec = pygame.math.Vector2(1, 0).rotate(-math.degrees(track_angle)) * self.BOOST_FORCE
                        self.sled_vel += boost_vec
                        self._create_particles(10, self.COLOR_TRACK_BOOST, 2)
                    
                    # Update sled visual angle
                    self.sled_angle = math.degrees(-track_angle)
                    
                    # Create snow particles
                    self._create_particles(2, self.COLOR_SLED, 1)
                    break # Found ground, stop checking

        # Update position
        self.sled_pos += self.sled_vel
        
    def _create_particles(self, count, color, speed_mult):
        for _ in range(count):
            angle = math.radians(self.sled_angle) + math.pi + random.uniform(-0.3, 0.3)
            speed = self.sled_vel.length() * random.uniform(0.2, 0.5) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(10, 20)
            self.particles.append([pygame.math.Vector2(self.sled_pos), vel, random.randint(2, 4), lifespan, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[3] -= 1 # lifespan -= 1
        self.particles = [p for p in self.particles if p[3] > 0]

    def _check_termination(self):
        if self.game_over:
            return True
        
        off_screen = not (0 < self.sled_pos.x < self.SCREEN_WIDTH and -50 < self.sled_pos.y < self.SCREEN_HEIGHT + 50)
        timeout = self.time_remaining <= 0
        win = self.sled_pos.x >= self.finish_x

        if off_screen or timeout or win:
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
        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START_LINE, (self.start_pos.x, 0), (self.start_pos.x, self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (self.finish_x, 0), (self.finish_x, self.SCREEN_HEIGHT), 3)

        # Draw track segments
        for segment in self.track_segments:
            color = self.COLOR_TRACK_BOOST if segment["type"] == "boost" else self.COLOR_TRACK_SOLID
            pygame.draw.line(self.screen, color, segment["start"], segment["end"], 5)

        # Draw particles
        for pos, vel, size, life, color in self.particles:
            alpha = max(0, min(255, int(255 * (life / 20.0))))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(pos.x - size), int(pos.y - size)))

        # Draw sled
        self._draw_sled()

    def _draw_sled(self):
        # Sled shape: a triangle
        sled_size = 10
        points = [
            pygame.math.Vector2(sled_size, 0),
            pygame.math.Vector2(-sled_size / 2, -sled_size / 2),
            pygame.math.Vector2(-sled_size / 2, sled_size / 2),
        ]
        
        # Rotate points
        rotated_points = [p.rotate(self.sled_angle) + self.sled_pos for p in points]
        
        # Draw glow using a temporary surface for alpha blending
        glow_size = sled_size * 2
        temp_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        glow_points = [(p.x - self.sled_pos.x + glow_size, p.y - self.sled_pos.y + glow_size) for p in rotated_points]
        pygame.draw.polygon(temp_surf, self.COLOR_SLED_GLOW, glow_points)
        self.screen.blit(temp_surf, (self.sled_pos.x - glow_size, self.sled_pos.y - glow_size))
        
        # Draw main body
        pygame.draw.polygon(self.screen, self.COLOR_SLED, [(int(p.x), int(p.y)) for p in rotated_points])

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Timer
        time_str = f"TIME: {max(0, self.time_remaining / self.FPS):.1f}"
        text_surf = self.font_small.render(time_str, True, self.COLOR_TEXT)
        draw_text(time_str, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 10))
        
        # Speed
        speed = self.sled_vel.length() * 10
        speed_str = f"SPEED: {speed:.0f}"
        draw_text(speed_str, self.font_small, self.COLOR_TEXT, (15, self.SCREEN_HEIGHT - 30))

        # Score
        score_str = f"SCORE: {int(self.score)}"
        draw_text(score_str, self.font_small, self.COLOR_TEXT, (15, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": round(self.time_remaining / self.FPS, 2),
            "sled_speed": self.sled_vel.length() if self.sled_vel else 0,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sled Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

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

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(1000) # Pause before restarting

        clock.tick(GameEnv.FPS)
        
    env.close()