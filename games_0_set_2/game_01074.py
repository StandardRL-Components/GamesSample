# Generated: 2025-08-27T15:46:48.018428
# Source Brief: brief_01074.md
# Brief Index: 1074

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to select track type (straight, sloped, curved). Hold Shift for a boost pad, or press Space for a jump ramp. Guide the rider to the red finish line!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game where you build a track in real-time to guide a rider to the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 200
        self.GRAVITY = 0.25
        self.RIDER_RADIUS = 8
        self.SEGMENT_LENGTH = 40
        self.PHYSICS_SUBSTEPS = 8

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_RIDER = (50, 150, 255)
        self.COLOR_RIDER_ACCENT = (255, 255, 255)
        self.COLOR_TRACK = (240, 240, 240)
        self.COLOR_BOOST = (255, 200, 0)
        self.COLOR_JUMP = (200, 50, 255)
        self.COLOR_START = (0, 255, 100)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

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
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Initialize state variables
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_angle = 0
        self.rider_on_ground = False
        self.tracks = []
        self.particles = []
        self.last_track_endpoint = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.max_x_reached = 0
        self.cleared_sections = set()
        self.game_over = False
        self.win_condition = False
        self.rng = None

        self.reset()
        # self.validate_implementation() # Commented out for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback to a default generator if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.rider_pos = pygame.Vector2(50, 100)
        self.rider_vel = pygame.Vector2(1.5, 0)
        self.rider_angle = 0
        self.rider_on_ground = True
        self.max_x_reached = self.rider_pos.x
        self.cleared_sections = set()

        self.particles = []
        self.tracks = []
        self.last_track_endpoint = pygame.Vector2(10, 150)
        
        # Create initial platform
        for i in range(4):
            start = self.last_track_endpoint.copy()
            end = start + pygame.Vector2(self.SEGMENT_LENGTH, 0)
            self.tracks.append({"start": start, "end": end, "type": "normal"})
            self.last_track_endpoint = end
        
        self.finish_line_x = self.WIDTH - 50

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Action: Place new track segment
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._add_track_segment(movement, space_held, shift_held)

        # 2. Simulate Physics
        self._run_physics()

        # 3. Update Game State
        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _add_track_segment(self, movement, space_held, shift_held):
        # # Sound effect placeholder:
        # pygame.mixer.Sound("sounds/place_track.wav").play()

        start_pos = self.last_track_endpoint.copy()
        
        if shift_held: # Priority 1: Boost Pad
            end_pos = start_pos + pygame.Vector2(self.SEGMENT_LENGTH, 0)
            self.tracks.append({"start": start_pos, "end": end_pos, "type": "boost"})
            self.last_track_endpoint = end_pos
        elif space_held: # Priority 2: Jump Ramp
            end_pos = start_pos + pygame.Vector2(self.SEGMENT_LENGTH * 0.7, -self.SEGMENT_LENGTH * 0.7)
            self.tracks.append({"start": start_pos, "end": end_pos, "type": "jump"})
            self.last_track_endpoint = end_pos
        else: # Default: Use movement action
            angle = 0
            if movement == 1: angle = -math.pi / 6   # Up
            elif movement == 2: angle = math.pi / 6  # Down
            
            if movement == 3: # Curve Up (CCW)
                curve_res = 5
                radius = self.SEGMENT_LENGTH * 1.5
                for i in range(curve_res):
                    a1 = -math.pi / (curve_res*2) * i
                    a2 = -math.pi / (curve_res*2) * (i + 1)
                    p1 = start_pos + pygame.Vector2(math.sin(a1), -math.cos(a1)) * radius
                    p2 = start_pos + pygame.Vector2(math.sin(a2), -math.cos(a2)) * radius
                    self.tracks.append({"start": p1, "end": p2, "type": "normal"})
                self.last_track_endpoint = self.tracks[-1]["end"]
            elif movement == 4: # Curve Down (CW)
                curve_res = 5
                radius = self.SEGMENT_LENGTH * 1.5
                for i in range(curve_res):
                    a1 = math.pi / (curve_res*2) * i
                    a2 = math.pi / (curve_res*2) * (i + 1)
                    p1 = start_pos + pygame.Vector2(math.sin(a1), math.cos(a1)) * radius
                    p2 = start_pos + pygame.Vector2(math.sin(a2), math.cos(a2)) * (i + 1)
                    self.tracks.append({"start": p1, "end": p2, "type": "normal"})
                self.last_track_endpoint = self.tracks[-1]["end"]
            else: # Straight segments (0, 1, 2)
                end_pos = start_pos + pygame.Vector2(self.SEGMENT_LENGTH * math.cos(angle), self.SEGMENT_LENGTH * math.sin(angle))
                self.tracks.append({"start": start_pos, "end": end_pos, "type": "normal"})
                self.last_track_endpoint = end_pos
        
        # Cull old tracks to prevent performance degradation
        if len(self.tracks) > 100:
            self.tracks.pop(0)

    def _run_physics(self):
        for _ in range(self.PHYSICS_SUBSTEPS):
            substep_dt = 1.0 / self.PHYSICS_SUBSTEPS

            # Apply gravity
            if not self.rider_on_ground:
                self.rider_vel.y += self.GRAVITY * substep_dt
            
            self.rider_pos += self.rider_vel * substep_dt

            # Collision detection and response
            self.rider_on_ground = False
            collided_track = self._check_collisions()

            if collided_track:
                self.rider_on_ground = True
                track_vec = collided_track["end"] - collided_track["start"]
                track_angle = math.atan2(track_vec.y, track_vec.x)
                track_normal = pygame.Vector2(-track_vec.y, track_vec.x).normalize()

                # Project rider velocity onto track direction
                speed = self.rider_vel.length()
                self.rider_vel = track_vec.normalize() * speed

                # Apply track-based acceleration/gravity
                gravity_on_slope = self.GRAVITY * math.sin(track_angle)
                self.rider_vel += track_vec.normalize() * gravity_on_slope * substep_dt
                
                # Friction
                self.rider_vel *= (1.0 - 0.01 * substep_dt)

                # Update rider angle
                self.rider_angle = math.degrees(-track_angle)

                # Handle special tracks
                if collided_track["type"] == 'boost':
                    self.rider_vel *= 1.05
                    # # Sound effect placeholder:
                    # pygame.mixer.Sound("sounds/boost.wav").play()
                    if self.rng.random() < 0.5:
                        self.particles.append(self._create_particle(self.COLOR_BOOST, 1.5))
                elif collided_track["type"] == 'jump':
                    self.rider_vel += track_normal * 4
                    self.rider_on_ground = False
                    # # Sound effect placeholder:
                    # pygame.mixer.Sound("sounds/jump.wav").play()
            
            # Update particles
            for p in self.particles[:]:
                p['pos'] += p['vel']
                p['life'] -= 1
                if p['life'] <= 0:
                    self.particles.remove(p)

    def _check_collisions(self):
        closest_track = None
        min_dist_sq = float('inf')

        for track in self.tracks:
            p1, p2 = track["start"], track["end"]
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider_pos - p1
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t)) # Clamp to segment
            
            closest_point = p1 + t * line_vec
            dist_sq = self.rider_pos.distance_squared_to(closest_point)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_track = track

        if min_dist_sq < self.RIDER_RADIUS ** 2:
            # Collision occurred, resolve penetration
            p1, p2 = closest_track["start"], closest_track["end"]
            line_vec = p2 - p1
            if line_vec.length_squared() > 0:
                closest_point = p1 + max(0, min(1, (self.rider_pos - p1).dot(line_vec) / line_vec.length_squared())) * line_vec
                
                penetration_vec = self.rider_pos - closest_point
                if penetration_vec.length_squared() > 0:
                    dist = penetration_vec.length()
                    correction = (self.RIDER_RADIUS - dist) * penetration_vec.normalize()
                    self.rider_pos += correction
                    
                    # Dampen velocity perpendicular to surface after collision
                    normal = penetration_vec.normalize()
                    self.rider_vel -= normal * self.rider_vel.dot(normal) * 0.5
            
            # Spawn landing particles
            if not self.rider_on_ground and self.rng.random() < 0.1:
                for _ in range(3): self.particles.append(self._create_particle(self.COLOR_TRACK, 0.8))

            return closest_track
        return None

    def _calculate_reward(self):
        reward = 0
        
        # Reward for forward progress
        progress = self.rider_pos.x - self.max_x_reached
        if progress > 0:
            reward += progress * 0.1
            self.max_x_reached = self.rider_pos.x
        else:
            reward += progress * 0.05 # Small penalty for moving backward

        # Reward for clearing sections
        section_width = self.WIDTH / 3
        current_section = int(self.rider_pos.x // section_width)
        if current_section not in self.cleared_sections and current_section > 0:
            self.cleared_sections.add(current_section)
            reward += 5
            self.score += 5

        # Check terminal conditions for major rewards/penalties
        if self.rider_pos.x >= self.finish_line_x:
            reward += 100
            self.score += 100
        elif self.rider_pos.y > self.HEIGHT + self.RIDER_RADIUS or self.steps >= self.MAX_STEPS:
            reward -= 100
            self.score -= 100

        return reward

    def _check_termination(self):
        if self.rider_pos.x >= self.finish_line_x:
            self.game_over = True
            self.win_condition = True
            # # Sound effect placeholder:
            # pygame.mixer.Sound("sounds/win.wav").play()
        elif self.rider_pos.y > self.HEIGHT + self.RIDER_RADIUS:
            self.game_over = True
            # # Sound effect placeholder:
            # pygame.mixer.Sound("sounds/fall.wav").play()
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            # # Sound effect placeholder:
            # pygame.mixer.Sound("sounds/timeout.wav").play()
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw start and finish lines
        pygame.draw.rect(self.screen, self.COLOR_START, (10, 0, 5, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0, 5, self.HEIGHT))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['size'], p['size']))

        # Draw tracks
        for track in self.tracks:
            color = self.COLOR_TRACK
            if track["type"] == "boost": color = self.COLOR_BOOST
            elif track["type"] == "jump": color = self.COLOR_JUMP
            pygame.draw.line(self.screen, color, track["start"], track["end"], 4)

        # Draw rider
        rider_surf = pygame.Surface((self.RIDER_RADIUS*2, self.RIDER_RADIUS*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(rider_surf, self.RIDER_RADIUS, self.RIDER_RADIUS, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(rider_surf, self.RIDER_RADIUS, self.RIDER_RADIUS, self.RIDER_RADIUS, self.COLOR_RIDER)
        
        # Draw accent to show rotation
        angle_rad = math.radians(-self.rider_angle)
        accent_pos = (
            self.RIDER_RADIUS + math.cos(angle_rad) * self.RIDER_RADIUS * 0.6,
            self.RIDER_RADIUS + math.sin(angle_rad) * self.RIDER_RADIUS * 0.6
        )
        pygame.gfxdraw.filled_circle(rider_surf, int(accent_pos[0]), int(accent_pos[1]), int(self.RIDER_RADIUS * 0.3), self.COLOR_RIDER_ACCENT)
        
        rotated_rider = pygame.transform.rotate(rider_surf, self.rider_angle)
        rider_rect = rotated_rider.get_rect(center=self.rider_pos)
        self.screen.blit(rotated_rider, rider_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        timer_text = self.font_main.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = self.COLOR_START if self.win_condition else self.COLOR_FINISH
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y)
        }
    
    def _create_particle(self, color, speed_mult):
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(1, 3) * speed_mult
        return {
            'pos': self.rider_pos.copy(),
            'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
            'life': self.rng.integers(10, 30),
            'max_life': 30,
            'color': color,
            'size': self.rng.integers(2, 5)
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    # For manual play, we want a window.
    os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Line Rider Gym Env")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample() # Start with a random action
    action.fill(0) # Default to no-op

    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")


    while not done:
        # Human controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0

        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(10) # Control the speed of the manual play

    print(f"Game Over! Final Score: {info['score']}")
    
    # Keep the window open for a few seconds after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)

    env.close()