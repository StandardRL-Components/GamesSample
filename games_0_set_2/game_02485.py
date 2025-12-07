
# Generated: 2025-08-28T04:59:58.795161
# Source Brief: brief_02485.md
# Brief Index: 2485

        
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


# Helper class for the sled
class Sled:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.size = 10
        self.on_ground = False
        self.angle = 0

    def draw(self, surface):
        # Draw a rotated rectangle for the sled for better visual feel
        points = [
            (-self.size, -self.size / 2),
            (self.size, -self.size / 2),
            (self.size * 0.8, self.size / 2),
            (-self.size, self.size / 2),
        ]
        rotated_points = [pygame.Vector2(p).rotate(-self.angle) + self.pos for p in points]
        pygame.draw.polygon(surface, (255, 255, 255), rotated_points)
        pygame.draw.polygon(surface, (200, 200, 255), rotated_points, 2) # Outline

    def update(self, track_segments, gravity):
        # Apply gravity if not on a surface
        if not self.on_ground:
            self.vel.y += gravity

        # Limit max speed to prevent physics explosions
        if self.vel.length() > 20:
            self.vel.scale_to_length(20)

        predicted_pos = self.pos + self.vel

        self.on_ground = False
        best_segment = None
        min_dist_sq = float('inf')
        
        # Find the closest track segment below the sled
        for seg_start, seg_end in track_segments:
            seg_vec = seg_end - seg_start
            if seg_vec.length_squared() == 0: continue

            # Check if sled is horizontally within the segment's bounds
            min_x, max_x = sorted((seg_start.x, seg_end.x))
            if not (min_x - self.size <= predicted_pos.x <= max_x + self.size):
                continue
            
            # Calculate line's y at sled's x
            line_y = seg_start.y + (predicted_pos.x - seg_start.x) * seg_vec.y / seg_vec.x if seg_vec.x != 0 else seg_start.y

            # Check if sled is about to cross the line from above
            if predicted_pos.y >= line_y - 5 and self.pos.y < line_y + 5: # Tolerance
                dist_sq = (predicted_pos - pygame.Vector2(predicted_pos.x, line_y)).length_squared()
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_segment = (seg_start, seg_end)

        if best_segment and min_dist_sq < (self.size * 2)**2:
            seg_start, seg_end = best_segment
            seg_vec = seg_end - seg_start
            
            # Snap to track
            line_y = seg_start.y + (predicted_pos.x - seg_start.x) * seg_vec.y / seg_vec.x if seg_vec.x != 0 else seg_start.y
            self.pos.x = predicted_pos.x
            self.pos.y = line_y
            self.on_ground = True

            # Align velocity with track
            track_angle_rad = math.atan2(seg_vec.y, seg_vec.x)
            self.angle = math.degrees(track_angle_rad)
            
            speed = self.vel.length()
            # Project gravity onto the slope to affect acceleration
            gravity_force_on_slope = gravity * math.sin(track_angle_rad)
            speed += gravity_force_on_slope

            # Apply friction
            friction = 0.995
            speed *= friction
            
            self.vel = pygame.Vector2(math.cos(track_angle_rad), math.sin(track_angle_rad)) * speed
        else:
            # Free fall
            self.pos = predicted_pos
            self.angle = math.degrees(math.atan2(self.vel.y, self.vel.x)) if self.vel.length() > 0.1 else 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to draw short lines. Space to draw a longer line along velocity. Shift to draw a curved line."
    )

    game_description = (
        "A physics-based puzzle game where you draw tracks for a sled to reach the finish line. Inspired by Line Rider."
    )

    auto_advance = False
    
    # Class-level state for difficulty progression across resets
    _successful_episodes = 0
    _base_finish_x = 200.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Constants ---
        self.GRAVITY = 0.2
        self.PHYSICS_SUBSTEPS = 10
        self.MAX_STEPS = 1000
        self.SPEED_THRESHOLD = 1.0
        self.LINE_LENGTH = 15
        self.LONG_LINE_LENGTH = 30
        
        # --- Colors ---
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_TRACK = (0, 0, 0)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_START = (0, 255, 0)
        self.COLOR_FINISH = (255, 0, 0)
        self.COLOR_PARTICLE = (200, 200, 255)
        self.COLOR_UI = (220, 220, 240)

        # --- Fonts ---
        try:
            self.ui_font = pygame.font.SysFont("Consolas", 18)
            self.large_font = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.ui_font = pygame.font.Font(None, 22)
            self.large_font = pygame.font.Font(None, 56)

        # --- Game State (initialized in reset) ---
        self.sled = None
        self.track_segments = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.finish_x = 0
        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty progression based on class variable
        difficulty_increase = (GameEnv._successful_episodes // 5) * 10
        self.finish_x = GameEnv._base_finish_x + difficulty_increase
        self.finish_x = min(self.finish_x, self.width - 40)

        # Initialize game objects
        self.sled = Sled(x=80, y=self.height / 2)
        self.track_segments = [
            (pygame.Vector2(40, self.height / 2 + 20), pygame.Vector2(120, self.height / 2 + 20))
        ] # Initial platform
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_action(movement, space_held, shift_held)
        
        reward = self._simulate_physics()

        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Max steps reached
            reward -= 1.0 # Penalty for running out of time
            self.score -= 1.0
        
        # If victory, increment the class-level counter
        if self.game_over and self.sled.pos.x >= self.finish_x:
            GameEnv._successful_episodes += 1
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, movement, space_held, shift_held):
        start_point = self.sled.pos.copy()
        end_point = None
        
        # Actions are prioritized: shift > space > movement
        if shift_held:
            # Draw a short downward arc (approximated as two line segments)
            if self.sled.vel.length() > 0.1:
                direction = self.sled.vel.normalize()
                p1 = start_point + direction.rotate(25) * self.LONG_LINE_LENGTH / 2
                p2 = p1 + direction.rotate(50) * self.LONG_LINE_LENGTH / 2
                self.track_segments.append((start_point, p1))
                self.track_segments.append((p1, p2))
                # Sound effect placeholder: # sfx_draw_arc()
        elif space_held:
            # Draw a longer line in the direction of velocity
            if self.sled.vel.length() > 0.1:
                direction = self.sled.vel.normalize()
                end_point = start_point + direction * self.LONG_LINE_LENGTH
                # Sound effect placeholder: # sfx_draw_long()
        elif movement != 0:
            # Draw short cardinal lines
            if movement == 1: # Up
                end_point = start_point + pygame.Vector2(0, -self.LINE_LENGTH)
            elif movement == 2: # Down
                end_point = start_point + pygame.Vector2(0, self.LINE_LENGTH)
            elif movement == 3: # Left
                end_point = start_point + pygame.Vector2(-self.LINE_LENGTH, 0)
            elif movement == 4: # Right
                end_point = start_point + pygame.Vector2(self.LINE_LENGTH, 0)
            # Sound effect placeholder: # sfx_draw_short()

        if end_point:
            self.track_segments.append((start_point, end_point))

    def _simulate_physics(self):
        reward = 0
        start_x = self.sled.pos.x

        for _ in range(self.PHYSICS_SUBSTEPS):
            if self.game_over: break
            
            self.sled.update(self.track_segments, self.GRAVITY)
            self._update_particles()
            
            # Check for termination conditions
            if not (0 < self.sled.pos.x < self.width and 0 < self.sled.pos.y < self.height):
                self.game_over = True
                reward = -1.0 # Crash penalty
                # Sound effect placeholder: # sfx_crash()
                break
            if self.sled.pos.x >= self.finish_x:
                self.game_over = True
                reward = 10.0 # Victory reward
                # Sound effect placeholder: # sfx_win()
                break
        
        # Calculate non-terminal rewards if game is not over
        if not self.game_over:
            # Reward for forward progress, scaled by distance
            progress = self.sled.pos.x - start_x
            reward += progress * 0.1
            
            # Penalty for being slow
            if self.sled.vel.length() < self.SPEED_THRESHOLD:
                reward -= 0.01
                
        return reward

    def _update_particles(self):
        # Add new particle for a trail effect
        if self.steps % 2 == 0 and self.sled.vel.length() > 1.0:
            p_pos = self.sled.pos.copy()
            p_vel = self.sled.vel.normalize().rotate(self.np_random.uniform(-15, 15)) * -1
            self.particles.append({
                "pos": p_pos, "vel": p_vel, "life": 20, "size": self.np_random.integers(2, 5)
            })

        # Update existing particles (move, shrink, fade)
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] *= 0.95
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw start and finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (80, 0), (80, self.height), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.height), 3)

        # Draw track
        for start, end in self.track_segments:
            pygame.draw.line(self.screen, self.COLOR_TRACK, start, end, 5)
        
        # Draw particles
        for p in self.particles:
            # Use gfxdraw for anti-aliased circles
            alpha = int(p["life"] * 12)
            if alpha > 0 and p["size"] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]),
                    (*self.COLOR_PARTICLE, alpha)
                )

        # Draw sled
        self.sled.draw(self.screen)
        
        # Draw game over text
        if self.game_over:
            text = "FINISH!" if self.sled.pos.x >= self.finish_x else "CRASHED!"
            text_surf = self.large_font.render(text, True, self.COLOR_UI)
            text_rect = text_surf.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Speedometer
        speed = self.sled.vel.length()
        speed_text = f"Speed: {speed:.1f}"
        speed_surf = self.ui_font.render(speed_text, True, self.COLOR_UI)
        self.screen.blit(speed_surf, (10, 10))

        # Score
        score_text = f"Score: {self.score:.2f}"
        score_surf = self.ui_font.render(score_text, True, self.COLOR_UI)
        score_rect = score_surf.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(score_surf, score_rect)

        # Steps
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.ui_font.render(steps_text, True, self.COLOR_UI)
        steps_rect = steps_surf.get_rect(topright=(self.width - 10, 30))
        self.screen.blit(steps_surf, steps_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled.pos.x, self.sled.pos.y),
            "sled_vel": (self.sled.vel.x, self.sled.vel.y),
            "finish_x": self.finish_x,
            "successful_episodes": GameEnv._successful_episodes
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
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=0)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    running = True
    terminated = False
    total_reward = 0
    
    pygame.display.set_caption("Line Rider Gym - Manual Test")
    display_screen = pygame.display.set_mode((env.width, env.height))

    while running:
        action = [0, 0, 0] # Default to no-op
        
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    total_reward = 0
                    terminated = False
                    print("\n--- Episode Reset ---")
                # Step on any key press for drawing
                if not terminated:
                    keys = pygame.key.get_pressed()
                    mov = 0
                    if keys[pygame.K_UP]: mov = 1
                    elif keys[pygame.K_DOWN]: mov = 2
                    elif keys[pygame.K_LEFT]: mov = 3
                    elif keys[pygame.K_RIGHT]: mov = 4
                    
                    space = 1 if keys[pygame.K_SPACE] else 0
                    shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                    action = [mov, space, shift]

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

        # --- Rendering ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()