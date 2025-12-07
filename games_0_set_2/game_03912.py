
# Generated: 2025-08-28T00:50:52.156266
# Source Brief: brief_03912.md
# Brief Index: 3912

        
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

    user_guide = (
        "Use arrow keys to draw track segments at different angles relative to the sled. "
        "Spacebar draws a long flat piece, Shift draws a gentle downward slope."
    )

    game_description = (
        "A physics-based puzzle game. Draw tracks in real-time to guide the sled from the start to the finish line."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (220, 220, 230)
    COLOR_TRACK = (80, 80, 90)
    COLOR_SLED = (20, 20, 20)
    COLOR_SLED_FLAG = (255, 80, 80)
    COLOR_START = (100, 200, 100)
    COLOR_FINISH = (200, 100, 100)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_SPEED_LINE = (100, 100, 255, 100) # RGBA
    COLOR_UI_TEXT = (50, 50, 50)
    
    # Physics
    GRAVITY = 0.15
    FRICTION = 0.995
    SLED_SIZE = (12, 6)
    MIN_SPEED_FOR_REWARD = 1.0
    MIN_SPEED_FOR_CRASH = 0.1
    STUCK_FRAMES_LIMIT = 90 # 3 seconds at 30fps
    
    # Gameplay
    MAX_STEPS = 1500
    FINISH_LINE_X = SCREEN_WIDTH - 40
    START_POS = (60, SCREEN_HEIGHT // 2)
    DRAW_DISTANCE_AHEAD = 20
    
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 48)

        # Initialize state variables
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_vel = pygame.Vector2(0, 0)
        self.sled_angle = 0.0
        self.track_lines = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.stuck_frames = 0
        self.game_over_message = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stuck_frames = 0
        self.game_over_message = ""
        
        self.sled_pos = pygame.Vector2(self.START_POS)
        self.sled_vel = pygame.Vector2(2, 0)
        self.sled_angle = 0.0
        
        # Initial platform
        start_platform_y = self.START_POS[1] + 20
        self.track_lines = [
            ((self.START_POS[0] - 50, start_platform_y), (self.START_POS[0] + 150, start_platform_y))
        ]
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_action(movement, space_held, shift_held)
        self._update_physics()
        
        self.steps += 1
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, movement, space_held, shift_held):
        # Define line properties based on action
        line_length = 20
        angle_offset = 0
        absolute_angle = None

        if space_held: # Priority action: long horizontal
            line_length = 40
            absolute_angle = 0
        elif shift_held: # Second priority: gentle downward slope
            line_length = 20
            absolute_angle = -math.radians(10)
        else:
            # Brief's Action 0 (no-op) -> absolute horizontal
            if movement == 0: 
                line_length = 15
                absolute_angle = 0
            # Brief's Action 1 (+45 deg) -> mapped to 'up'
            elif movement == 1: 
                angle_offset = math.radians(45)
            # Brief's Action 2 (-45 deg) -> mapped to 'down'
            elif movement == 2:
                angle_offset = -math.radians(45)
            # Brief's Action 4 (180 deg) -> mapped to 'left'
            elif movement == 3:
                angle_offset = math.radians(180)
            # Brief's Action 3 (0 deg) -> mapped to 'right'
            elif movement == 4:
                angle_offset = 0

        # Determine start point of the new line
        if self.sled_vel.magnitude() > 0.1:
            draw_direction = self.sled_vel.normalize()
        else: # If stopped, use orientation
            draw_direction = pygame.Vector2(math.cos(self.sled_angle), math.sin(self.sled_angle))
        
        start_point = self.sled_pos + draw_direction * self.DRAW_DISTANCE_AHEAD

        # Determine end point
        if absolute_angle is not None:
            end_point = start_point + pygame.Vector2(math.cos(absolute_angle), math.sin(absolute_angle)) * line_length
        else:
            final_angle = self.sled_angle + angle_offset
            end_point = start_point + pygame.Vector2(math.cos(final_angle), math.sin(final_angle)) * line_length
        
        # Clamp line to screen boundaries
        start_point.x = max(0, min(self.SCREEN_WIDTH, start_point.x))
        start_point.y = max(0, min(self.SCREEN_HEIGHT, start_point.y))
        end_point.x = max(0, min(self.SCREEN_WIDTH, end_point.x))
        end_point.y = max(0, min(self.SCREEN_HEIGHT, end_point.y))

        # Add the new line if it has any length
        if (start_point - end_point).magnitude() > 1:
            self.track_lines.append(( (start_point.x, start_point.y), (end_point.x, end_point.y) ))
            # Optional: limit total number of lines to prevent performance issues
            if len(self.track_lines) > 200:
                self.track_lines.pop(1) # Keep initial platform

    def _update_physics(self):
        # Apply gravity
        self.sled_vel.y += self.GRAVITY
        
        # Collision detection and response
        collision_data = self._find_collision_line()
        
        if collision_data:
            line, line_y_at_sled = collision_data
            
            # Snap to line
            self.sled_pos.y = line_y_at_sled - self.SLED_SIZE[1] / 2
            
            # Get line vector and normal
            p1, p2 = pygame.Vector2(line[0]), pygame.Vector2(line[1])
            line_vec = p2 - p1
            if line_vec.magnitude() == 0: return

            # Project velocity onto line
            proj_vel = self.sled_vel.dot(line_vec.normalize()) * line_vec.normalize()
            self.sled_vel = proj_vel * self.FRICTION
            
            # Update sled angle to match line
            self.sled_angle = math.atan2(line_vec.y, line_vec.x)
        else:
            # In freefall, update angle based on velocity
            if self.sled_vel.magnitude() > 0.1:
                self.sled_angle = math.atan2(self.sled_vel.y, self.sled_vel.x)

        # Update position
        self.sled_pos += self.sled_vel
        
        # Update particles
        self._update_particles()
        # Spawn new particles
        if self.sled_vel.magnitude() > 1:
            #_sound_effect: whoosh_trail
            self.particles.append([pygame.Vector2(self.sled_pos), self.np_random.uniform(-0.5, 0.5, 2), self.np_random.integers(10, 20)])

    def _find_collision_line(self):
        # Find the highest valid line segment directly below the sled
        sled_bottom = self.sled_pos.y + self.SLED_SIZE[1] / 2
        
        best_candidate = None
        highest_y = float('inf')

        for line in self.track_lines:
            p1, p2 = line
            x1, y1 = p1
            x2, y2 = p2

            # Bounding box check for quick discard
            if not (min(x1, x2) <= self.sled_pos.x <= max(x1, x2)):
                continue

            # Handle vertical lines
            if abs(x1 - x2) < 1e-6:
                y_on_line = max(y1, y2)
            else:
                # Linear interpolation to find line Y at sled's X
                y_on_line = y1 + (y2 - y1) * ((self.sled_pos.x - x1) / (x2 - x1))

            # Check if sled is just above this line
            if sled_bottom - 1 <= y_on_line < highest_y and y_on_line >= sled_bottom - 5:
                highest_y = y_on_line
                best_candidate = (line, y_on_line)
                
        return best_candidate

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # move
            p[2] -= 1    # decrease lifetime

    def _calculate_reward(self):
        if self._check_win_condition():
            # _sound_effect: win_fanfare
            reward = 10.0
            if self.steps < 500:
                reward += 100.0
            return reward
        
        if self._check_crash_condition():
            # _sound_effect: crash_sound
            return -100.0
            
        speed = self.sled_vel.magnitude()
        if speed > self.MIN_SPEED_FOR_REWARD:
            return 0.1
        else:
            return -0.01

    def _check_termination(self):
        if self.game_over_message:
            return True

        if self._check_win_condition():
            self.game_over_message = "FINISH!"
            return True
        
        if self._check_crash_condition():
            self.game_over_message = "CRASHED!"
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over_message = "TIME UP"
            return True
            
        return False

    def _check_win_condition(self):
        return self.sled_pos.x >= self.FINISH_LINE_X

    def _check_crash_condition(self):
        # Off-screen crash
        if not (0 < self.sled_pos.x < self.SCREEN_WIDTH and 0 < self.sled_pos.y < self.SCREEN_HEIGHT):
            return True
        
        # Stuck crash
        if self.sled_vel.magnitude() < self.MIN_SPEED_FOR_CRASH:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
        
        if self.stuck_frames > self.STUCK_FRAMES_LIMIT:
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start and finish lines
        pygame.gfxdraw.filled_circle(self.screen, int(self.START_POS[0]), int(self.START_POS[1]), 10, self.COLOR_START)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.SCREEN_HEIGHT), 5)
        
        # Draw track lines
        for line in self.track_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, line[0], line[1], 3)
            
        # Draw particles
        for p in self.particles:
            size = max(0, p[2] / 4)
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p[0], size)

        # Draw speed lines
        speed = self.sled_vel.magnitude()
        if speed > 8:
            #_sound_effect: high_speed_wind
            for _ in range(3):
                offset = pygame.Vector2(self.np_random.uniform(-20, 20), self.np_random.uniform(-20, 20))
                start_pos = self.sled_pos + offset
                end_pos = start_pos - self.sled_vel.normalize() * self.np_random.uniform(10, 30)
                pygame.draw.aaline(self.screen, self.COLOR_SPEED_LINE, start_pos, end_pos, 1)

        # Draw sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE[0], self.SLED_SIZE[1])
        sled_rect.center = self.sled_pos
        
        rotated_surface = pygame.Surface(sled_rect.size, pygame.SRCALPHA)
        rotated_surface.fill((0,0,0,0)) # Transparent background
        
        sled_body_rect = pygame.Rect(0,0, self.SLED_SIZE[0], self.SLED_SIZE[1])
        pygame.draw.rect(rotated_surface, self.COLOR_SLED, sled_body_rect, border_radius=2)
        # Add a flag for orientation
        pygame.draw.line(rotated_surface, self.COLOR_SLED_FLAG, (self.SLED_SIZE[0]*0.7, self.SLED_SIZE[1]/2), (self.SLED_SIZE[0], self.SLED_SIZE[1]/2), 2)

        rotated_surface = pygame.transform.rotate(rotated_surface, -math.degrees(self.sled_angle))
        new_rect = rotated_surface.get_rect(center=sled_rect.center)

        self.screen.blit(rotated_surface, new_rect.topleft)

    def _render_ui(self):
        speed_text = f"Speed: {self.sled_vel.magnitude():.1f}"
        time_text = f"Time: {self.steps / self.FPS:.1f}s"
        score_text = f"Score: {self.score:.1f}"

        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_UI_TEXT)
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(speed_surf, (10, 10))
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH/2 - score_surf.get_width()/2, 10))

        if self.game_over_message:
            msg_surf = self.font_msg.render(self.game_over_message, True, self.COLOR_FINISH if "FINISH" in self.game_over_message else self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

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

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Line Rider Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # This is a basic mapping for human play testing.
    # An RL agent would provide the action array directly.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                done = True # Force reset

        if done:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            done = False
            continue

        # --- Action Generation for Manual Play ---
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # Default 'none' action
        for key, action_val in key_to_action.items():
            if keys[key]:
                movement_action = action_val
                break

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()