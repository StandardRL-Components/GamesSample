
# Generated: 2025-08-27T17:51:02.709073
# Source Brief: brief_01658.md
# Brief Index: 1658

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to aim the track up/down. ←/→ to aim horizontally. The track draws automatically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a path for the sled to ride on. Reach the green finish line without falling off the screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 48)

        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.1
        self.FRICTION = 0.995
        self.SLED_RADIUS = 8
        self.DRAW_SEGMENT_LENGTH = 10
        self.ANGLE_CHANGE_RATE = math.pi / 32
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (35, 40, 45)
        self.COLOR_TRACK = (255, 255, 255)
        self.COLOR_SLED = (255, 60, 60)
        self.COLOR_SLED_GLOW = (255, 150, 150)
        self.COLOR_FINISH = (60, 255, 60)
        self.COLOR_SPARK = (255, 180, 80)
        self.COLOR_TEXT = (220, 220, 220)

        # Initialize state variables (will be set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.sled_pos = None
        self.sled_vel = None
        self.track_points = None
        self.total_track_length = 0
        self.draw_angle = 0
        self.finish_x = self.WIDTH - 40
        self.particles = deque()
        self.crash_reason = ""
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crash_reason = ""
        
        # Initialize sled
        self.sled_pos = pygame.math.Vector2(50, self.HEIGHT / 2 - 50)
        self.sled_vel = pygame.math.Vector2(1.5, 0)
        
        # Initialize track
        self.track_points = [
            pygame.math.Vector2(0, self.HEIGHT / 2),
            pygame.math.Vector2(80, self.HEIGHT / 2)
        ]
        self.total_track_length = 80
        self.draw_angle = 0 # Horizontal
        
        # Clear particles
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do not advance state.
            # Return the final observation and info.
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 1. Handle player input to change drawing angle
        self._handle_action(movement)
        
        # 2. Update game logic
        self.steps += 1
        self._extend_track()
        self._update_sled()
        self._update_particles()
        
        # 3. Check for termination
        terminated, is_crash, is_win = self._check_termination()
        self.game_over = terminated
        
        # 4. Calculate reward
        reward = self._calculate_reward(terminated, is_crash, is_win)
        self.score += reward
        
        # 5. Return 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, movement):
        if movement == 1:  # Up
            self.draw_angle -= self.ANGLE_CHANGE_RATE
        elif movement == 2:  # Down
            self.draw_angle += self.ANGLE_CHANGE_RATE
        elif movement == 3:  # Left
            self.draw_angle = math.pi
        elif movement == 4:  # Right
            self.draw_angle = 0
        # movement == 0 is a no-op, angle remains the same
        
        # Clamp angle to prevent going fully vertical which can cause issues
        self.draw_angle = max(-math.pi/2 + 0.1, min(self.draw_angle, math.pi/2 - 0.1))

    def _extend_track(self):
        last_point = self.track_points[-1]
        new_point = last_point + pygame.math.Vector2(
            math.cos(self.draw_angle) * self.DRAW_SEGMENT_LENGTH,
            math.sin(self.draw_angle) * self.DRAW_SEGMENT_LENGTH,
        )
        
        # Prevent drawing off-screen
        new_point.y = max(0, min(new_point.y, self.HEIGHT))
        
        # Prevent drawing backwards past the start
        if new_point.x < 0:
            new_point.x = 0

        self.track_points.append(new_point)
        self.total_track_length += self.DRAW_SEGMENT_LENGTH

    def _update_sled(self):
        # Apply gravity
        self.sled_vel.y += self.GRAVITY
        
        # Update position
        self.sled_pos += self.sled_vel
        
        # Collision detection and resolution
        collided = False
        for i in range(len(self.track_points) - 1, 0, -1):
            p1 = self.track_points[i - 1]
            p2 = self.track_points[i]

            # Broad phase check
            if not (min(p1.x, p2.x) - self.SLED_RADIUS < self.sled_pos.x < max(p1.x, p2.x) + self.SLED_RADIUS):
                continue

            seg_vec = p2 - p1
            if seg_vec.length() == 0:
                continue

            # Project sled position onto the segment line
            t = seg_vec.dot(self.sled_pos - p1) / seg_vec.length_squared()
            
            # We only care about collisions within the segment bounds for resolution
            if 0 <= t <= 1:
                closest_point = p1 + t * seg_vec
                dist_vec = self.sled_pos - closest_point
                
                if dist_vec.length() < self.SLED_RADIUS:
                    # Collision
                    collided = True
                    
                    # Resolve position
                    penetration = self.SLED_RADIUS - dist_vec.length()
                    self.sled_pos += dist_vec.normalize() * penetration
                    
                    # Resolve velocity
                    normal = dist_vec.normalize()
                    vel_dot_normal = self.sled_vel.dot(normal)
                    if vel_dot_normal < 0: # Moving into the surface
                        self.sled_vel -= normal * vel_dot_normal # Remove normal component
                        self.sled_vel *= self.FRICTION # Apply friction
                    
                    # Create speed particles
                    speed = self.sled_vel.length()
                    if speed > 1.5:
                        for _ in range(int(speed)):
                            self._create_particle(self.sled_pos, p_type='trail')
                    
                    # A sled can only be on one segment at a time
                    break
    
    def _update_particles(self):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                self.particles.append(p)

    def _create_particle(self, pos, p_type='trail', count=1):
        for _ in range(count):
            if p_type == 'trail':
                vel = pygame.math.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)) - self.sled_vel * 0.1
                life = random.randint(15, 30)
                radius = random.uniform(1, 3)
                color = self.COLOR_TRACK
            elif p_type == 'crash':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 5)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                life = random.randint(30, 60)
                radius = random.uniform(2, 5)
                color = self.COLOR_SPARK

            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'radius': radius, 'color': color})

    def _check_termination(self):
        is_win = self.sled_pos.x >= self.finish_x
        is_crash = not (0 < self.sled_pos.x < self.WIDTH and 0 < self.sled_pos.y < self.HEIGHT)
        max_steps_reached = self.steps >= self.MAX_STEPS

        if is_crash:
            self.crash_reason = "Fell off the track!"
            # Sound: crash.wav
            self._create_particle(self.sled_pos, p_type='crash', count=50)

        if is_win:
            # Sound: win.wav
            pass

        if max_steps_reached and not is_win:
            is_crash = True
            self.crash_reason = "Ran out of time!"

        terminated = is_win or is_crash
        return terminated, is_crash, is_win

    def _calculate_reward(self, terminated, is_crash, is_win):
        if is_win:
            # Reward for finishing + bonus for track length efficiency
            base_reward = 100
            length_bonus = self.total_track_length / 10
            time_penalty = self.steps * 0.1
            return base_reward + length_bonus - time_penalty
        if is_crash:
            return -10
        
        # Small reward for staying alive and moving forward
        return self.sled_vel.x * 0.1

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "track_length": self.total_track_length,
            "sled_speed": self.sled_vel.length() if self.sled_vel else 0,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw finish line
        for y in range(0, self.HEIGHT, 10):
            pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, y), (self.finish_x, y + 5), 2)

        # Draw track
        if len(self.track_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, self.track_points, 3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 60.0))))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), (*color, alpha))

        # Draw sled
        if self.sled_pos:
            pos = (int(self.sled_pos.x), int(self.sled_pos.y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLED_RADIUS + 3, (*self.COLOR_SLED_GLOW, 100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLED_RADIUS + 3, (*self.COLOR_SLED_GLOW, 100))
            # Sled body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED)

    def _render_ui(self):
        # Draw text helper
        def draw_text(text, font, color, x, y, align="topleft"):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            setattr(rect, align, (x, y))
            self.screen.blit(surface, rect)

        # UI Info
        draw_text(f"SCORE: {self.score:.0f}", self.font_ui, self.COLOR_TEXT, 10, 10)
        draw_text(f"TIME: {self.steps}", self.font_ui, self.COLOR_TEXT, 10, 35)
        
        # Game Over message
        if self.game_over:
            if self.sled_pos.x >= self.finish_x: # Win
                draw_text("FINISH!", self.font_big, self.COLOR_FINISH, self.WIDTH // 2, self.HEIGHT // 2 - 40, "center")
            else: # Lose
                draw_text("CRASH!", self.font_big, self.COLOR_SLED, self.WIDTH // 2, self.HEIGHT // 2 - 40, "center")
                draw_text(self.crash_reason, self.font_ui, self.COLOR_TEXT, self.WIDTH // 2, self.HEIGHT // 2, "center")
            draw_text("Resetting...", self.font_ui, self.COLOR_TEXT, self.WIDTH // 2, self.HEIGHT // 2 + 30, "center")
        elif self.steps < 150: # Show controls for a few seconds
            draw_text(self.user_guide, self.font_ui, self.COLOR_TEXT, self.WIDTH // 2, self.HEIGHT - 20, "midbottom")

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Override screen for direct rendering
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Drawer")

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Default action: no-op, buttons released
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The environment handles its own rendering in _get_observation
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Run at 30 FPS

    env.close()