
# Generated: 2025-08-28T04:16:33.295965
# Source Brief: brief_02255.md
# Brief Index: 2255

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a 2D physics-based sledding game.
    The player draws lines to guide a sled to the finish line, navigating
    a minimalist world governed by gravity and collisions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys to select a direction, and hold Space to draw a line. "
        "Hold Shift while drawing to create a longer line. Guide the sled to the red finish line."
    )

    game_description = (
        "A minimalist physics puzzle. Draw lines on the screen to create a path for your sled. "
        "Use gravity to your advantage to reach the finish line as quickly as possible. "
        "Crashing or stopping for too long will end the run."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Used for time calculations
        self.MAX_STEPS = 1000
        self.STOP_THRESHOLD_STEPS = 10
        self.MAX_LINES = 25

        # Colors
        self.COLOR_BG = (230, 235, 240)
        self.COLOR_SLED = (0, 128, 255)
        self.COLOR_SLED_OUTLINE = (255, 255, 255)
        self.COLOR_LINE = (20, 20, 20)
        self.COLOR_START_ZONE = (0, 200, 0, 50)
        self.COLOR_FINISH_ZONE = (200, 0, 0, 50)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_PARTICLE = (255, 255, 255)

        # Physics
        self.GRAVITY = 0.15
        self.AIR_FRICTION = 0.995
        self.LINE_BOUNCE = 0.6
        self.LINE_FRICTION = 0.1
        self.SLED_RADIUS = 8
        self.SHORT_LINE_LEN = 40
        self.LONG_LINE_LEN = 80

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.sled_pos = None
        self.sled_vel = None
        self.lines = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.stopped_counter = 0
        self.terminated = False
        
        # Initial call to reset to set up the initial state
        self.reset()
        
        # Self-validation check
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sled_pos = np.array([60.0, 100.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.lines = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.stopped_counter = 0
        self.terminated = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self.reset()

        self._handle_action(action)
        self._update_physics()
        self._update_particles()

        self.terminated = self._check_termination()
        reward = self._calculate_reward()
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if not space_held or movement == 0:
            return

        line_len = self.LONG_LINE_LEN if shift_held else self.SHORT_LINE_LEN
        start_pos = self.sled_pos
        
        # SFX: // Play "swoosh" line drawing sound

        angle = 0
        if movement == 1: angle = -math.pi / 2  # Up
        elif movement == 2: angle = math.pi / 2   # Down
        elif movement == 3: angle = math.pi       # Left
        elif movement == 4: angle = 0             # Right

        end_pos = start_pos + np.array([math.cos(angle), math.sin(angle)]) * line_len

        self.lines.append((start_pos.copy(), end_pos.copy()))
        if len(self.lines) > self.MAX_LINES:
            self.lines.pop(0)

    def _update_physics(self):
        # Apply gravity and air friction
        self.sled_vel[1] += self.GRAVITY
        self.sled_vel *= self.AIR_FRICTION

        self._handle_collisions()

        # Update position
        self.sled_pos += self.sled_vel

        # Update stopped counter
        if np.linalg.norm(self.sled_vel) < 0.05:
            self.stopped_counter += 1
        else:
            self.stopped_counter = 0

    def _handle_collisions(self):
        for p1, p2 in self.lines:
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            p1_to_sled = self.sled_pos - p1
            t = np.clip(np.dot(p1_to_sled, line_vec) / line_len_sq, 0, 1)

            closest_point = p1 + t * line_vec
            dist_vec = self.sled_pos - closest_point
            distance = np.linalg.norm(dist_vec)

            if distance < self.SLED_RADIUS:
                # SFX: // Play "thud" or "scrape" sound
                self._create_particles(self.sled_pos, 5, self.sled_vel)

                # Resolve penetration
                overlap = self.SLED_RADIUS - distance
                self.sled_pos += (dist_vec / distance) * overlap

                # Calculate collision response
                normal = dist_vec / distance
                vel_comp_normal = np.dot(self.sled_vel, normal)

                if vel_comp_normal < 0:
                    # Bounce
                    restitution_vec = -(1 + self.LINE_BOUNCE) * vel_comp_normal * normal
                    self.sled_vel += restitution_vec

                    # Friction
                    tangent = np.array([-normal[1], normal[0]])
                    vel_comp_tangent = np.dot(self.sled_vel, tangent)
                    friction_impulse = -vel_comp_tangent * self.LINE_FRICTION * tangent
                    self.sled_vel += friction_impulse
    
    def _check_termination(self):
        is_finished = self.sled_pos[0] > self.WIDTH - 60
        is_out_of_bounds = not (0 < self.sled_pos[0] < self.WIDTH and -50 < self.sled_pos[1] < self.HEIGHT + 50)
        is_stopped = self.stopped_counter >= self.STOP_THRESHOLD_STEPS
        is_timeout = self.steps >= self.MAX_STEPS
        
        return is_finished or is_out_of_bounds or is_stopped or is_timeout

    def _calculate_reward(self):
        if self.terminated:
            if self.sled_pos[0] > self.WIDTH - 60:  # Reached finish
                # SFX: // Play "victory" sound
                speed_bonus = (self.MAX_STEPS - self.steps) * 0.01
                return 10.0 + speed_bonus
            else:  # Crashed, stopped, or timed out
                # SFX: // Play "failure" sound
                return -10.0
        
        # Continuous reward for moving right
        if self.sled_vel[0] > 0.1:
            return 0.02
        # Small penalty for moving left or being static
        else:
            return -0.01

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Zones
        start_surface = pygame.Surface((80, self.HEIGHT), pygame.SRCALPHA)
        start_surface.fill(self.COLOR_START_ZONE)
        self.screen.blit(start_surface, (0, 0))

        finish_surface = pygame.Surface((80, self.HEIGHT), pygame.SRCALPHA)
        finish_surface.fill(self.COLOR_FINISH_ZONE)
        self.screen.blit(finish_surface, (self.WIDTH - 80, 0))
        
        # Drawn lines
        for p1, p2 in self.lines:
            pygame.draw.line(self.screen, self.COLOR_LINE, p1, p2, 4)

        # Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                size = max(1, int(self.SLED_RADIUS / 2 * (p['life'] / p['max_life'])))
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'].astype(int), size)

        # Sled
        pos = self.sled_pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_SLED_OUTLINE, pos, self.SLED_RADIUS + 2)
        pygame.draw.circle(self.screen, self.COLOR_SLED, pos, self.SLED_RADIUS)

    def _render_ui(self):
        speed = np.linalg.norm(self.sled_vel) * 10
        time_elapsed = self.steps / self.FPS

        speed_text = self.font.render(f"Speed: {speed:.0f}", True, self.COLOR_TEXT)
        time_text = self.font.render(f"Time: {time_elapsed:.2f}s", True, self.COLOR_TEXT)
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)

        self.screen.blit(speed_text, (20, 15))
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 15))
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, 15))

    def _create_particles(self, pos, count, initial_vel):
        for _ in range(count):
            # Eject particles opposite to impact velocity for a "spark" effect
            base_angle = math.atan2(-initial_vel[1], -initial_vel[0])
            angle_spread = (self.np_random.random() - 0.5) * math.pi / 2
            final_angle = base_angle + angle_spread
            speed = self.np_random.random() * 2 + 1
            
            vel = np.array([math.cos(final_angle), math.sin(final_angle)]) * speed
            
            self.particles.append({
                'pos': pos.copy() + vel, # Start slightly outside sled
                'vel': vel,
                'life': self.np_random.integers(15, 25),
                'max_life': 25
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("--- Running Implementation Validation ---")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        print("✓ Action space is correct.")
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        print("✓ Observation space is correct.")
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        print("✓ reset() returns correct format.")
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ step() returns correct format.")
        
        print("--- ✓ Implementation Validated Successfully ---")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Control Setup ---
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # --- Pygame Display Setup ---
    pygame.display.set_caption("Sled Rider")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("      SLED RIDER CONTROLS")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Action Polling ---
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: terminated = True
        
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Only one direction at a time
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()