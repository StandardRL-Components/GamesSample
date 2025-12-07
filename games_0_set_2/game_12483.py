import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:57:25.215945
# Source Brief: brief_02483.md
# Brief Index: 2483
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Laser Maze Environment for Gymnasium.

    The player controls the angle of a laser emitter. The goal is to reflect the
    laser off a series of procedurally generated mirrors to achieve 10 reflections
    while keeping the laser's intensity above 50%.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim a laser to reflect it off mirrors. Achieve 10 reflections while keeping the laser's intensity high to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim the laser. ↑↓ for fine-tuning, ←→ for larger adjustments."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_REFLECTIONS = 10
    MIN_INTENSITY = 0.5
    INTENSITY_LOSS_PER_REFLECTION = 0.05

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_MIRROR = (180, 180, 190)
    COLOR_LASER_BRIGHT = (100, 255, 150)
    COLOR_LASER_DIM = (20, 80, 40)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    # Physics
    SMALL_ANGLE_CHANGE = math.radians(0.5)
    LARGE_ANGLE_CHANGE = math.radians(2.0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laser_origin = (50, self.SCREEN_HEIGHT / 2)
        self.laser_angle = 0.0
        self.intensity = 1.0
        self.reflection_count = 0
        self.mirrors = []
        self.laser_path = []
        self.win_condition = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.intensity = 1.0
        self.reflection_count = 0
        self.laser_angle = self.np_random.uniform(-math.pi / 8, math.pi / 8)
        
        self._generate_mirrors()
        self.laser_path, self.reflection_count, self.intensity = self._calculate_laser_path()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # 1. Update game state from action
        self._update_angle(movement)
        self.steps += 1
        
        # 2. Update game logic
        old_reflection_count = self.reflection_count
        self.laser_path, self.reflection_count, self.intensity = self._calculate_laser_path()

        # 3. Calculate reward and check termination
        terminated = self._check_termination()
        reward = self._calculate_reward(old_reflection_count, terminated)
        self.score += reward

        if terminated:
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_angle(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1:  # Up (decrease angle)
            self.laser_angle -= self.SMALL_ANGLE_CHANGE
        elif movement == 2:  # Down (increase angle)
            self.laser_angle += self.SMALL_ANGLE_CHANGE
        elif movement == 3:  # Left (larger decrease)
            self.laser_angle -= self.LARGE_ANGLE_CHANGE
        elif movement == 4:  # Right (larger increase)
            self.laser_angle += self.LARGE_ANGLE_CHANGE
        
        # Clamp angle to avoid spinning too far
        self.laser_angle = max(-math.pi / 2, min(math.pi / 2, self.laser_angle))

    def _calculate_reward(self, old_reflection_count, terminated):
        if not terminated:
            # Reward for new reflections
            return (self.reflection_count - old_reflection_count) * 0.1
        else:
            if self.win_condition:
                # Win reward: +5 for 10 reflections, +50 for winning
                return 55.0
            else:
                # Lose penalty
                return -50.0

    def _check_termination(self):
        if self.reflection_count >= self.WIN_REFLECTIONS and self.intensity >= self.MIN_INTENSITY:
            self.win_condition = True
            return True
        if self.intensity < self.MIN_INTENSITY:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        # Laser path ending off-screen is handled by the path calculation, not a termination itself
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "intensity": self.intensity,
            "reflections": self.reflection_count
        }

    def _render_game(self):
        # Draw mirrors
        for x1, y1, x2, y2 in self.mirrors:
            pygame.draw.aaline(self.screen, self.COLOR_MIRROR, (x1, y1), (x2, y2))
        
        # Draw laser beam with glow
        if len(self.laser_path) > 1:
            # Interpolate color based on intensity
            current_color = (
                int(self.COLOR_LASER_DIM[0] + (self.COLOR_LASER_BRIGHT[0] - self.COLOR_LASER_DIM[0]) * self.intensity),
                int(self.COLOR_LASER_DIM[1] + (self.COLOR_LASER_BRIGHT[1] - self.COLOR_LASER_DIM[1]) * self.intensity),
                int(self.COLOR_LASER_DIM[2] + (self.COLOR_LASER_BRIGHT[2] - self.COLOR_LASER_DIM[2]) * self.intensity)
            )
            
            # Glow effect
            pygame.draw.lines(self.screen, current_color, False, self.laser_path, width=5)
            # Core beam
            pygame.draw.lines(self.screen, (255, 255, 255), False, self.laser_path, width=1)

        # Draw reflection points
        for i, point in enumerate(self.laser_path):
            if 0 < i < len(self.laser_path): # Don't draw at origin
                pygame.gfxdraw.filled_circle(self.screen, int(point[0]), int(point[1]), 3, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, int(point[0]), int(point[1]), 3, (255, 255, 255))

    def _render_ui(self):
        # Reflections text
        refl_text = self.font_ui.render(f"Reflections: {self.reflection_count}/{self.WIN_REFLECTIONS}", True, self.COLOR_TEXT)
        self.screen.blit(refl_text, (10, 10))

        # Intensity text
        intensity_perc = int(self.intensity * 100)
        intensity_color = self.COLOR_WIN if intensity_perc >= 50 else self.COLOR_LOSE
        intensity_text = self.font_ui.render(f"Intensity: {intensity_perc}%", True, intensity_color)
        text_rect = intensity_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(intensity_text, text_rect)

        # Game over message
        if self.game_over:
            if self.win_condition:
                msg_text = self.font_msg.render("VICTORY", True, self.COLOR_WIN)
            else:
                msg_text = self.font_msg.render("FAILURE", True, self.COLOR_LOSE)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _generate_mirrors(self):
        self.mirrors = []
        # A simple configuration that guarantees a solution
        # This can be replaced with more complex procedural generation
        self.mirrors.append((200, 50, 200, 350))
        self.mirrors.append((350, 50, 350, 350))
        self.mirrors.append((500, 50, 500, 350))
        
        # Add some angled mirrors for more interesting paths
        self.mirrors.append((120, 50, 280, 200)) # Angled top-left
        self.mirrors.append((120, 350, 280, 200)) # Angled bottom-left
        self.mirrors.append((420, 50, 580, 200)) # Angled top-right
        self.mirrors.append((420, 350, 580, 200)) # Angled bottom-right

    def _calculate_laser_path(self):
        path = [self.laser_origin]
        current_pos = self.laser_origin
        current_angle = self.laser_angle
        current_intensity = 1.0
        
        for i in range(self.WIN_REFLECTIONS + 5): # Allow a few extra reflections
            direction = (math.cos(current_angle), math.sin(current_angle))
            
            closest_hit = None
            min_dist = float('inf')

            # Check for intersections with mirrors
            for mirror in self.mirrors:
                hit = self._get_line_intersection(current_pos, direction, mirror)
                if hit:
                    dist = math.hypot(hit['point'][0] - current_pos[0], hit['point'][1] - current_pos[1])
                    if dist < min_dist and dist > 1e-6: # Avoid self-intersection
                        min_dist = dist
                        closest_hit = hit
            
            # Check for intersections with screen boundaries
            boundaries = [
                (0, 0, self.SCREEN_WIDTH, 0), (0, 0, 0, self.SCREEN_HEIGHT),
                (self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
                (0, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            ]
            for boundary in boundaries:
                hit = self._get_line_intersection(current_pos, direction, boundary)
                if hit:
                    dist = math.hypot(hit['point'][0] - current_pos[0], hit['point'][1] - current_pos[1])
                    if dist < min_dist and dist > 1e-6:
                        min_dist = dist
                        closest_hit = {'point': hit['point'], 'mirror': None}

            if closest_hit:
                path.append(closest_hit['point'])
                if closest_hit['mirror']:
                    # Reflection
                    current_pos = closest_hit['point']
                    mirror_vec = (closest_hit['mirror'][2] - closest_hit['mirror'][0], closest_hit['mirror'][3] - closest_hit['mirror'][1])
                    mirror_angle = math.atan2(mirror_vec[1], mirror_vec[0])
                    normal_angle = mirror_angle - math.pi / 2
                    current_angle = 2 * normal_angle - current_angle + math.pi
                    current_intensity *= (1.0 - self.INTENSITY_LOSS_PER_REFLECTION)
                else:
                    # Hit a boundary, path ends
                    break
            else:
                # No intersection found, path ends
                break

        reflection_count = max(0, len(path) - 2)
        return path, reflection_count, current_intensity

    @staticmethod
    def _get_line_intersection(ray_origin, ray_dir, segment):
        p1 = ray_origin
        p2 = (ray_origin[0] + ray_dir[0], ray_origin[1] + ray_dir[1])
        p3 = (segment[0], segment[1])
        p4 = (segment[2], segment[3])

        den = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if den == 0:
            return None  # Parallel lines

        t_num = (p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])
        u_num = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0]))

        t = t_num / den
        u = u_num / den

        if t > 0 and 0 <= u <= 1:
            intersection_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
            return {'point': intersection_point, 'mirror': segment}
        
        return None

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        # Test game logic assertions
        assert 0 <= self.intensity <= 1.0
        assert self.reflection_count <= self.WIN_REFLECTIONS + 5 # Max reflections in path calc
        assert self.steps <= self.MAX_STEPS
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Script ---
    # The following block will not run in a Gymnasium environment,
    # but is useful for manual testing and development.
    # To run, execute this script directly: `python your_env_file.py`
    
    # Check if a display is available for the manual play mode
    try:
        os.environ.pop("SDL_VIDEODRIVER")
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        print("No display available, running in headless mode. Manual play disabled.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Re-initialize pygame in dummy mode
        pygame.init()
        pygame.font.init()
        
        # If no display, just run a short headless test
        env = GameEnv()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        env.close()
        print("Headless test complete.")
        exit()


    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Laser Maze - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit | R: Reset")
    
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
        
        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Reflections: {info['reflections']}, Intensity: {info['intensity']:.2f}")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()