import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:22:07.129139
# Source Brief: brief_02552.md
# Brief Index: 2552
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a beam of light.

    The goal is to adjust the angle of a light emitter to reflect a beam
    off a series of 10 mirrors and hit a final target. This must be
    accomplished before a 45-second timer runs out. The game rewards
    precision and quick thinking, with a strong emphasis on visual feedback
    and a minimalist, neon aesthetic.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right.
      Up/Left decrease the beam angle, Down/Right increase it.
    - `actions[1]` (Space): Unused.
    - `actions[2]` (Shift): Unused.

    **Observation Space:** `Box(0, 255, (400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 for each unique mirror the beam reflects off in a step.
    - +100 for hitting the target after reflecting off all 10 mirrors (win).
    - -100 for the timer running out (loss).

    **Termination:**
    - The episode ends if the player wins or the timer reaches zero.
    - Max episode length is 2250 steps (45 seconds at 50 FPS).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a beam of light by adjusting the emitter's angle. Reflect the beam off all mirrors to hit the final target before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to adjust the angle of the light beam."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 50
    MAX_TIME_SECONDS = 45
    NUM_MIRRORS = 10
    ANGLE_STEP = math.radians(0.5) # Rotation speed of the emitter
    MAX_BOUNCES = 15 # Max reflections to prevent infinite loops

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_EMITTER = (220, 220, 255)
    COLOR_BEAM = (0, 255, 255)
    COLOR_BEAM_CORE = (200, 255, 255)
    COLOR_MIRROR = (150, 150, 170)
    COLOR_MIRROR_HIT = (0, 255, 255)
    COLOR_TARGET = (255, 150, 0)
    COLOR_TARGET_HIT = (255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GRID = (20, 30, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.timer = 0.0
        self.max_steps = self.FPS * self.MAX_TIME_SECONDS
        
        self.emitter_pos = pygame.math.Vector2(50, self.SCREEN_HEIGHT / 2)
        self.beam_angle = 0.0
        
        self.mirrors = []
        self._setup_level()
        
        self.target_pos = pygame.math.Vector2(self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT / 2)
        self.target_radius = 20
        
        self.beam_path = []
        self.hit_mirrors_indices = set()
        self.target_is_hit = False
        
        # --- Self-Validation ---
        # self.validate_implementation() # Removed for submission

    def _setup_level(self):
        """Defines the positions and orientations of mirrors for the puzzle."""
        self.mirrors = []
        mirror_definitions = [
            ((150, 50), (150, 150)),
            ((120, 300), (220, 300)),
            ((250, 100), (350, 200)),
            ((280, 350), (280, 250)),
            ((400, 50), (500, 50)),
            ((480, 150), (480, 250)),
            ((550, 350), (450, 350)),
            ((380, 300), (420, 260)),
            ((220, 50), (220, 150)),
            ((580, 100), (580, 300))
        ]
        
        for p1_coords, p2_coords in mirror_definitions[:self.NUM_MIRRORS]:
            p1 = pygame.math.Vector2(p1_coords)
            p2 = pygame.math.Vector2(p2_coords)
            
            # For rendering as a thick line
            rect = pygame.draw.line(self.screen, (0,0,0), p1, p2, 1)
            
            # For physics
            tangent = (p2 - p1).normalize()
            normal = tangent.rotate(90)

            self.mirrors.append({
                "p1": p1, "p2": p2, "rect": rect, "normal": normal
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.timer = self.MAX_TIME_SECONDS
        self.beam_angle = math.radians(self.np_random.uniform(-15, 15))

        self._calculate_beam_path()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1=up, 2=down, 3=left, 4=right
        if movement in [1, 3]: # Up or Left -> Counter-Clockwise
            self.beam_angle -= self.ANGLE_STEP
        elif movement in [2, 4]: # Down or Right -> Clockwise
            self.beam_angle += self.ANGLE_STEP
        
        # --- Game Logic Update ---
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        self._calculate_beam_path()
        
        # --- Reward Calculation ---
        reward = 0.0
        # Continuous reward for hitting mirrors
        reward += len(self.hit_mirrors_indices) * 0.1
        
        # --- Termination Check ---
        terminated = False
        self.win_condition_met = self.target_is_hit and len(self.hit_mirrors_indices) == self.NUM_MIRRORS
        
        if self.win_condition_met:
            # Win condition: Hit target after all mirrors
            reward += 100.0 # Goal-oriented win reward
            terminated = True
            self.game_over = True
            # Sound: Win Jingle
        elif self.timer <= 0 or self.steps >= self.max_steps:
            # Lose condition: Timer runs out
            reward = -100.0 # Overwrite other rewards on loss
            terminated = True
            self.game_over = True
            # Sound: Loss Buzzer

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated is always False
            self._get_info()
        )

    def _calculate_beam_path(self):
        """Traces the beam's path, handling reflections and target hits."""
        self.beam_path = [self.emitter_pos]
        self.hit_mirrors_indices = set()
        self.target_is_hit = False

        pos = pygame.math.Vector2(self.emitter_pos)
        direction = pygame.math.Vector2(1, 0).rotate(math.degrees(self.beam_angle))

        for bounce_num in range(self.MAX_BOUNCES):
            intersections = []

            # Check for mirror intersections
            for i, mirror in enumerate(self.mirrors):
                intersect_pt = self._get_line_segment_intersection(pos, direction, mirror["p1"], mirror["p2"])
                if intersect_pt:
                    dist = pos.distance_to(intersect_pt)
                    if dist > 1e-6: # Avoid self-intersection
                        intersections.append(("mirror", dist, intersect_pt, i))

            # Check for target intersection
            target_intersect_pts = self._get_line_circle_intersection(pos, direction, self.target_pos, self.target_radius)
            for pt in target_intersect_pts:
                dist = pos.distance_to(pt)
                if dist > 1e-6:
                    intersections.append(("target", dist, pt, None))

            if not intersections:
                # No intersections, beam goes off-screen
                self.beam_path.append(pos + direction * 2000)
                break

            # Find the closest intersection
            intersections.sort(key=lambda x: x[1])
            obj_type, dist, point, index = intersections[0]

            self.beam_path.append(point)
            
            if obj_type == "mirror":
                # Sound: Beam reflect
                self.hit_mirrors_indices.add(index)
                pos = point
                
                # Ensure normal points towards the incoming beam
                normal = self.mirrors[index]["normal"]
                if direction.dot(normal) > 0:
                    normal = -normal
                direction = direction.reflect(normal)
            
            elif obj_type == "target":
                # Sound: Target hit
                self.target_is_hit = True
                break
        
    def _get_observation(self):
        """Renders the game state to an RGB array."""
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "mirrors_hit": len(self.hit_mirrors_indices),
            "target_hit": self.target_is_hit
        }

    # --- Rendering Methods ---
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_game(self):
        # Render mirrors
        for i, mirror in enumerate(self.mirrors):
            color = self.COLOR_MIRROR_HIT if i in self.hit_mirrors_indices else self.COLOR_MIRROR
            pygame.draw.line(self.screen, color, mirror["p1"], mirror["p2"], 5)
        
        # Render target
        target_color = self.COLOR_TARGET_HIT if self.target_is_hit else self.COLOR_TARGET
        if self.target_is_hit and self.steps % 10 < 5: # Flashing effect
             target_color = self.COLOR_BEAM_CORE
        pygame.gfxdraw.filled_circle(self.screen, int(self.target_pos.x), int(self.target_pos.y), self.target_radius, target_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.target_pos.x), int(self.target_pos.y), self.target_radius, target_color)

        # Render beam
        if len(self.beam_path) > 1:
            # Glow effect
            pygame.draw.lines(self.screen, self.COLOR_BEAM, False, self.beam_path, 7)
            pygame.draw.lines(self.screen, self.COLOR_BEAM, False, self.beam_path, 5)
            # Core beam
            pygame.draw.lines(self.screen, self.COLOR_BEAM_CORE, False, self.beam_path, 2)
        
        # Render emitter and reflection glows
        self._render_glow(self.emitter_pos, 15, self.COLOR_EMITTER)
        for i, point in enumerate(self.beam_path):
            if 0 < i < len(self.beam_path):
                self._render_glow(point, 10, self.COLOR_BEAM)

    def _render_glow(self, pos, max_radius, color):
        """Renders a soft glow effect using transparent circles."""
        x, y = int(pos.x), int(pos.y)
        for r in range(max_radius, 0, -1):
            alpha = int(100 * (1 - r / max_radius))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, glow_color)

    def _render_ui(self):
        # Timer display
        timer_text = f"TIME: {max(0, self.timer):.2f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        # Score display
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Mirrors hit display
        mirror_text = f"MIRRORS: {len(self.hit_mirrors_indices)}/{self.NUM_MIRRORS}"
        mirror_surf = self.font_ui.render(mirror_text, True, self.COLOR_TEXT)
        self.screen.blit(mirror_surf, (self.SCREEN_WIDTH // 2 - mirror_surf.get_width() // 2, 10))

        # Game over message
        if self.game_over:
            msg_text = "SUCCESS!" if self.win_condition_met else "TIME UP!"
            msg_color = self.COLOR_TARGET_HIT if self.win_condition_met else self.COLOR_TARGET
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    # --- Utility and Physics Methods ---
    def _get_line_segment_intersection(self, ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)
        dot_v2_v3 = v2.dot(v3)
        if abs(dot_v2_v3) < 1e-6:
            return None # Parallel lines
        
        t1 = v2.cross(v1) / dot_v2_v3
        t2 = v1.dot(v3) / dot_v2_v3

        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return ray_origin + t1 * ray_dir
        return None

    def _get_line_circle_intersection(self, ray_origin, ray_dir, circle_center, radius):
        # Ray: P = O + t*D
        # Circle: |X - C|^2 = r^2
        # Substitute P for X: |O + t*D - C|^2 = r^2
        # Let L = O - C. |L + t*D|^2 = r^2 -> (L+tD).(L+tD)=r^2
        # L.L + 2t(L.D) + t^2(D.D) = r^2
        # t^2(D.D) + t(2L.D) + (L.L - r^2) = 0
        # This is a quadratic equation at^2 + bt + c = 0
        
        L = ray_origin - circle_center
        a = ray_dir.dot(ray_dir) # Should be 1 if ray_dir is normalized
        b = 2 * L.dot(ray_dir)
        c = L.dot(L) - radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return [] # No intersection

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        
        points = []
        if t1 >= 0: points.append(ray_origin + t1 * ray_dir)
        if t2 >= 0 and t1 != t2: points.append(ray_origin + t2 * ray_dir)
        
        return points

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local development.
    # To run it, you'll need to `pip install pygame`.
    # It sets up a human-playable version of the game.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS

    env = GameEnv()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Light Bender Environment")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # --- Main Game Loop ---
    while not done:
        action = [0, 0, 0] # Default to no-op
        
        # Event handling for human control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Display final state for 2 seconds before resetting
            # The observation is (W, H, C), but pygame surface wants (W, H)
            # and surfarray.make_surface expects (W, H) or (W, H, C).
            # The obs is (H, W, C), so we need to transpose it for display.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()