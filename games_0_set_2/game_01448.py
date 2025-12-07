
# Generated: 2025-08-27T17:11:24.321310
# Source Brief: brief_01448.md
# Brief Index: 1448

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to select a mirror, ←/→ to rotate the selected mirror. "
        "Goal: Hit the green target with the red laser."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A neon-drenched puzzle game. Manipulate mirrors to guide a laser beam through a maze to its target. "
        "You have a limited number of moves, so plan your rotations carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10
        self.LASER_MAX_BOUNCES = 20
        self.MIRROR_LENGTH = 50
        self.ROTATION_STEP = 5  # degrees

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_big = pygame.font.SysFont(None, 60)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_LASER_GLOW = (255, 50, 50, 100)
        self.COLOR_MIRROR = (80, 200, 255)
        self.COLOR_MIRROR_SELECTED = (255, 255, 100)
        self.COLOR_TARGET = (50, 255, 50)
        self.COLOR_TARGET_GLOW = (50, 255, 50, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.target_hit = False
        self.laser_start_pos = None
        self.laser_start_dir = None
        self.walls = []
        self.mirrors = []
        self.target = None
        self.selected_mirror_idx = 0
        self.laser_path = []
        self.num_reflections = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.target_hit = False
        self.selected_mirror_idx = 0

        # Level Design
        self.laser_start_pos = pygame.math.Vector2(50, self.HEIGHT / 2)
        self.laser_start_dir = pygame.math.Vector2(1, 0)
        
        self.target = pygame.Rect(self.WIDTH - 80, self.HEIGHT - 60, 30, 30)

        self.walls = [
            pygame.Rect(0, 0, self.WIDTH, 10),
            pygame.Rect(0, self.HEIGHT - 10, self.WIDTH, 10),
            pygame.Rect(0, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH - 10, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH / 2 - 5, 0, 10, self.HEIGHT/2 - 20),
            pygame.Rect(self.WIDTH / 2 - 5, self.HEIGHT/2 + 20, 10, self.HEIGHT/2 - 20),
        ]

        self.mirrors = [
            {'center': pygame.math.Vector2(180, 120), 'angle': 110},
            {'center': pygame.math.Vector2(450, 150), 'angle': 10},
        ]
        
        self._calculate_laser_path()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self.steps += 1
        reward = -0.2  # Cost per move/step
        
        needs_recalc = False
        if len(self.mirrors) > 0:
            if movement == 1:  # Up: Select previous mirror
                self.selected_mirror_idx = (self.selected_mirror_idx - 1) % len(self.mirrors)
            elif movement == 2:  # Down: Select next mirror
                self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
            elif movement == 3:  # Left: Rotate CCW
                self.mirrors[self.selected_mirror_idx]['angle'] -= self.ROTATION_STEP
                needs_recalc = True
            elif movement == 4:  # Right: Rotate CW
                self.mirrors[self.selected_mirror_idx]['angle'] += self.ROTATION_STEP
                needs_recalc = True

        if needs_recalc:
            # sfx: mirror_rotate.wav
            self._calculate_laser_path()

        reward += self.num_reflections * 1.0

        if self.target_hit:
            # sfx: success.wav
            reward += 50
            self.game_over = True
        
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            # sfx: fail.wav
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_laser_path(self):
        self.laser_path = [self.laser_start_pos]
        self.num_reflections = 0
        self.target_hit = False

        ray_origin = pygame.math.Vector2(self.laser_start_pos)
        ray_dir = pygame.math.Vector2(self.laser_start_dir)

        for _ in range(self.LASER_MAX_BOUNCES):
            min_dist = float('inf')
            closest_intersection = None
            reflected_dir = None
            hit_target = False

            # Check for intersections with walls
            for wall in self.walls:
                for i in range(4):
                    p1 = wall.topleft if i == 0 else wall.topright if i == 1 else wall.bottomright if i == 2 else wall.bottomleft
                    p2 = wall.topright if i == 0 else wall.bottomright if i == 1 else wall.bottomleft if i == 2 else wall.topleft
                    
                    intersection = self._get_intersection(ray_origin, ray_dir, p1, p2)
                    if intersection:
                        dist, point = intersection
                        if dist < min_dist:
                            min_dist = dist
                            closest_intersection = point
                            line_vec = pygame.math.Vector2(p2) - pygame.math.Vector2(p1)
                            normal = line_vec.rotate(90).normalize()
                            if ray_dir.dot(normal) > 0:
                                normal = -normal
                            reflected_dir = ray_dir.reflect(normal)
                            hit_target = False

            # Check for intersections with mirrors
            for mirror in self.mirrors:
                angle_rad = math.radians(mirror['angle'])
                m_dir = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * (self.MIRROR_LENGTH / 2)
                p1 = mirror['center'] - m_dir
                p2 = mirror['center'] + m_dir

                intersection = self._get_intersection(ray_origin, ray_dir, p1, p2)
                if intersection:
                    dist, point = intersection
                    if dist < min_dist:
                        min_dist = dist
                        closest_intersection = point
                        normal = m_dir.rotate(90).normalize()
                        if ray_dir.dot(normal) > 0:
                            normal = -normal
                        reflected_dir = ray_dir.reflect(normal)
                        hit_target = False

            # Check for intersection with target
            for i in range(4):
                p1 = self.target.topleft if i == 0 else self.target.topright if i == 1 else self.target.bottomright if i == 2 else self.target.bottomleft
                p2 = self.target.topright if i == 0 else self.target.bottomright if i == 1 else self.target.bottomleft if i == 2 else self.target.topleft
                intersection = self._get_intersection(ray_origin, ray_dir, p1, p2)
                if intersection:
                    dist, point = intersection
                    if dist < min_dist:
                        min_dist = dist
                        closest_intersection = point
                        hit_target = True

            if closest_intersection:
                self.laser_path.append(closest_intersection)
                if hit_target:
                    self.target_hit = True
                    break
                
                # sfx: laser_bounce.wav
                self.num_reflections += 1
                ray_origin = closest_intersection + reflected_dir * 0.01 # Epsilon to avoid self-intersection
                ray_dir = reflected_dir
            else:
                # Laser goes off-screen
                self.laser_path.append(ray_origin + ray_dir * 2000)
                break
    
    def _get_intersection(self, ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - pygame.math.Vector2(p1)
        v2 = pygame.math.Vector2(p2) - pygame.math.Vector2(p1)
        v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)

        dot = v2.dot(v3)
        if abs(dot) < 1e-6:
            return None # Parallel lines

        t1 = v2.cross(v1) / dot
        t2 = v1.dot(v3) / dot

        if t1 >= 1e-6 and 0.0 <= t2 <= 1.0:
            return t1, ray_origin + ray_dir * t1
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw target
        glow_rect = self.target.inflate(10, 10)
        pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_TARGET_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_TARGET, self.target)

        # Draw mirrors
        for i, mirror in enumerate(self.mirrors):
            angle_rad = math.radians(mirror['angle'])
            m_dir = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * (self.MIRROR_LENGTH / 2)
            p1 = mirror['center'] - m_dir
            p2 = mirror['center'] + m_dir
            
            color = self.COLOR_MIRROR_SELECTED if i == self.selected_mirror_idx else self.COLOR_MIRROR
            pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 4)

            if i == self.selected_mirror_idx:
                pulse_radius = 15 + 5 * math.sin(pygame.time.get_ticks() * 0.005)
                pygame.gfxdraw.aacircle(self.screen, int(mirror['center'].x), int(mirror['center'].y), int(pulse_radius), color)

        # Draw laser path
        if len(self.laser_path) > 1:
            path_for_drawing = [(int(p.x), int(p.y)) for p in self.laser_path]
            pygame.draw.aalines(self.screen, self.COLOR_LASER_GLOW, False, path_for_drawing, 3)
            pygame.draw.aalines(self.screen, self.COLOR_LASER, False, path_for_drawing, 1)
        
        # Draw particle at last reflection point
        if len(self.laser_path) > 1:
            last_point = self.laser_path[-1]
            for _ in range(5):
                offset_x = (random.random() - 0.5) * 10
                offset_y = (random.random() - 0.5) * 10
                pygame.gfxdraw.pixel(self.screen, int(last_point.x + offset_x), int(last_point.y + offset_y), self.COLOR_LASER)

    def _render_ui(self):
        moves_text = f"Moves: {self.MAX_STEPS - self.steps}/{self.MAX_STEPS}"
        self._draw_text(moves_text, (15, 15), self.COLOR_TEXT, self.font_ui)
        
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self._draw_text(score_text, (self.WIDTH - score_surf.get_width() - 15, 15), self.COLOR_TEXT, self.font_ui)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.target_hit:
                self._draw_text("TARGET HIT!", (self.WIDTH / 2, self.HEIGHT / 2), self.COLOR_SUCCESS, self.font_big, center=True)
            else:
                self._draw_text("OUT OF MOVES", (self.WIDTH / 2, self.HEIGHT / 2), self.COLOR_FAIL, self.font_big, center=True)

    def _draw_text(self, text, pos, color, font, center=False):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.MAX_STEPS - self.steps,
            "reflections": self.num_reflections,
            "target_hit": self.target_hit,
        }

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Laser Maze")
    clock = pygame.time.Clock()

    running = True
    while running:
        action_taken = False
        action = np.array([0, 0, 0]) # Default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                movement = 0
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    continue
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                
                if movement > 0:
                    action[0] = movement
                    action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to reset.")
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()