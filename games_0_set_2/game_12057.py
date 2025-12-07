import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:02:05.728099
# Source Brief: brief_02057.md
# Brief Index: 2057
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque, namedtuple

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where you reflect a laser with mirrors. Activate all switches by hitting them "
        "with the beam and its past echoes before time runs out."
    )
    user_guide = (
        "Controls: ↑/↓ to rotate the selected mirror. Press space to cycle to the next mirror."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60 # For rendering smoothness, not game logic
    MAX_STEPS = 3600 # 60 seconds * 60 steps/sec logic update
    
    # Visuals
    COLOR_BG = (15, 18, 32)
    COLOR_BEAM = (255, 255, 0)
    COLOR_ECHO = (255, 255, 0)
    COLOR_MIRROR = (220, 220, 255)
    COLOR_MIRROR_SELECTED = (0, 255, 255)
    COLOR_SWITCH_INACTIVE = (70, 70, 90)
    COLOR_SWITCH_ACTIVE = (0, 150, 255)
    COLOR_SWITCH_ACTIVE_GLOW = (0, 150, 255, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_TIMER_NORMAL = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 80, 80)
    
    # Gameplay
    INITIAL_MIRROR_COUNT = 2
    MIRROR_LENGTH = 80
    MIRROR_ROTATION_SPEED = 2.5 # degrees per step
    SWITCH_SIZE = 16
    NUM_SWITCHES = 8
    BEAM_MAX_REFLECTIONS = 15
    ECHO_DELAYS = [20, 40, 60] # In steps

    class Mirror:
        def __init__(self, x, y, angle, length):
            self.center = pygame.math.Vector2(x, y)
            self.angle = angle
            self.length = length
            self.p1 = pygame.math.Vector2(0, 0)
            self.p2 = pygame.math.Vector2(0, 0)
            self.update_endpoints()

        def rotate(self, amount):
            self.angle = (self.angle + amount) % 360
            self.update_endpoints()

        def update_endpoints(self):
            half_len = self.length / 2
            self.p1 = self.center + pygame.math.Vector2(half_len, 0).rotate(self.angle)
            self.p2 = self.center + pygame.math.Vector2(-half_len, 0).rotate(self.angle)
        
        def get_normal(self):
            return (self.p2 - self.p1).rotate(90).normalize()

    class Switch:
        def __init__(self, x, y, size):
            self.rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            self.is_active = False
            self.center = pygame.math.Vector2(x, y)
            self.activation_step = -1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_timer = pygame.font.SysFont("Consolas", 32)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_timer = pygame.font.SysFont(None, 36)

        self.mirror_count = self.INITIAL_MIRROR_COUNT
        self.game_won_previously = False
        
        # Initialize state variables to be defined in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mirrors = []
        self.switches = []
        self.beam_origin = pygame.math.Vector2(10, self.HEIGHT / 2)
        self.beam_path_history = deque(maxlen=max(self.ECHO_DELAYS) + 5)
        self.selected_mirror_idx = 0
        self.last_step_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_won_previously:
            self.mirror_count += 2
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won_previously = False
        
        self._generate_level()
        
        self.beam_path_history.clear()
        self.selected_mirror_idx = 0
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.switches = []
        for _ in range(self.NUM_SWITCHES):
            while True:
                x = self.np_random.integers(self.WIDTH // 4, self.WIDTH - 50)
                y = self.np_random.integers(50, self.HEIGHT - 50)
                new_switch = self.Switch(x, y, self.SWITCH_SIZE)
                # Ensure switches don't overlap
                if not any(new_switch.rect.colliderect(s.rect.inflate(self.SWITCH_SIZE, self.SWITCH_SIZE)) for s in self.switches):
                    self.switches.append(new_switch)
                    break

        self.mirrors = []
        if self.mirror_count <= self.INITIAL_MIRROR_COUNT: # Tutorial level
            self.mirrors.append(self.Mirror(180, self.HEIGHT / 2, -45, self.MIRROR_LENGTH))
            self.mirrors.append(self.Mirror(450, self.HEIGHT / 2, 45, self.MIRROR_LENGTH))
        else:
            for i in range(self.mirror_count):
                 while True:
                    x = self.np_random.integers(100, self.WIDTH - 100)
                    y = self.np_random.integers(50, self.HEIGHT - 50)
                    angle = self.np_random.uniform(0, 360)
                    new_mirror = self.Mirror(x, y, angle, self.MIRROR_LENGTH)
                    # Ensure mirrors don't overlap heavily
                    if not any(new_mirror.center.distance_to(m.center) < self.MIRROR_LENGTH for m in self.mirrors):
                        self.mirrors.append(new_mirror)
                        break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_step_reward = 0.0
        
        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            self.last_step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, _ = action
        
        if not self.mirrors: return
        
        # 0=no-op (cycles selected mirror), 1=up (rot+), 2=down (rot-)
        if movement == 0:
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
        elif movement == 1: # Rotate Clockwise
            self.mirrors[self.selected_mirror_idx].rotate(self.MIRROR_ROTATION_SPEED)
        elif movement == 2: # Rotate Counter-Clockwise
            self.mirrors[self.selected_mirror_idx].rotate(-self.MIRROR_ROTATION_SPEED)

    def _update_game_state(self):
        main_beam_path = self._calculate_beam_path()
        self.beam_path_history.append(main_beam_path)

        main_beam_hits = self._get_path_switch_collisions(main_beam_path)
        
        # Reward for main beam hitting switches
        self.last_step_reward += 0.1 * len(main_beam_hits)

        all_echo_hits = set()
        newly_activated_switches = set()

        for delay in self.ECHO_DELAYS:
            if len(self.beam_path_history) > delay:
                echo_path = self.beam_path_history[-delay-1]
                echo_hits = self._get_path_switch_collisions(echo_path)
                all_echo_hits.update(echo_hits)
                
                # Check for activation
                possible_activations = main_beam_hits.intersection(echo_hits)
                for switch_idx in possible_activations:
                    if not self.switches[switch_idx].is_active:
                        newly_activated_switches.add(switch_idx)

        # Reward for echoes hitting switches
        self.last_step_reward += 0.5 * len(all_echo_hits)

        # Process and reward activations
        if newly_activated_switches:
            # SFX: SwitchActivate.wav
            for switch_idx in newly_activated_switches:
                self.switches[switch_idx].is_active = True
                self.switches[switch_idx].activation_step = self.steps
                self.score += 10
                self.last_step_reward += 10

    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        if all(s.is_active for s in self.switches):
            self.last_step_reward += 100
            self.score += 100
            self.game_over = True
            self.game_won_previously = True
            # SFX: LevelComplete.wav
            return True
        
        # Lose condition
        if self.steps >= self.MAX_STEPS:
            self.last_step_reward -= 100
            self.game_over = True
            # SFX: TimeOut.wav
            return True
        
        return False

    def _calculate_beam_path(self):
        path = [self.beam_origin]
        ray_origin = self.beam_origin
        ray_dir = pygame.math.Vector2(1, 0)

        for _ in range(self.BEAM_MAX_REFLECTIONS):
            intersections = []
            for mirror in self.mirrors:
                intersect_point = self._get_ray_segment_intersection(ray_origin, ray_dir, mirror.p1, mirror.p2)
                if intersect_point:
                    dist = ray_origin.distance_to(intersect_point)
                    if dist > 1e-5: # Epsilon to avoid self-intersection
                        intersections.append((dist, intersect_point, mirror))
            
            if not intersections:
                break

            dist, point, mirror = min(intersections, key=lambda x: x[0])
            path.append(point)
            
            # SFX: BeamReflect.wav
            ray_origin = point
            ray_dir = ray_dir.reflect(mirror.get_normal())

        # Extend final segment to screen boundary
        end_point = self._get_ray_boundary_intersection(ray_origin, ray_dir)
        path.append(end_point)
        return path

    def _get_path_switch_collisions(self, path):
        collided_indices = set()
        if len(path) < 2:
            return collided_indices
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            for idx, switch in enumerate(self.switches):
                if switch.rect.clipline(p1, p2):
                    collided_indices.add(idx)
        return collided_indices

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render echoes
        for delay in sorted(self.ECHO_DELAYS, reverse=True):
            if len(self.beam_path_history) > delay:
                path = self.beam_path_history[-delay-1]
                alpha = int(100 * (1 - delay / max(self.ECHO_DELAYS)))
                self._render_beam(path, (self.COLOR_ECHO[0], self.COLOR_ECHO[1], self.COLOR_ECHO[2], alpha), 2)
        
        # Render main beam
        if self.beam_path_history:
            main_beam_path = self.beam_path_history[-1]
            # Glow effect
            self._render_beam(main_beam_path, (self.COLOR_BEAM[0], self.COLOR_BEAM[1], self.COLOR_BEAM[2], 60), 7)
            self._render_beam(main_beam_path, self.COLOR_BEAM, 3)

        # Render switches
        for switch in self.switches:
            if switch.is_active:
                # Glow effect for active switches
                glow_size = self.SWITCH_SIZE * (1.5 + 0.5 * math.sin(self.steps * 0.1))
                pygame.gfxdraw.filled_circle(self.screen, int(switch.center.x), int(switch.center.y), int(glow_size), self.COLOR_SWITCH_ACTIVE_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, int(switch.center.x), int(switch.center.y), self.SWITCH_SIZE // 2, self.COLOR_SWITCH_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, int(switch.center.x), int(switch.center.y), self.SWITCH_SIZE // 2, self.COLOR_SWITCH_ACTIVE)
            else:
                pygame.draw.rect(self.screen, self.COLOR_SWITCH_INACTIVE, switch.rect, border_radius=3)

        # Render mirrors
        for i, mirror in enumerate(self.mirrors):
            color = self.COLOR_MIRROR_SELECTED if i == self.selected_mirror_idx and not self.game_over else self.COLOR_MIRROR
            pygame.draw.line(self.screen, color, mirror.p1, mirror.p2, 3)

    def _render_beam(self, path, color, width):
        if len(path) < 2: return
        
        # Use aaline for antialiasing, but it doesn't support width > 1.
        # So we draw multiple lines for thickness, or just use draw.lines for simplicity.
        # For transparent lines, we need a separate surface.
        if len(color) == 4: # Has alpha
            line_surface = self.screen.copy()
            line_surface.fill((0,0,0))
            line_surface.set_colorkey((0,0,0))
            pygame.draw.lines(line_surface, color[:3], False, [(int(p.x), int(p.y)) for p in path], width)
            line_surface.set_alpha(color[3])
            self.screen.blit(line_surface, (0,0))
        else:
            pygame.draw.lines(self.screen, color, False, [(int(p.x), int(p.y)) for p in path], width)

    def _render_ui(self):
        # Switches activated
        active_switches = sum(1 for s in self.switches if s.is_active)
        switch_text = f"Switches: {active_switches}/{self.NUM_SWITCHES}"
        text_surf = self.font_ui.render(switch_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 60.0
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_TIMER_NORMAL
        timer_text = f"{max(0, time_left):05.2f}"
        text_surf = self.font_timer.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "active_switches": sum(1 for s in self.switches if s.is_active),
            "mirror_count": self.mirror_count
        }

    # --- Math Helper Methods ---
    def _get_ray_segment_intersection(self, ray_origin, ray_dir, p1, p2):
        v1 = ray_origin - p1
        v2 = p2 - p1
        v3 = pygame.math.Vector2(-ray_dir.y, ray_dir.x)
        dot_v2_v3 = v2.dot(v3)
        if abs(dot_v2_v3) < 1e-6: return None # Parallel lines

        t1 = v2.cross(v1) / dot_v2_v3
        t2 = v1.dot(v3) / dot_v2_v3

        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return ray_origin + t1 * ray_dir
        return None

    def _get_ray_boundary_intersection(self, ray_origin, ray_dir):
        t_vals = []
        if ray_dir.x != 0:
            t_vals.append((0 - ray_origin.x) / ray_dir.x)
            t_vals.append((self.WIDTH - ray_origin.x) / ray_dir.x)
        if ray_dir.y != 0:
            t_vals.append((0 - ray_origin.y) / ray_dir.y)
            t_vals.append((self.HEIGHT - ray_origin.y) / ray_dir.y)

        # Get smallest positive t-value
        min_t = float('inf')
        for t in t_vals:
            if t > 1e-5:
                p = ray_origin + t * ray_dir
                if 0 <= p.x <= self.WIDTH and 0 <= p.y <= self.HEIGHT:
                    min_t = min(min_t, t)
        
        return ray_origin + min_t * ray_dir if min_t != float('inf') else ray_origin

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Human Playable Demo ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Override Pygame display for human interaction
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Echo Chamber")
    clock = pygame.time.Clock()

    total_reward = 0.0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("Q: Quit")

    action = [3, 0, 0] # Start with a no-op action
    
    while not done:
        movement_action = 3 # no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_SPACE:
                    # Cycle mirror on space press
                    action = [0, 0, 0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    # Reset action to no-op after single press
                    action = [3, 0, 0]

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement_action = 1
            elif keys[pygame.K_DOWN]:
                movement_action = 2
            
            action = [movement_action, 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

        if done:
            print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()