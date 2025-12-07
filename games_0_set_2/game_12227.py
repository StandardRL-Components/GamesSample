import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:32:39.715365
# Source Brief: brief_02227.md
# Brief Index: 2227
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Fractal Echo is a puzzle game where the player launches energy pulses
    that reflect off portals and time-frozen echoes of previous pulses.
    The goal is to replicate a target fractal pattern by carefully aiming
    the launcher and timing the creation of echoes.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch energy pulses to reflect off portals and time-frozen echoes. "
        "Replicate a target fractal pattern by aiming the launcher and creating echoes."
    )
    user_guide = (
        "Controls: Use arrow keys to aim the launcher (↑↓ for fine, ←→ for coarse). "
        "Press space to fire a pulse and shift to create an echo."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (26, 26, 46) # #1a1a2e
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PULSE = (0, 255, 255)
    COLOR_PORTAL = (224, 64, 251) # Magenta
    COLOR_TARGET = (255, 255, 255) # White
    COLOR_TEXT = (240, 240, 240)
    COLOR_SUCCESS = (0, 255, 127)
    COLOR_FAIL = (255, 82, 82)

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
        self.font_ui = pygame.font.SysFont("monospace", 18)
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_echo = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.render_mode = render_mode
        self.completed_levels = 0

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launcher_angle = 0.0
        self.launcher_pos = (0, 0)
        self.pulses = []
        self.echo_layers = []
        self.generated_segments = []
        self.particles = []
        self.portals = []
        self.target_fractal = []
        self.matched_target_indices = set()
        self.resonance_score = 0.0
        self.remaining_echoes = 0
        self.max_echoes = 0
        self.pulse_cooldown = 0
        self.echo_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.terminal_reason = ""
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_reason = ""
        
        level = self.completed_levels + 1
        self.max_echoes = max(3, 5 - (self.completed_levels // 10))
        num_target_segments = 3 + (self.completed_levels // 5)
        
        self.launcher_angle = 90.0
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        
        self.pulses = []
        self.echo_layers = []
        self.generated_segments = []
        self.particles = []
        self.matched_target_indices = set()
        self.resonance_score = 0.0
        
        self.remaining_echoes = self.max_echoes
        self.pulse_cooldown = 0
        self.echo_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._setup_level(num_target_segments)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_state()
        
        newly_matched = self._update_resonance()
        reward += newly_matched * 0.1
        self.score += newly_matched * 10 # UI score
        
        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        
        if terminated:
            self.game_over = True
            if self.resonance_score >= 99.9:
                reward += 100.0
                self.score += 1000
                self.completed_levels += 1
                self.terminal_reason = "RESONANCE ACHIEVED"
            else:
                reward -= 10.0 # Less harsh penalty
                self.terminal_reason = "FAILURE"
                if self.steps >= self.MAX_STEPS:
                    self.terminal_reason = "TIME LIMIT REACHED"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        angle_speed_fine = 1.0
        angle_speed_coarse = 2.5
        if movement == 1: self.launcher_angle -= angle_speed_fine
        if movement == 2: self.launcher_angle += angle_speed_fine
        if movement == 3: self.launcher_angle -= angle_speed_coarse
        if movement == 4: self.launcher_angle += angle_speed_coarse
        self.launcher_angle %= 360
        
        if space_pressed and not self.prev_space_held and self.pulse_cooldown <= 0:
            self._fire_pulse()
            self.pulse_cooldown = 10
        self.prev_space_held = space_pressed
        
        if shift_pressed and not self.prev_shift_held and self.echo_cooldown <= 0 and self.remaining_echoes > 0 and self.pulses:
            self._create_echo()
            self.echo_cooldown = 15
            self.remaining_echoes -= 1
        self.prev_shift_held = shift_pressed

    def _update_game_state(self):
        self.pulse_cooldown = max(0, self.pulse_cooldown - 1)
        self.echo_cooldown = max(0, self.echo_cooldown - 1)
        
        self._update_particles()
        self._update_pulses()

    def _update_pulses(self):
        pulses_to_remove = []
        for i, pulse in enumerate(self.pulses):
            pulse['life'] -= 1
            if pulse['life'] <= 0 or pulse['bounces'] >= pulse['max_bounces']:
                pulses_to_remove.append(i)
                continue
            
            self._move_and_reflect_pulse(pulse)
            
        for i in sorted(pulses_to_remove, reverse=True):
            self._record_segment(self.pulses[i])
            del self.pulses[i]

    def _move_and_reflect_pulse(self, pulse):
        start_pos = np.array(pulse['pos'], dtype=float)
        velocity = np.array(pulse['vel'], dtype=float)
        
        intersections = []
        
        # Walls
        if velocity[0] != 0:
            t_x1 = (0 - start_pos[0]) / velocity[0]
            if t_x1 > 1e-6: intersections.append((t_x1, np.array([1, 0]), "wall"))
            t_x2 = (self.WIDTH - start_pos[0]) / velocity[0]
            if t_x2 > 1e-6: intersections.append((t_x2, np.array([-1, 0]), "wall"))
        if velocity[1] != 0:
            t_y1 = (0 - start_pos[1]) / velocity[1]
            if t_y1 > 1e-6: intersections.append((t_y1, np.array([0, 1]), "wall"))
            t_y2 = (self.HEIGHT - start_pos[1]) / velocity[1]
            if t_y2 > 1e-6: intersections.append((t_y2, np.array([0, -1]), "wall"))

        # Portals (circles)
        for portal in self.portals:
            p_center = np.array(portal['pos'])
            oc = start_pos - p_center
            a = np.dot(velocity, velocity)
            b = 2 * np.dot(oc, velocity)
            c = np.dot(oc, oc) - portal['radius']**2
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b - math.sqrt(discriminant)) / (2*a)
                if t1 > 1e-6:
                    hit_pos = start_pos + t1 * velocity
                    normal = (hit_pos - p_center) / np.linalg.norm(hit_pos - p_center)
                    intersections.append((t1, normal, "portal"))

        # Echoes (line segments)
        for layer in self.echo_layers:
            for p1, p2 in layer:
                p1, p2 = np.array(p1), np.array(p2)
                v_seg = p2 - p1
                v_perp = np.array([-v_seg[1], v_seg[0]])
                
                denom = np.dot(velocity, v_perp)
                if abs(denom) > 1e-6:
                    t = np.dot(p1 - start_pos, v_perp) / denom
                    if t > 1e-6:
                        hit_pos = start_pos + t * velocity
                        # Check if hit_pos is on the segment
                        d_sq = np.dot(v_seg, v_seg)
                        if d_sq > 1e-6:
                            u = np.dot(hit_pos - p1, v_seg) / d_sq
                            if 0 <= u <= 1:
                                normal = v_perp / np.linalg.norm(v_perp)
                                intersections.append((t, normal, "echo"))

        # Find closest valid intersection
        closest_intersection = None
        min_t = 1.0
        for t, normal, type in intersections:
            if t < min_t:
                min_t = t
                closest_intersection = (t, normal, type)

        if closest_intersection:
            t, normal, type = closest_intersection
            pulse['pos'] = list(start_pos + (t - 1e-5) * velocity) # Move just before impact
            pulse['vel'] = list(velocity - 2 * np.dot(velocity, normal) * normal)
            self._record_segment(pulse, new_pos=pulse['pos'])
            pulse['trail_start'] = pulse['pos'][:]
            pulse['bounces'] += 1
            # SFX: Reflect
            self._spawn_particles(pulse['pos'], type)
        else:
            pulse['pos'] = list(start_pos + velocity)

    def _record_segment(self, pulse, new_pos=None):
        end_pos = new_pos if new_pos is not None else pulse['pos']
        segment = (tuple(pulse['trail_start']), tuple(end_pos))
        if self._segment_length(segment) > 2:
            self.generated_segments.append(segment)

    def _update_resonance(self):
        new_matches = 0
        for i, target_seg in enumerate(self.target_fractal):
            if i in self.matched_target_indices:
                continue
            for gen_seg in self.generated_segments[-20:]: # Check recent segments
                if self._are_segments_similar(gen_seg, target_seg):
                    self.matched_target_indices.add(i)
                    new_matches += 1
                    # SFX: Resonance chime
                    break
        
        if self.target_fractal:
            self.resonance_score = (len(self.matched_target_indices) / len(self.target_fractal)) * 100
        else:
            self.resonance_score = 100.0
        return new_matches

    def _check_termination(self):
        if self.resonance_score >= 99.9: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.remaining_echoes <= 0 and not self.pulses: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_portals()
        self._render_echoes()
        self._render_generated_fractal()
        self._render_pulses()
        self._render_launcher()
        self._render_particles()

    def _render_background(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_portals(self):
        for portal in self.portals:
            pos = (int(portal['pos'][0]), int(portal['pos'][1]))
            radius = int(portal['radius'])
            for i in range(4):
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius - i, (*self.COLOR_PORTAL, 200 - i * 40))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius-4, self.COLOR_BG)

    def _render_echoes(self):
        num_layers = len(self.echo_layers)
        for i, layer in enumerate(self.echo_layers):
            alpha = int(100 * (i + 1) / num_layers)
            color = (*self.COLOR_PULSE[:3], alpha)
            for p1, p2 in layer:
                pygame.draw.aaline(self.screen, color, p1, p2, True)

    def _render_generated_fractal(self):
        for i, seg_idx in enumerate(self.matched_target_indices):
            p1, p2 = self.target_fractal[seg_idx]
            color = self.COLOR_SUCCESS
            pygame.draw.line(self.screen, color, p1, p2, 2)

    def _render_pulses(self):
        for pulse in self.pulses:
            start = pulse['trail_start']
            end = pulse['pos']
            for i in range(5):
                alpha = 255 - i * 50
                pygame.draw.aaline(self.screen, (*self.COLOR_PULSE, alpha), start, end, True)
            pygame.gfxdraw.filled_circle(self.screen, int(end[0]), int(end[1]), 3, self.COLOR_PULSE)

    def _render_launcher(self):
        pos = self.launcher_pos
        angle = self.launcher_angle
        rad = math.radians(angle)
        
        p1 = (pos[0] + 15 * math.cos(rad), pos[1] - 15 * math.sin(rad))
        p2 = (pos[0] + 8 * math.cos(rad + 2.356), pos[1] - 8 * math.sin(rad + 2.356))
        p3 = (pos[0] + 8 * math.cos(rad - 2.356), pos[1] - 8 * math.sin(rad - 2.356))
        
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'], color)

    def _render_ui(self):
        # Target fractal
        target_offset = (20, 20)
        target_scale = 0.15
        for p1, p2 in self.target_fractal:
            sp1 = (target_offset[0] + p1[0] * target_scale, target_offset[1] + p1[1] * target_scale)
            sp2 = (target_offset[0] + p2[0] * target_scale, target_offset[1] + p2[1] * target_scale)
            pygame.draw.aaline(self.screen, self.COLOR_TARGET, sp1, sp2)
        
        # UI Text
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 15))
        
        res_text = self.font_ui.render(f"RESONANCE: {self.resonance_score:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (self.WIDTH - res_text.get_width() - 20, 40))

        # Echo count
        echo_text = self.font_echo.render(f"ECHOES: {self.remaining_echoes}", True, self.COLOR_PLAYER if self.remaining_echoes > 0 else self.COLOR_FAIL)
        self.screen.blit(echo_text, (self.WIDTH // 2 - echo_text.get_width() // 2, self.HEIGHT - 40))
        
        # Game Over message
        if self.game_over:
            color = self.COLOR_SUCCESS if self.resonance_score >= 99.9 else self.COLOR_FAIL
            end_text = self.font_title.render(self.terminal_reason, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 50))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resonance": self.resonance_score,
            "completed_levels": self.completed_levels,
            "remaining_echoes": self.remaining_echoes,
        }
        
    def close(self):
        pygame.quit()

    # --- HELPER METHODS ---
    def _setup_level(self, num_segments):
        self.portals = [
            {'pos': (self.WIDTH * 0.25, self.HEIGHT * 0.5), 'radius': 60},
            {'pos': (self.WIDTH * 0.75, self.HEIGHT * 0.5), 'radius': 60},
        ]
        if self.completed_levels > 2:
            self.portals.append({'pos': (self.WIDTH * 0.5, self.HEIGHT * 0.2), 'radius': 40})
        if self.completed_levels > 5:
            self.portals.append({'pos': (self.WIDTH * 0.5, self.HEIGHT * 0.8), 'radius': 40})
            
        self.target_fractal = self._generate_target_fractal(num_segments)
    
    def _generate_target_fractal(self, num_segments):
        # Simulate pulse reflections to create a solvable target
        original_portals = self.portals
        self.portals = [p.copy() for p in original_portals]
        
        temp_launcher_angle = self.np_random.uniform(0, 360)
        
        sim_pulse = self._create_pulse_object(temp_launcher_angle)
        sim_pulse['max_bounces'] = num_segments
        
        segments = []
        
        while sim_pulse['bounces'] < num_segments and len(segments) < num_segments:
            start_pos = sim_pulse['trail_start']
            self._move_and_reflect_pulse(sim_pulse)
            end_pos = sim_pulse['pos']
            segment = (tuple(start_pos), tuple(end_pos))
            if self._segment_length(segment) > 2:
                segments.append(segment)
        
        self.portals = original_portals # Restore original portals
        return segments

    def _fire_pulse(self):
        # SFX: Pulse fire
        self.pulses.append(self._create_pulse_object(self.launcher_angle))
        
    def _create_pulse_object(self, angle):
        angle_rad = math.radians(angle)
        speed = 8
        vel = [math.cos(angle_rad) * speed, -math.sin(angle_rad) * speed]
        start_pos = [self.launcher_pos[0] + vel[0]*2, self.launcher_pos[1] + vel[1]*2]
        return {
            'pos': start_pos, 'vel': vel, 'life': 400,
            'bounces': 0, 'max_bounces': 15,
            'trail_start': start_pos.copy()
        }

    def _create_echo(self):
        # SFX: Echo activate
        new_echo_layer = []
        for pulse in self.pulses:
            segment = (tuple(pulse['trail_start']), tuple(pulse['pos']))
            if self._segment_length(segment) > 1:
                new_echo_layer.append(segment)
        if new_echo_layer:
            self.echo_layers.append(new_echo_layer)
            # Spawn particles along the new echo lines
            for p1, p2 in new_echo_layer:
                for i in range(10):
                    t = i / 9.0
                    pos = [p1[0] * (1-t) + p2[0] * t, p1[1] * (1-t) + p2[1] * t]
                    self._spawn_particles(pos, "echo_creation", count=1, color=self.COLOR_PLAYER)

    def _spawn_particles(self, pos, type, count=5, color=None):
        if type == "portal":
            color = self.COLOR_PORTAL
        elif type == "echo":
            color = self.COLOR_PULSE
        elif type == "wall":
            color = self.COLOR_TEXT
        
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _segment_length(self, seg):
        p1, p2 = seg
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _segment_midpoint(self, seg):
        p1, p2 = seg
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _segment_angle(self, seg):
        p1, p2 = seg
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def _are_segments_similar(self, seg1, seg2):
        len_thresh = 0.2
        mid_thresh = 15
        angle_thresh = 0.1 # radians
        
        len1 = self._segment_length(seg1)
        len2 = self._segment_length(seg2)
        if len1 < 5 or len2 < 5 or abs(len1 - len2) / max(len1, len2) > len_thresh:
            return False
            
        mid1 = self._segment_midpoint(seg1)
        mid2 = self._segment_midpoint(seg2)
        if math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) > mid_thresh:
            return False

        angle1 = self._segment_angle(seg1)
        angle2 = self._segment_angle(seg2)
        angle_diff = abs(angle1 - angle2)
        if min(angle_diff, 2 * math.pi - angle_diff) > angle_thresh:
            # Check reverse direction
            angle_diff_rev = abs(angle1 - (angle2 + math.pi)) % (2 * math.pi)
            if min(angle_diff_rev, 2 * math.pi - angle_diff_rev) > angle_thresh:
                return False
        
        return True
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for interactive testing.
    # It will not be run by the evaluation system.
    # Set the video driver to a real one for display.
    os.environ["SDL_VIDEODRIVER"] = "" 
    
    env = GameEnv()
    
    # --- Interactive Human Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for interactive mode
    pygame.display.set_caption("Fractal Echo")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # 0=none
        space = 0    # 0=released
        shift = 0    # 0=released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Reason: {env.terminal_reason}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            terminated = False

        clock.tick(env.FPS)
        
    env.close()