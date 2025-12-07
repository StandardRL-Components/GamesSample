import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:30:41.578693
# Source Brief: brief_01140.md
# Brief Index: 1140
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race a futuristic broomstick through a neon track, dodging stun towers. "
        "Use clones as decoys and create portals to shortcut your way to victory."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to accelerate/decelerate and ←→ to turn. "
        "Press SHIFT to create a clone and SPACE to open a portal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.LAPS_TO_WIN = 1

        # Player properties
        self.PLAYER_MAX_SPEED = 6.0
        self.PLAYER_MIN_SPEED = 1.5
        self.PLAYER_ACCELERATION = 0.2
        self.PLAYER_TURN_RATE = 0.07 # radians per step

        # Cooldowns (in steps)
        self.CLONE_COOLDOWN_TOTAL = 150 # 5 seconds
        self.PORTAL_COOLDOWN_TOTAL = 300 # 10 seconds

        # Entity properties
        self.TOWER_RANGE = 120
        self.TOWER_BEAM_WIDTH = 0.1 # radians
        self.STUN_DURATION = 30 # 1 second
        self.CLONE_LIFETIME = 150 # 5 seconds
        self.PORTAL_LIFETIME = 210 # 7 seconds
        self.PORTAL_SHORTCUT_DISTANCE = 150 # pixels

        # Colors
        self.COLOR_BG = (15, 10, 35)
        self.COLOR_TRACK = (40, 30, 70)
        self.COLOR_TRACK_OUTLINE = (60, 50, 100)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_CLONE = (255, 255, 0)
        self.COLOR_CLONE_GLOW = (255, 255, 0, 50)
        self.COLOR_TOWER = (200, 0, 50)
        self.COLOR_TOWER_BEAM = (255, 50, 100, 100)
        self.COLOR_PORTAL = (100, 150, 255)
        self.COLOR_PORTAL_GLOW = (100, 150, 255, 70)
        self.COLOR_FINISH_LINE = (200, 0, 255)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_UI_BG = (25, 20, 50, 180)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # To detect single presses
        self.prev_space_held = False
        self.prev_shift_held = False

        # Initialize state variables
        self.track_waypoints = []
        self.towers = []
        self.player_pos = [0,0] # etc. - all defined in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._create_track_and_towers()

        self.player_pos = list(self.track_waypoints[0])
        self.player_angle = math.atan2(
            self.track_waypoints[1][1] - self.player_pos[1],
            self.track_waypoints[1][0] - self.player_pos[0]
        )
        self.player_velocity = self.PLAYER_MIN_SPEED
        self.player_trail = []
        self.player_stun_timer = 0

        self.clones = []
        self.portals = []
        
        self.clone_cooldown = 0
        self.portal_cooldown = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laps_completed = 0
        self.next_waypoint_idx = 1
        
        self.prev_dist_to_waypoint = self._get_dist_to_waypoint(self.player_pos, self.next_waypoint_idx)

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self.steps += 1
        reward = self._update_game_state(movement, space_press, shift_press)
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_game_state(self, movement, space_press, shift_press):
        step_reward = 0

        # Update cooldowns
        self.clone_cooldown = max(0, self.clone_cooldown - 1)
        self.portal_cooldown = max(0, self.portal_cooldown - 1)

        # Handle player stun
        if self.player_stun_timer > 0:
            self.player_stun_timer -= 1
            movement = 0 # No movement while stunned
        else:
            # Handle actions
            if shift_press and self.clone_cooldown == 0:
                self._create_clone() # sfx: whoosh_clone.wav
                self.clone_cooldown = self.CLONE_COOLDOWN_TOTAL
            if space_press and self.portal_cooldown == 0:
                self._create_portal() # sfx: portal_open.wav
                self.portal_cooldown = self.PORTAL_COOLDOWN_TOTAL

        # Update player movement
        self._update_player(movement)
        
        # Update dynamic entities
        self._update_clones()
        self._update_portals()
        self._update_towers()

        # Check for portal teleportation
        self._check_portal_entry()

        # Check for tower detection
        if self.player_stun_timer == 0 and self._check_tower_detection():
            self.player_stun_timer = self.STUN_DURATION
            step_reward -= 5.0 # Increased penalty for better learning signal
            # sfx: zap_stun.wav
        
        # Calculate progress reward
        dist_to_waypoint = self._get_dist_to_waypoint(self.player_pos, self.next_waypoint_idx)
        progress = self.prev_dist_to_waypoint - dist_to_waypoint
        step_reward += progress * 0.1 # Reward for getting closer
        self.prev_dist_to_waypoint = dist_to_waypoint

        # Check for waypoint completion
        if dist_to_waypoint < 40:
            self.next_waypoint_idx = (self.next_waypoint_idx + 1) % len(self.track_waypoints)
            self.prev_dist_to_waypoint = self._get_dist_to_waypoint(self.player_pos, self.next_waypoint_idx)
            step_reward += 0.5 # Small bonus for hitting a waypoint
            
            # Check for lap completion
            if self.next_waypoint_idx == 1: # Passed the finish line (waypoint 0)
                self.laps_completed += 1
                step_reward += 50.0 # Big reward for completing a lap
                # sfx: lap_complete.wav
                if self.laps_completed >= self.LAPS_TO_WIN:
                    self.game_over = True
                    step_reward += 100.0 # Victory bonus

        # Penalty for being off-track
        if self._get_distance_from_track_centerline(self.player_pos) > 40:
            step_reward -= 0.1

        return step_reward

    def _update_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Accelerate
            self.player_velocity = min(self.PLAYER_MAX_SPEED, self.player_velocity + self.PLAYER_ACCELERATION)
        elif movement == 2: # Decelerate
            self.player_velocity = max(self.PLAYER_MIN_SPEED, self.player_velocity - self.PLAYER_ACCELERATION)
        elif movement == 3: # Turn left
            self.player_angle -= self.PLAYER_TURN_RATE
        elif movement == 4: # Turn right
            self.player_angle += self.PLAYER_TURN_RATE
        
        # Keep angle in [-pi, pi]
        self.player_angle = (self.player_angle + math.pi) % (2 * math.pi) - math.pi

        # Apply slowdown if off-track
        track_dist = self._get_distance_from_track_centerline(self.player_pos)
        if track_dist > 40:
            slowdown_factor = max(0.5, 1 - (track_dist - 40) / 50)
            effective_velocity = self.player_velocity * slowdown_factor
        else:
            effective_velocity = self.player_velocity

        # Update position
        self.player_pos[0] += math.cos(self.player_angle) * effective_velocity
        self.player_pos[1] += math.sin(self.player_angle) * effective_velocity

        # Clamp position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # Update trail
        self.player_trail.append(list(self.player_pos))
        if len(self.player_trail) > 20:
            self.player_trail.pop(0)

    def _create_clone(self):
        self.clones.append({
            'pos': list(self.player_pos),
            'angle': self.player_angle,
            'velocity': self.PLAYER_MAX_SPEED * 0.8,
            'lifetime': self.CLONE_LIFETIME,
            'trail': []
        })

    def _update_clones(self):
        for clone in self.clones[:]:
            clone['lifetime'] -= 1
            if clone['lifetime'] <= 0:
                self.clones.remove(clone)
                continue
            
            clone['pos'][0] += math.cos(clone['angle']) * clone['velocity']
            clone['pos'][1] += math.sin(clone['angle']) * clone['velocity']
            
            clone['trail'].append(list(clone['pos']))
            if len(clone['trail']) > 15:
                clone['trail'].pop(0)

    def _create_portal(self):
        entry_pos = list(self.player_pos)
        
        # Find point on track ahead of player
        target_waypoint = self.track_waypoints[self.next_waypoint_idx]
        next_waypoint = self.track_waypoints[(self.next_waypoint_idx + 1) % len(self.track_waypoints)]
        
        # Project a point ahead on the track segment
        vec_track = (next_waypoint[0] - target_waypoint[0], next_waypoint[1] - target_waypoint[1])
        vec_mag = math.hypot(*vec_track)
        if vec_mag == 0: return # Avoid division by zero
        
        unit_vec = (vec_track[0] / vec_mag, vec_track[1] / vec_mag)
        
        # Find closest point on line from player
        vec_player_to_target = (target_waypoint[0] - self.player_pos[0], target_waypoint[1] - self.player_pos[1])
        dot_product = vec_player_to_target[0] * unit_vec[0] + vec_player_to_target[1] * unit_vec[1]
        
        closest_point_on_line = (
            target_waypoint[0] - dot_product * unit_vec[0],
            target_waypoint[1] - dot_product * unit_vec[1]
        )
        
        exit_pos = [
            closest_point_on_line[0] + unit_vec[0] * self.PORTAL_SHORTCUT_DISTANCE,
            closest_point_on_line[1] + unit_vec[1] * self.PORTAL_SHORTCUT_DISTANCE
        ]
        
        self.portals.append({
            'entry': entry_pos,
            'exit': exit_pos,
            'lifetime': self.PORTAL_LIFETIME
        })

    def _update_portals(self):
        for portal in self.portals[:]:
            portal['lifetime'] -= 1
            if portal['lifetime'] <= 0:
                self.portals.remove(portal)

    def _check_portal_entry(self):
        for portal in self.portals:
            if math.hypot(self.player_pos[0] - portal['entry'][0], self.player_pos[1] - portal['entry'][1]) < 20:
                self.player_pos = list(portal['exit'])
                portal['lifetime'] = 0 # Portal is consumed
                # sfx: whoosh_teleport.wav
                # Update progress tracking to avoid huge reward jump
                self.prev_dist_to_waypoint = self._get_dist_to_waypoint(self.player_pos, self.next_waypoint_idx)
                break

    def _update_towers(self):
        # Difficulty scaling
        difficulty_factor = 1.0 + (self.steps / 1000) * 0.1
        for tower in self.towers:
            tower['angle'] = (tower['angle'] + tower['speed'] * difficulty_factor) % (2 * math.pi)

    def _check_tower_detection(self):
        for tower in self.towers:
            dist = math.hypot(self.player_pos[0] - tower['pos'][0], self.player_pos[1] - tower['pos'][1])
            if dist > self.TOWER_RANGE:
                continue

            angle_to_player = math.atan2(self.player_pos[1] - tower['pos'][1], self.player_pos[0] - tower['pos'][0])
            
            # Normalize angle difference
            angle_diff = (tower['angle'] - angle_to_player + math.pi) % (2 * math.pi) - math.pi
            
            if abs(angle_diff) < self.TOWER_BEAM_WIDTH:
                return True
        return False

    def _check_termination(self):
        return self.game_over

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
            "laps": self.laps_completed,
            "stunned": self.player_stun_timer > 0
        }

    def _render_game(self):
        self._render_track()
        self._render_portals()
        self._render_clones()
        self._render_towers()
        self._render_player()

    def _render_track(self):
        # Draw wide track path
        pygame.draw.lines(self.screen, self.COLOR_TRACK, True, self.track_waypoints, width=80)
        pygame.draw.lines(self.screen, self.COLOR_TRACK_OUTLINE, True, self.track_waypoints, width=3)
        
        # Draw finish line
        p1 = self.track_waypoints[0]
        p2 = self.track_waypoints[-1]
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0]) + math.pi / 2
        
        for i in range(-40, 41, 10):
            start_x = mid[0] + i * math.cos(angle)
            start_y = mid[1] + i * math.sin(angle)
            end_x = start_x + 10 * math.cos(angle - math.pi/2)
            end_y = start_y + 10 * math.sin(angle - math.pi/2)
            color = self.COLOR_FINISH_LINE if (i // 10) % 2 == 0 else self.COLOR_BG
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 5)


    def _render_player(self):
        self._render_broomstick(
            self.player_pos, self.player_angle, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW,
            self.player_trail, self.player_stun_timer > 0
        )

    def _render_clones(self):
        for clone in self.clones:
            self._render_broomstick(
                clone['pos'], clone['angle'], self.COLOR_CLONE, self.COLOR_CLONE_GLOW,
                clone['trail']
            )

    def _render_broomstick(self, pos, angle, color, glow_color, trail, is_stunned=False):
        # Trail
        if len(trail) > 1:
            for i, p in enumerate(trail):
                alpha = int(255 * (i / len(trail)))
                radius = int(5 * (i / len(trail)))
                if radius > 0:
                    trail_color = (color[0], color[1], color[2], alpha)
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, trail_color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(p[0] - radius), int(p[1] - radius)))

        # Glow
        glow_radius = 20
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)))
        
        # Body
        broom_len = 20
        p1 = (pos[0] + broom_len/2 * math.cos(angle), pos[1] + broom_len/2 * math.sin(angle))
        p2 = (pos[0] - broom_len/2 * math.cos(angle) + 5 * math.cos(angle + math.pi/2),
              pos[1] - broom_len/2 * math.sin(angle) + 5 * math.sin(angle + math.pi/2))
        p3 = (pos[0] - broom_len/2 * math.cos(angle) - 5 * math.cos(angle + math.pi/2),
              pos[1] - broom_len/2 * math.sin(angle) - 5 * math.sin(angle + math.pi/2))
        
        points = [p1, p2, p3]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        
        # Stun effect
        if is_stunned:
            for _ in range(3):
                offset_angle = random.uniform(0, 2 * math.pi)
                offset_dist = random.uniform(5, 15)
                start_pos = (pos[0] + offset_dist * math.cos(offset_angle),
                             pos[1] + offset_dist * math.sin(offset_angle))
                end_pos = (start_pos[0] + random.uniform(-5,5), start_pos[1] + random.uniform(-5,5))
                pygame.draw.line(self.screen, self.COLOR_CLONE, start_pos, end_pos, 2)


    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            # Base
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_TOWER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_TOWER)
            
            # Beam
            beam_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            p1 = tower['pos']
            p2 = (tower['pos'][0] + self.TOWER_RANGE * math.cos(tower['angle'] - self.TOWER_BEAM_WIDTH),
                  tower['pos'][1] + self.TOWER_RANGE * math.sin(tower['angle'] - self.TOWER_BEAM_WIDTH))
            p3 = (tower['pos'][0] + self.TOWER_RANGE * math.cos(tower['angle'] + self.TOWER_BEAM_WIDTH),
                  tower['pos'][1] + self.TOWER_RANGE * math.sin(tower['angle'] + self.TOWER_BEAM_WIDTH))
            
            int_points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.filled_polygon(beam_surf, int_points, self.COLOR_TOWER_BEAM)
            self.screen.blit(beam_surf, (0,0))

    def _render_portals(self):
        for portal in self.portals:
            life_ratio = portal['lifetime'] / self.PORTAL_LIFETIME
            self._render_portal_effect(portal['entry'], life_ratio)
            self._render_portal_effect(portal['exit'], life_ratio)
            
            # Connection line
            alpha = int(100 * life_ratio)
            pygame.draw.line(self.screen, (*self.COLOR_PORTAL, alpha), portal['entry'], portal['exit'], 1)

    def _render_portal_effect(self, pos, life_ratio):
        radius = int(20 * life_ratio)
        if radius < 2: return
        
        # Glow
        glow_radius = int(radius * 1.5)
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PORTAL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)))
        
        # Swirl
        num_arcs = 4
        for i in range(num_arcs):
            angle_offset = (self.steps * 0.1 + (i * 2 * math.pi / num_arcs))
            rect = (int(pos[0]-radius), int(pos[1]-radius), radius*2, radius*2)
            pygame.draw.arc(self.screen, self.COLOR_PORTAL, rect, angle_offset, angle_offset + math.pi/2, 3)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, self.HEIGHT - 60))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.HEIGHT - 50))
        
        # Laps
        lap_text = self.font_large.render(f"LAP: {self.laps_completed + 1}/{self.LAPS_TO_WIN}", True, self.COLOR_TEXT)
        lap_rect = lap_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 35))
        self.screen.blit(lap_text, lap_rect)

        # Cooldowns
        clone_cd_ratio = self.clone_cooldown / self.CLONE_COOLDOWN_TOTAL
        portal_cd_ratio = self.portal_cooldown / self.PORTAL_COOLDOWN_TOTAL
        
        self._render_cooldown_bar(self.WIDTH - 160, self.HEIGHT - 45, "CLONE (SHIFT)", 1-clone_cd_ratio, self.COLOR_CLONE)
        self._render_cooldown_bar(self.WIDTH - 160, self.HEIGHT - 25, "PORTAL (SPACE)", 1-portal_cd_ratio, self.COLOR_PORTAL)

    def _render_cooldown_bar(self, x, y, label, ratio, color):
        label_text = self.font_small.render(label, True, self.COLOR_TEXT)
        self.screen.blit(label_text, (x, y))
        
        bar_x = x + 100
        bar_w = 50
        bar_h = 10
        
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (bar_x, y, bar_w, bar_h))
        if ratio > 0:
            pygame.draw.rect(self.screen, color, (bar_x, y, int(bar_w * ratio), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, y, bar_w, bar_h), 1)

    def _create_track_and_towers(self):
        self.track_waypoints = [
            (100, 200), (250, 100), (450, 100),
            (540, 200), (450, 300), (250, 300),
        ]
        self.towers = [
            {'pos': [350, 200], 'angle': 0, 'speed': 0.01},
            {'pos': [150, 120], 'angle': math.pi, 'speed': 0.015},
            {'pos': [500, 280], 'angle': math.pi/2, 'speed': 0.012}
        ]
    
    def _get_dist_to_waypoint(self, pos, waypoint_idx):
        wx, wy = self.track_waypoints[waypoint_idx]
        return math.hypot(pos[0] - wx, pos[1] - wy)

    def _get_distance_from_track_centerline(self, pos):
        min_dist = float('inf')
        for i in range(len(self.track_waypoints)):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % len(self.track_waypoints)]
            
            # Vector from p1 to p2
            line_vec = (p2[0] - p1[0], p2[1] - p1[1])
            line_mag_sq = line_vec[0]**2 + line_vec[1]**2
            if line_mag_sq == 0: continue

            # Vector from p1 to pos
            p_vec = (pos[0] - p1[0], pos[1] - p1[1])

            # Projection of p_vec onto line_vec
            t = (p_vec[0] * line_vec[0] + p_vec[1] * line_vec[1]) / line_mag_sq
            t = max(0, min(1, t)) # Clamp to segment

            closest_point = (p1[0] + t * line_vec[0], p1[1] + t * line_vec[1])
            dist = math.hypot(pos[0] - closest_point[0], pos[1] - closest_point[1])
            min_dist = min(min_dist, dist)
        
        return min_dist

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and is not used by the evaluation system.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Override Pygame screen for direct display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Broomstick Racer")
    
    total_reward = 0
    total_steps = 0
    
    while not done:
        # Action mapping for human control
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        total_steps += 1
        
        # Render to the display window
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        env.screen.blit(surf, (0,0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    print(f"Game Over! Score: {total_reward:.2f} in {total_steps} steps.")
    env.close()