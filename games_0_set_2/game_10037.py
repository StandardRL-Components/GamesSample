import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:43:07.941297
# Source Brief: brief_00037.md
# Brief Index: 37
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
        "Control two robotic arms in mirrored motion to assemble a device by picking up "
        "and placing components in their target locations before time runs out."
    )
    user_guide = (
        "Use arrow keys to move the arms. Press space to operate the left arm's gripper "
        "and left shift to operate the right arm's gripper."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.ARM_SPEED = 5
        self.WIN_SCORE = 200
        self.MAX_TIME_SECONDS = 90
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.GRAB_RADIUS = 25
        self.PLACEMENT_TOLERANCE = 15
        self.ARM_COLLISION_RADIUS = 15
        self.ARM_BASE_Y = self.HEIGHT // 2
        self.ARM_1_BASE_X = 100
        self.ARM_2_BASE_X = self.WIDTH - 100

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_ASSEMBLY_AREA = (25, 35, 50)
        self.COLOR_TARGET_OUTLINE = (60, 70, 90)
        self.COLOR_ARM1 = (0, 191, 255)
        self.COLOR_ARM2 = (255, 165, 0)
        self.COLOR_ARM1_GLOW = (0, 191, 255, 50)
        self.COLOR_ARM2_GLOW = (255, 165, 0, 50)
        self.COLOR_SUCCESS = (0, 255, 127)
        self.COLOR_FAIL = (255, 69, 0)
        self.COLOR_TEXT = (220, 220, 240)
        
        self.PART_COLORS = [
            (255, 0, 128),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 255, 0),   # Yellow
            (128, 0, 255),   # Purple
            (0, 255, 0),     # Green
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.arm1_pos = pygame.Vector2(0, 0)
        self.arm2_pos = pygame.Vector2(0, 0)
        self.arm1_held_part_idx = None
        self.arm2_held_part_idx = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.parts = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Reset arms
        self.arm1_pos = pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.5)
        self.arm2_pos = pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT * 0.5)
        self.arm1_held_part_idx = None
        self.arm2_held_part_idx = None
        self.prev_space_held = False
        self.prev_shift_held = False

        # Reset particles
        self.particles = []

        # Define parts
        self.parts = []
        part_defs = [
            {'shape': 'rect', 'size': (30, 30), 'value': 20, 'target_pos': (self.WIDTH / 2, self.HEIGHT / 2)},
            {'shape': 'circle', 'size': 18, 'value': 30, 'target_pos': (self.WIDTH / 2, self.HEIGHT / 2 - 60)},
            {'shape': 'circle', 'size': 18, 'value': 30, 'target_pos': (self.WIDTH / 2, self.HEIGHT / 2 + 60)},
            {'shape': 'rect', 'size': (40, 15), 'value': 10, 'target_pos': (self.WIDTH / 2 - 80, self.HEIGHT / 2)},
            {'shape': 'rect', 'size': (40, 15), 'value': 10, 'target_pos': (self.WIDTH / 2 + 80, self.HEIGHT / 2)},
        ]

        for i, p_def in enumerate(part_defs):
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(30, 80)
            start_pos = pygame.Vector2(
                p_def['target_pos'][0] + math.cos(angle) * radius,
                p_def['target_pos'][1] + math.sin(angle) * radius
            )
            self.parts.append({
                'id': i,
                'shape': p_def['shape'],
                'color': self.PART_COLORS[i % len(self.PART_COLORS)],
                'size': p_def['size'],
                'pos': start_pos,
                'target_pos': pygame.Vector2(p_def['target_pos']),
                'value': p_def['value'],
                'placed': False,
                'held_by': None
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Calculate Proximity Reward ---
        dist_before = self._get_held_parts_distance()

        # --- Handle Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            move_vec *= self.ARM_SPEED

        self.arm1_pos += move_vec
        self.arm2_pos += pygame.Vector2(-move_vec.x, move_vec.y) # Mirrored X

        # Clamp arm positions
        self.arm1_pos.x = np.clip(self.arm1_pos.x, 0, self.WIDTH)
        self.arm1_pos.y = np.clip(self.arm1_pos.y, 0, self.HEIGHT)
        self.arm2_pos.x = np.clip(self.arm2_pos.x, 0, self.WIDTH)
        self.arm2_pos.y = np.clip(self.arm2_pos.y, 0, self.HEIGHT)

        # --- Handle Grippers (Toggle on Press) ---
        if space_held and not self.prev_space_held: # Arm 1
            reward += self._toggle_gripper(1)
        if shift_held and not self.prev_shift_held: # Arm 2
            reward += self._toggle_gripper(2)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Held Parts ---
        if self.arm1_held_part_idx is not None:
            self.parts[self.arm1_held_part_idx]['pos'] = self.arm1_pos
        if self.arm2_held_part_idx is not None:
            self.parts[self.arm2_held_part_idx]['pos'] = self.arm2_pos
        
        # --- Handle Collisions ---
        if self.arm1_pos.distance_to(self.arm2_pos) < self.ARM_COLLISION_RADIUS * 2:
            reward -= 5
            self._create_sparks((self.arm1_pos + self.arm2_pos) / 2)
            # sfx: collision_spark.wav

        # --- Update Proximity Reward ---
        dist_after = self._get_held_parts_distance()
        reward += (dist_before - dist_after) * 0.1

        # --- Update Particles & Game State ---
        self._update_particles()
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        time_ran_out = self.steps >= self.MAX_STEPS
        win_condition_met = self.score >= self.WIN_SCORE

        if time_ran_out or win_condition_met:
            terminated = True
            self.game_over = True
            if win_condition_met:
                reward += 100
                # sfx: game_win.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_held_parts_distance(self):
        total_dist = 0
        if self.arm1_held_part_idx is not None:
            part = self.parts[self.arm1_held_part_idx]
            total_dist += part['pos'].distance_to(part['target_pos'])
        if self.arm2_held_part_idx is not None:
            part = self.parts[self.arm2_held_part_idx]
            total_dist += part['pos'].distance_to(part['target_pos'])
        return total_dist

    def _toggle_gripper(self, arm_id):
        pos = self.arm1_pos if arm_id == 1 else self.arm2_pos
        held_idx_attr = 'arm1_held_part_idx' if arm_id == 1 else 'arm2_held_part_idx'
        held_idx = getattr(self, held_idx_attr)

        # --- Release Part ---
        if held_idx is not None:
            part = self.parts[held_idx]
            part['held_by'] = None
            setattr(self, held_idx_attr, None)
            
            # Check for placement
            if part['pos'].distance_to(part['target_pos']) < self.PLACEMENT_TOLERANCE:
                part['placed'] = True
                part['pos'] = part['target_pos'] # Snap to target
                self.score += part['value']
                self._create_success_effect(part['target_pos'])
                # sfx: part_place_success.wav
                return part['value']
            else:
                # sfx: part_release_fail.wav
                return 0
        
        # --- Grab Part ---
        else:
            # Find closest, unplaced, unheld part
            best_part_idx = -1
            min_dist = self.GRAB_RADIUS
            
            # Prevent arm 2 from grabbing a part arm 1 is trying to grab in the same frame
            other_arm_held_idx = self.arm2_held_part_idx if arm_id == 1 else self.arm1_held_part_idx

            for i, part in enumerate(self.parts):
                if not part['placed'] and part['held_by'] is None and i != other_arm_held_idx:
                    dist = pos.distance_to(part['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        best_part_idx = i
            
            if best_part_idx != -1:
                self.parts[best_part_idx]['held_by'] = arm_id
                setattr(self, held_idx_attr, best_part_idx)
                # sfx: gripper_grab.wav
        return 0

    def _create_sparks(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': random.choice([self.COLOR_FAIL, (255, 255, 0)]),
                'radius': self.np_random.uniform(1, 3)
            })

    def _create_success_effect(self, pos):
        self.particles.append({
            'pos': pos.copy(),
            'vel': pygame.Vector2(0,0),
            'life': 20,
            'color': self.COLOR_SUCCESS,
            'radius': self.PLACEMENT_TOLERANCE,
            'type': 'glow'
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw assembly area
        pygame.draw.rect(self.screen, self.COLOR_ASSEMBLY_AREA, (150, 50, 340, 300))
        
        # Draw targets and parts
        for part in self.parts:
            # Draw target outline
            if not part['placed']:
                self._draw_dashed_shape(part['shape'], part['target_pos'], part['size'], self.COLOR_TARGET_OUTLINE)
            
            # Draw part
            color = part['color']
            if part['placed']:
                color = tuple(c * 0.7 for c in part['color']) # Muted color when placed
            
            if part['shape'] == 'rect':
                w, h = part['size']
                rect = pygame.Rect(part['pos'].x - w/2, part['pos'].y - h/2, w, h)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
            elif part['shape'] == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, int(part['pos'].x), int(part['pos'].y), int(part['size']), color)
                pygame.gfxdraw.aacircle(self.screen, int(part['pos'].x), int(part['pos'].y), int(part['size']), color)

        # Draw arms
        self._draw_arm(1)
        self._draw_arm(2)
        
        # Draw particles
        for p in self.particles:
            if p.get('type') == 'glow':
                alpha = int(255 * (p['life'] / 20))
                self._draw_glow_circle(p['pos'], p['radius'], (*p['color'], alpha))
            else:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius']))

    def _draw_dashed_shape(self, shape, pos, size, color):
        if shape == 'rect':
            w, h = size
            tl = (pos.x - w/2, pos.y - h/2)
            tr = (pos.x + w/2, pos.y - h/2)
            bl = (pos.x - w/2, pos.y + h/2)
            br = (pos.x + w/2, pos.y + h/2)
            self._draw_dashed_line(tl, tr, color)
            self._draw_dashed_line(tr, br, color)
            self._draw_dashed_line(br, bl, color)
            self._draw_dashed_line(bl, tl, color)
        elif shape == 'circle':
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), color)


    def _draw_dashed_line(self, p1, p2, color, dash_len=5):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        dashes = int(dist / dash_len)
        for i in range(0, dashes, 2):
            start = (x1 + (x2 - x1) * (i / dashes), y1 + (y2 - y1) * (i / dashes))
            end = (x1 + (x2 - x1) * ((i + 1) / dashes), y1 + (y2 - y1) * ((i + 1) / dashes))
            pygame.draw.aaline(self.screen, color, start, end)

    def _draw_arm(self, arm_id):
        if arm_id == 1:
            base_pos = pygame.Vector2(self.ARM_1_BASE_X, self.ARM_BASE_Y)
            tip_pos = self.arm1_pos
            color = self.COLOR_ARM1
            glow_color = self.COLOR_ARM1_GLOW
            held_idx = self.arm1_held_part_idx
        else:
            base_pos = pygame.Vector2(self.ARM_2_BASE_X, self.ARM_BASE_Y)
            tip_pos = self.arm2_pos
            color = self.COLOR_ARM2
            glow_color = self.COLOR_ARM2_GLOW
            held_idx = self.arm2_held_part_idx

        # Draw glow
        pygame.draw.aaline(self.screen, glow_color, base_pos, tip_pos, 10)
        # Draw main arm line
        pygame.draw.line(self.screen, color, base_pos, tip_pos, 4)
        
        # Draw gripper
        gripper_open = held_idx is None
        gripper_angle = 45 if gripper_open else 15
        arm_vec = tip_pos - base_pos
        if arm_vec.length() > 0:
            angle = math.degrees(math.atan2(arm_vec.y, arm_vec.x))
            
            g1_end = tip_pos + pygame.Vector2(15, 0).rotate(angle + gripper_angle)
            g2_end = tip_pos + pygame.Vector2(15, 0).rotate(angle - gripper_angle)
            
            pygame.draw.line(self.screen, color, tip_pos, g1_end, 3)
            pygame.draw.line(self.screen, color, tip_pos, g2_end, 3)

    def _draw_glow_circle(self, pos, radius, color):
        r, g, b, a = color
        for i in range(4):
            current_radius = int(radius * (1 + i * 0.25))
            current_alpha = int(a / (i + 1.5))
            if current_alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), current_radius, (r,g,b,current_alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = max(0, self.MAX_TIME_SECONDS - (self.steps / self.FPS))
        time_color = self.COLOR_FAIL if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "ASSEMBLY COMPLETE" if self.score >= self.WIN_SCORE else "TIME UP"
            msg_color = self.COLOR_SUCCESS if self.score >= self.WIN_SCORE else self.COLOR_FAIL
            
            end_text = self.font_large.render(msg, True, msg_color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use arrow keys for movement, space for arm 1, left shift for arm 2
    running = True
    terminated = False
    
    # Pygame window for human play
    # Un-comment the next line to run with a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.set_caption("Robotic Arm Assembly")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    while running:
        if terminated:
            # Wait a bit then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Get Human Action ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()