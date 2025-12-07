import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:50:57.859775
# Source Brief: brief_00062.md
# Brief Index: 62
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a robotic arm grabs and combines objects to
    synthesize new items. The goal is to collect 10 unique synthesized items.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a robotic arm to grab, drop, and combine basic objects on a conveyor belt to synthesize new items."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the arm. Press space to grab or drop an object. "
        "Hold an object over another and press shift to combine them."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    ARM_SPEED = 15
    OBJECT_SPAWN_RATE = 5  # Spawn a new object every N steps
    OBJECT_RADIUS = 12
    ARM_EFFECTOR_RADIUS = 15
    GRAB_RADIUS = 25
    PARTICLE_LIFETIME = 20
    NUM_UNIQUE_ITEMS_TO_WIN = 10

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_CONVEYOR = (25, 30, 50)
    COLOR_ARM_BASE = (100, 110, 120)
    COLOR_ARM_SHAFT = (140, 150, 160)
    COLOR_ARM_EFFECTOR = (200, 210, 220)
    COLOR_ARM_EFFECTOR_HOLD = (255, 180, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 50, 70, 180) # RGBA for transparency
    
    BASE_OBJECT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    SYNTH_ITEM_COLORS = [
        (0, 255, 255),   # Cyan
        (255, 165, 0),   # Orange
        (128, 0, 128),   # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),     # Dark Green
        (128, 0, 0),     # Maroon
        (0, 0, 128),     # Navy
        (245, 245, 220), # Beige
        (70, 130, 180),  # Steel Blue
        (210, 105, 30),  # Chocolate
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.arm_pos = np.array([0.0, 0.0])
        self.arm_render_pos = np.array([0.0, 0.0])
        self.arm_base_pos = np.array([self.SCREEN_WIDTH / 2, 120.0])
        self.held_object = None
        self.objects_on_belt = []
        self.synthesized_items = set()
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.last_feedback_msg = ""
        self.feedback_timer = 0
        
        # --- Combination Recipes ---
        # (type1, type2) -> synth_type, where type1 <= type2
        self.recipes = {
            (0, 1): 10, (0, 2): 11, (0, 3): 12, (0, 4): 13,
            (1, 2): 14, (1, 3): 15, (1, 4): 16,
            (2, 3): 17, (2, 4): 18,
            (3, 4): 19
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.arm_pos = np.copy(self.arm_base_pos)
        self.arm_render_pos = np.copy(self.arm_base_pos)
        self.held_object = None
        self.objects_on_belt = []
        self.synthesized_items = set()
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.last_feedback_msg = ""
        self.feedback_timer = 0

        # Spawn initial objects
        for _ in range(5):
            self._spawn_object()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        self._handle_movement(movement)
        
        if space_press:
            reward += self._handle_grab_drop()

        if shift_press:
            reward += self._handle_combine()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- Update Game State ---
        self._update_arm_render_pos()
        self._update_particles()
        if self.steps % self.OBJECT_SPAWN_RATE == 0:
            self._spawn_object()
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        
        self.score += reward
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if len(self.synthesized_items) >= self.NUM_UNIQUE_ITEMS_TO_WIN:
            reward += 100.0
            self.score += 100.0
            terminated = True
            self._set_feedback("VICTORY!", 120)
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self._set_feedback("MAX STEPS REACHED", 120)
        
        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement_action):
        delta = np.array([0, 0])
        if movement_action == 1: delta[1] = -1  # Up
        elif movement_action == 2: delta[1] = 1   # Down
        elif movement_action == 3: delta[0] = -1  # Left
        elif movement_action == 4: delta[0] = 1   # Right
        
        self.arm_pos += delta * self.ARM_SPEED
        self.arm_pos[0] = np.clip(self.arm_pos[0], 0, self.SCREEN_WIDTH)
        self.arm_pos[1] = np.clip(self.arm_pos[1], 0, self.SCREEN_HEIGHT)

    def _handle_grab_drop(self):
        if self.held_object is None:
            # Attempt to grab
            for i, obj in enumerate(self.objects_on_belt):
                dist = np.linalg.norm(self.arm_pos - obj['pos'])
                if dist < self.GRAB_RADIUS:
                    self.held_object = self.objects_on_belt.pop(i)
                    self._set_feedback("OBJECT GRABBED", 30)
                    # sfx: grab_sound
                    return 1.0
            self._set_feedback("GRAB FAILED", 30)
            return 0.0
        else:
            # Attempt to drop
            if self.arm_pos[1] > self.SCREEN_HEIGHT - 100: # On conveyor
                self.held_object['pos'] = np.copy(self.arm_pos)
                self.objects_on_belt.append(self.held_object)
                self.held_object = None
                self._set_feedback("OBJECT DROPPED", 30)
                # sfx: drop_sound
                return 0.0
            else:
                self._set_feedback("CANNOT DROP HERE", 30)
                return 0.0

    def _handle_combine(self):
        if self.held_object is None:
            self._set_feedback("MUST HOLD AN OBJECT", 30)
            return 0.0
        
        target_obj_idx = -1
        min_dist = float('inf')
        for i, obj in enumerate(self.objects_on_belt):
            dist = np.linalg.norm(self.arm_pos - obj['pos'])
            if dist < self.GRAB_RADIUS and dist < min_dist:
                min_dist = dist
                target_obj_idx = i
        
        if target_obj_idx != -1:
            target_obj = self.objects_on_belt[target_obj_idx]
            
            # Combine attempt reward
            reward = 2.0
            
            type1 = self.held_object['type']
            type2 = target_obj['type']
            recipe_key = tuple(sorted((type1, type2)))
            
            if recipe_key in self.recipes:
                # sfx: success_synth
                synth_type = self.recipes[recipe_key]
                reward += 5.0
                
                if synth_type not in self.synthesized_items:
                    reward += 10.0
                    self.synthesized_items.add(synth_type)
                    self._set_feedback(f"NEW ITEM SYNTHESIZED: {synth_type}", 60)
                else:
                    self._set_feedback(f"ITEM SYNTHESIZED: {synth_type}", 60)

                self._create_combination_particles(target_obj['pos'], synth_type)
                self.held_object = None
                self.objects_on_belt.pop(target_obj_idx)
            else:
                # sfx: fail_synth
                self._set_feedback("INVALID COMBINATION", 30)
                # No extra reward, but the attempt reward of 2.0 remains
            
            return reward
        
        self._set_feedback("NO OBJECT TO COMBINE WITH", 30)
        return 0.0

    def _spawn_object(self):
        if len(self.objects_on_belt) > 20: return # Prevent overcrowding
        obj_type = self.np_random.integers(0, 5)
        pos = np.array([
            self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
            self.np_random.uniform(self.SCREEN_HEIGHT - 80, self.SCREEN_HEIGHT - 30)
        ])
        self.objects_on_belt.append({'type': obj_type, 'pos': pos, 'pulse': 0.0})

    def _update_arm_render_pos(self):
        # Interpolate for smooth movement
        lerp_factor = 0.25
        self.arm_render_pos += (self.arm_pos - self.arm_render_pos) * lerp_factor

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _create_combination_particles(self, pos, synth_type):
        color = self.SYNTH_ITEM_COLORS[synth_type - 10]
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': np.copy(pos),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': self.np_random.uniform(2, 6),
                'life': self.PARTICLE_LIFETIME,
                'color': color
            })

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
            "items_collected": len(self.synthesized_items),
            "held_object_type": self.held_object['type'] if self.held_object else -1,
        }
        
    def _set_feedback(self, msg, duration):
        self.last_feedback_msg = msg
        self.feedback_timer = duration

    def _render_game(self):
        # Conveyor belt
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, (0, self.SCREEN_HEIGHT - 100, self.SCREEN_WIDTH, 100))
        
        # Objects on belt
        for obj in self.objects_on_belt:
            obj['pulse'] = (obj['pulse'] + 0.1) % (2 * math.pi)
            glow_radius = self.OBJECT_RADIUS + 2 + math.sin(obj['pulse']) * 2
            color = self.BASE_OBJECT_COLORS[obj['type']]
            pos_int = (int(obj['pos'][0]), int(obj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(glow_radius), (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.OBJECT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.OBJECT_RADIUS, color)

        # Particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFETIME))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)

        # Robotic arm
        self._render_arm()

    def _render_arm(self):
        start_pos = (int(self.arm_base_pos[0]), int(self.arm_base_pos[1]))
        end_pos = (int(self.arm_render_pos[0]), int(self.arm_render_pos[1]))
        
        # Arm shaft
        pygame.draw.aaline(self.screen, self.COLOR_ARM_SHAFT, start_pos, end_pos, True)
        pygame.draw.line(self.screen, self.COLOR_ARM_SHAFT, start_pos, end_pos, 3)
        
        # Arm base
        pygame.gfxdraw.aacircle(self.screen, start_pos[0], start_pos[1], 15, self.COLOR_ARM_BASE)
        pygame.gfxdraw.filled_circle(self.screen, start_pos[0], start_pos[1], 15, self.COLOR_ARM_BASE)
        
        # Arm effector (pincers)
        effector_color = self.COLOR_ARM_EFFECTOR_HOLD if self.held_object else self.COLOR_ARM_EFFECTOR
        pincer_angle = 0.5 if self.held_object else 0.8 # radians
        for sign in [-1, 1]:
            angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0]) + sign * pincer_angle
            pincer_end_x = end_pos[0] + self.ARM_EFFECTOR_RADIUS * math.cos(angle)
            pincer_end_y = end_pos[1] + self.ARM_EFFECTOR_RADIUS * math.sin(angle)
            pygame.draw.line(self.screen, effector_color, end_pos, (pincer_end_x, pincer_end_y), 4)

        # Held object
        if self.held_object:
            color = self.BASE_OBJECT_COLORS[self.held_object['type']]
            pygame.gfxdraw.aacircle(self.screen, end_pos[0], end_pos[1], self.OBJECT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], self.OBJECT_RADIUS, color)

    def _render_ui(self):
        # UI Background Panel
        panel = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BG)
        self.screen.blit(panel, (0, 0))

        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(steps_text, (10, 25))
        
        # Feedback message
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30)))
            feedback_surf = self.font_large.render(self.last_feedback_msg, True, self.COLOR_TEXT)
            # Create a new surface with per-pixel alpha for fading
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(feedback_surf, feedback_rect)

        # Collected Items Display
        title_text = self.font_small.render("SYNTHESIZED ITEMS", True, self.COLOR_TEXT)
        self.screen.blit(title_text, (self.SCREEN_WIDTH - 200, 5))

        slot_size = 12
        padding = 5
        start_x = self.SCREEN_WIDTH - (self.NUM_UNIQUE_ITEMS_TO_WIN * (slot_size + padding)) + slot_size // 2
        for i in range(self.NUM_UNIQUE_ITEMS_TO_WIN):
            item_type = 10 + i
            center_x = start_x + i * (slot_size + padding)
            center_y = 35
            if item_type in self.synthesized_items:
                color = self.SYNTH_ITEM_COLORS[i]
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, slot_size // 2, color)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, slot_size // 2, color)
            else:
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, slot_size // 2, self.COLOR_UI_BG)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, (slot_size // 2) -1, self.COLOR_TEXT)

    def close(self):
        pygame.quit()