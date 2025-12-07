import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A futuristic bowling game set in a warped, neon-lit alley. "
        "Master different ball types to score points and unlock new abilities."
    )
    user_guide = (
        "Controls: ←→ to move, ↑↓ to adjust angle. "
        "Hold space to charge power and release to launch. Press shift to cycle ball types."
    )
    auto_advance = True

    # Class-level state for unlocks persisting across episodes in a single run
    unlocked_ball_types = {"standard"}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_SCORE = 500
        self.MAX_EPISODE_STEPS = 1500
        self.LANE_Y_START = 80
        self.LANE_Y_END = 350
        self.PIN_SETUP_Y = self.LANE_Y_START + 40
        self.BALL_START_Y = self.LANE_Y_END - 15

        # --- Colors ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_LANE = (20, 15, 50)
        self.COLOR_LANE_GRID = (40, 30, 80)
        self.COLOR_BALL = (0, 150, 255)
        self.COLOR_BALL_GLOW = (0, 100, 200)
        self.COLOR_PIN = (200, 200, 220)
        self.COLOR_PIN_GLOW = (150, 150, 170)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_POWER_BAR = (255, 50, 50)
        self.COLOR_POWER_BAR_BG = (80, 20, 20)
        self.COLOR_SCORE_PLUS = (50, 255, 50)
        self.COLOR_SCORE_MINUS = (255, 50, 50)

        # --- Physics & Gameplay ---
        self.BALL_SPECS = {
            "standard": {"mass": 10, "friction": 0.99, "radius": 10, "color": self.COLOR_BALL},
            "heavy": {"mass": 15, "friction": 0.985, "radius": 12, "color": (255, 100, 0)},
            "slick": {"mass": 8, "friction": 0.995, "radius": 9, "color": (150, 0, 255)},
        }
        self.PIN_RADIUS = 6
        self.PIN_MASS = 2
        self.MAX_POWER = 15
        self.POWER_CHARGE_RATE = 0.25
        self.ANGLE_ADJUST_RATE = 0.02
        self.POS_ADJUST_RATE = 2

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 18, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball = None
        self.pins = []
        self.particles = []
        self.floating_scores = []
        self.game_phase = "aiming" # aiming, rolling, scoring
        self.frame = 0
        self.roll_in_frame = 0
        self.pins_down_this_roll = 0
        self.pins_down_this_frame = 0
        self.launch_power = 0
        self.launch_angle = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.ball_stuck_counter = 0
        self.pin_layout_complexity = 0
        self.available_ball_types = []
        self.current_ball_type_idx = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.frame = 1
        self.roll_in_frame = 1
        self.pins_down_this_frame = 0
        self.pin_layout_complexity = self.score // 500
        
        self.available_ball_types = sorted(list(self.unlocked_ball_types))
        self.current_ball_type_idx = 0
        
        self._setup_pins()
        self._reset_ball_for_aiming()
        
        self.particles = []
        self.floating_scores = []
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        if self.game_phase == "aiming":
            self._handle_aiming(movement, space_held, shift_held)
            space_released = self.prev_space_held and not space_held
            if space_released and self.launch_power > 1:
                self._launch_ball()

        elif self.game_phase == "rolling":
            self._update_physics()
            if self._is_roll_over():
                reward = self._process_scoring()
                self.game_phase = "scoring"

        elif self.game_phase == "scoring":
            self._advance_frame()
            if not self.game_over:
                self._reset_ball_for_aiming()
            
        self.steps += 1
        self._update_effects()
        self.prev_space_held = space_held
        
        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on time limits in the same way
        if self.steps >= self.MAX_EPISODE_STEPS and not terminated:
            terminated = True # Terminate if max steps are reached
            reward += -100 if self.score < self.TARGET_SCORE else 100
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_aiming(self, movement, space_held, shift_held):
        # Adjust angle
        if movement == 1: self.launch_angle = min(0.5, self.launch_angle + self.ANGLE_ADJUST_RATE)
        elif movement == 2: self.launch_angle = max(-0.5, self.launch_angle - self.ANGLE_ADJUST_RATE)
        # Adjust position
        elif movement == 3: self.ball['pos'][0] = max(self.WIDTH * 0.2, self.ball['pos'][0] - self.POS_ADJUST_RATE)
        elif movement == 4: self.ball['pos'][0] = min(self.WIDTH * 0.8, self.ball['pos'][0] + self.POS_ADJUST_RATE)
        
        # Charge power
        if space_held:
            self.launch_power = min(self.MAX_POWER, self.launch_power + self.POWER_CHARGE_RATE)
        
        # Cycle ball type
        if shift_held and not self.prev_shift_held:
            self.current_ball_type_idx = (self.current_ball_type_idx + 1) % len(self.available_ball_types)
            self._reset_ball_for_aiming() # Update ball properties
        self.prev_shift_held = shift_held

    def _launch_ball(self):
        self.game_phase = "rolling"
        self.ball['vel'][0] = self.launch_power * math.sin(self.launch_angle)
        self.ball['vel'][1] = -self.launch_power * math.cos(self.launch_angle)
        self.launch_power = 0
        self.ball_stuck_counter = 0

    def _update_physics(self):
        # --- Update Ball ---
        ball_spec = self.BALL_SPECS[self.ball['type']]
        # Apply gravity warp
        warp_factor = 0.03
        normalized_x = (self.ball['pos'][0] - self.WIDTH / 2) / (self.WIDTH / 2)
        self.ball['vel'][0] -= warp_factor * normalized_x
        
        # Update position and apply friction
        self.ball['pos'][0] += self.ball['vel'][0]
        self.ball['pos'][1] += self.ball['vel'][1]
        self.ball['vel'][0] *= ball_spec['friction']
        self.ball['vel'][1] *= ball_spec['friction']
        self.ball['angle'] += self.ball['vel'][0] * 0.1

        # Wall collisions
        if not (ball_spec['radius'] < self.ball['pos'][0] < self.WIDTH - ball_spec['radius']):
            self.ball['vel'][0] *= -0.8
            self.ball['pos'][0] = np.clip(self.ball['pos'][0], ball_spec['radius'], self.WIDTH - ball_spec['radius'])

        # --- Update Pins ---
        for pin in self.pins:
            if not pin['standing']:
                pin['pos'] += pin['vel']
                pin['vel'] *= 0.97 # Pin friction
                pin['angle'] += pin['ang_vel']
                pin['ang_vel'] *= 0.98

        # --- Collision Detection ---
        # Ball-Pin
        for pin in self.pins:
            if pin['standing']:
                dist = np.linalg.norm(self.ball['pos'] - pin['pos'])
                if dist < ball_spec['radius'] + self.PIN_RADIUS:
                    self._handle_collision(self.ball, pin, ball_spec['mass'])

        # Pin-Pin
        for i in range(len(self.pins)):
            for j in range(i + 1, len(self.pins)):
                p1, p2 = self.pins[i], self.pins[j]
                if not p1['standing'] or not p2['standing']: # Only check moving pins against others
                    dist = np.linalg.norm(p1['pos'] - p2['pos'])
                    if dist < self.PIN_RADIUS * 2:
                        self._handle_collision(p1, p2, self.PIN_MASS, is_pin_pin=True)

    def _handle_collision(self, obj1, obj2, mass1, is_pin_pin=False):
        # Simplified impulse resolution
        collision_vec = obj1['pos'] - obj2['pos']
        dist = np.linalg.norm(collision_vec)
        if dist == 0: return
        
        normal = collision_vec / dist
        relative_vel = obj1['vel'] - obj2['vel']
        vel_along_normal = np.dot(relative_vel, normal)

        if vel_along_normal > 0: return

        restitution = 0.7
        mass2 = self.PIN_MASS
        impulse_scalar = -(1 + restitution) * vel_along_normal / (1/mass1 + 1/mass2)
        
        impulse = impulse_scalar * normal
        obj1['vel'] += impulse / mass1
        obj2['vel'] -= impulse / mass2

        if not is_pin_pin: # Ball-Pin collision
            obj2['standing'] = False
            obj2['ang_vel'] = (random.random() - 0.5) * 0.5
            self._create_particles(obj2['pos'])
        else: # Pin-Pin collision
            if np.linalg.norm(obj1['vel']) > 0.1 and obj1['standing']:
                obj1['standing'] = False
                obj1['ang_vel'] = (random.random() - 0.5) * 0.3
            if np.linalg.norm(obj2['vel']) > 0.1 and obj2['standing']:
                obj2['standing'] = False
                obj2['ang_vel'] = (random.random() - 0.5) * 0.3


    def _is_roll_over(self):
        ball_speed = np.linalg.norm(self.ball['vel'])
        pins_moving = any(np.linalg.norm(p['vel']) > 0.05 for p in self.pins if not p['standing'])

        if ball_speed < 0.05 and not pins_moving:
            self.ball_stuck_counter += 1
        else:
            self.ball_stuck_counter = 0

        off_screen = self.ball['pos'][1] < 0 or self.ball['pos'][1] > self.HEIGHT
        return off_screen or self.ball_stuck_counter > 50

    def _process_scoring(self):
        newly_downed_pins = [p for p in self.pins if not p['standing'] and p['was_standing']]
        self.pins_down_this_roll = len(newly_downed_pins)
        self.pins_down_this_frame += self.pins_down_this_roll
        
        for pin in newly_downed_pins:
            pin['was_standing'] = False

        # --- Calculate Reward ---
        reward = self.pins_down_this_roll * 0.1
        is_strike = self.roll_in_frame == 1 and self.pins_down_this_frame == 10
        is_spare = self.roll_in_frame == 2 and self.pins_down_this_frame == 10

        score_change = self.pins_down_this_roll * 10
        if is_strike:
            score_change += 10 # Bonus for clearing in one
            reward += 2.0
        if is_spare:
            score_change += 5 # Bonus for clearing in two
            reward += 1.0

        self.score += score_change
        if score_change > 0:
            self._add_floating_score(score_change)
        
        # Check for ball unlocks
        if self.score >= 200 and "heavy" not in self.unlocked_ball_types:
            self.unlocked_ball_types.add("heavy")
            self.available_ball_types = sorted(list(self.unlocked_ball_types))
        if self.score >= 400 and "slick" not in self.unlocked_ball_types:
            self.unlocked_ball_types.add("slick")
            self.available_ball_types = sorted(list(self.unlocked_ball_types))

        # Terminal reward check
        if self.frame == 10:
            if self.score >= self.TARGET_SCORE:
                reward += 100
                self._add_floating_score(100, self.COLOR_SCORE_PLUS)
            else:
                reward -= 100
                self._add_floating_score(-100, self.COLOR_SCORE_MINUS)

        return reward

    def _advance_frame(self):
        is_strike = self.roll_in_frame == 1 and self.pins_down_this_frame == 10
        
        if self.frame >= 10:
            self.game_over = True
            self.game_phase = "game_over"
            return

        if is_strike or self.roll_in_frame == 2:
            self.frame += 1
            self.roll_in_frame = 1
            self.pins_down_this_frame = 0
            self._setup_pins()
        else:
            self.roll_in_frame = 2
        
        self.game_phase = "aiming"

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
            "frame": self.frame,
            "roll": self.roll_in_frame,
            "ball_type": self.ball['type']
        }

    def _render_game(self):
        # Draw lane
        pygame.draw.rect(self.screen, self.COLOR_LANE, (0, self.LANE_Y_START, self.WIDTH, self.LANE_Y_END - self.LANE_Y_START))
        
        # Draw warped grid lines
        for i in range(11):
            y = self.LANE_Y_START + i * (self.LANE_Y_END - self.LANE_Y_START) / 10
            points = []
            for x in range(0, self.WIDTH + 1, 10):
                offset = 20 * math.sin(x / 100 + y / 50) * ((y - self.LANE_Y_START) / (self.LANE_Y_END - self.LANE_Y_START))**2
                points.append((x + offset, y))
            pygame.draw.aalines(self.screen, self.COLOR_LANE_GRID, False, points)

        # Draw pins
        for pin in self.pins:
            self._draw_glowing_shape('pin', pin['pos'], self.COLOR_PIN, self.COLOR_PIN_GLOW, self.PIN_RADIUS, pin['angle'])

        # Draw ball
        ball_spec = self.BALL_SPECS[self.ball['type']]
        self._draw_glowing_shape('circle', self.ball['pos'], ball_spec['color'], self.COLOR_BALL_GLOW, ball_spec['radius'], self.ball['angle'])

        # Draw particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = (
                int(self.COLOR_BALL[0] * alpha + self.COLOR_PIN[0] * (1 - alpha)),
                int(self.COLOR_BALL[1] * alpha + self.COLOR_PIN[1] * (1 - alpha)),
                int(self.COLOR_BALL[2] * alpha + self.COLOR_PIN[2] * (1 - alpha)),
            )
            pygame.draw.circle(self.screen, color, p['pos'], int(p['size'] * alpha))

    def _render_ui(self):
        # Aiming indicator
        if self.game_phase == "aiming":
            start_pos = self.ball['pos']
            end_pos = (
                start_pos[0] + 50 * math.sin(self.launch_angle),
                start_pos[1] - 50 * math.cos(self.launch_angle)
            )
            pygame.draw.aaline(self.screen, self.COLOR_POWER_BAR, start_pos, end_pos, 2)
            
            # Power bar
            bar_width = 100
            bar_height = 15
            bar_x = self.WIDTH/2 - bar_width/2
            bar_y = self.HEIGHT - 25
            fill_width = (self.launch_power / self.MAX_POWER) * bar_width
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR, (bar_x, bar_y, fill_width, bar_height))

        # Text UI
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        frame_text = self.font_large.render(f"FRAME: {self.frame}/10", True, self.COLOR_UI_TEXT)
        self.screen.blit(frame_text, (self.WIDTH - frame_text.get_width() - 10, 10))
        
        roll_text = self.font_small.render(f"ROLL: {self.roll_in_frame}", True, self.COLOR_UI_TEXT)
        self.screen.blit(roll_text, (self.WIDTH - roll_text.get_width() - 10, 40))

        ball_type_text = self.font_small.render(f"BALL: {self.ball['type'].upper()}", True, self.BALL_SPECS[self.ball['type']]['color'])
        self.screen.blit(ball_type_text, (10, 40))
        
        # Floating scores
        for fs in self.floating_scores:
            text = self.font_score.render(f"{'+' if fs['val'] > 0 else ''}{fs['val']}", True, fs['color'])
            text.set_alpha(fs['alpha'])
            self.screen.blit(text, fs['pos'])

    def _draw_glowing_shape(self, shape_type, pos, color, glow_color, radius, angle=0):
        # Draw multiple layers for a soft glow
        for i in range(4, 0, -1):
            glow_radius = radius + i * 2
            alpha = 80 - i * 18
            current_glow_color = (*glow_color, alpha)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            if shape_type == 'circle':
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, current_glow_color)
            elif shape_type == 'pin':
                self._draw_pin_shape(temp_surf, glow_radius, glow_radius, glow_radius, current_glow_color, angle)
            self.screen.blit(temp_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)))
        
        # Draw main shape
        if shape_type == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color)
        elif shape_type == 'pin':
            surf = pygame.Surface((radius*2, radius*4), pygame.SRCALPHA)
            self._draw_pin_shape(surf, radius, radius*2, radius, color)
            rotated_surf = pygame.transform.rotate(surf, math.degrees(angle))
            rect = rotated_surf.get_rect(center=(int(pos[0]), int(pos[1])))
            self.screen.blit(rotated_surf, rect)

    def _draw_pin_shape(self, surface, cx, cy, r, color, angle=0):
        # Simplified pin shape: circle on top of a trapezoid
        h = r * 2.5
        pygame.draw.circle(surface, color, (cx, cy-h/2), int(r*0.8))
        points = [
            (cx - r, cy + h/2), (cx + r, cy + h/2),
            (cx + r*0.7, cy - h/2), (cx - r*0.7, cy - h/2)
        ]
        pygame.draw.polygon(surface, color, points)


    def _setup_pins(self):
        self.pins = []
        spacing_mod = 1.0 - min(0.3, (self.score // 500) * 0.05)
        
        pin_layout = [
            (0, 0),
            (-1, -1), (1, -1),
            (-2, -2), (0, -2), (2, -2),
            (-3, -3), (-1, -3), (1, -3), (3, -3)
        ]
        
        for dx, dy in pin_layout:
            x = self.WIDTH / 2 + dx * self.PIN_RADIUS * 2.5 * spacing_mod
            y = self.PIN_SETUP_Y - dy * self.PIN_RADIUS * 3.5 * spacing_mod
            self.pins.append({
                'pos': np.array([x, y], dtype=float),
                'vel': np.array([0, 0], dtype=float),
                'angle': 0.0,
                'ang_vel': 0.0,
                'standing': True,
                'was_standing': True
            })

    def _reset_ball_for_aiming(self):
        ball_type = self.available_ball_types[self.current_ball_type_idx]
        ball_spec = self.BALL_SPECS[ball_type]
        start_x = self.ball['pos'][0] if self.ball else self.WIDTH / 2
        
        self.ball = {
            'pos': np.array([start_x, self.BALL_START_Y], dtype=float),
            'vel': np.array([0, 0], dtype=float),
            'angle': 0.0,
            'type': ball_type
        }
        self.launch_power = 0
        self.game_phase = "aiming"

    def _create_particles(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': random.randint(20, 40),
                'max_life': 40,
                'size': random.uniform(1, 4)
            })

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            
        # Update floating scores
        self.floating_scores = [fs for fs in self.floating_scores if fs['alpha'] > 0]
        for fs in self.floating_scores:
            fs['pos'][1] -= 0.5
            fs['alpha'] -= 5
    
    def _add_floating_score(self, value, color=None):
        if color is None:
            color = self.COLOR_SCORE_PLUS if value > 0 else self.COLOR_SCORE_MINUS
        self.floating_scores.append({
            'val': value,
            'pos': [self.WIDTH/2, self.HEIGHT/2],
            'alpha': 255,
            'color': color
        })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Galactic Bowling")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("R: Reset environment")
        
        while not terminated:
            movement, space, shift = 0, 0, 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            if keys[pygame.K_r]:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0
                continue

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            clock.tick(30) # Run at 30 FPS for smooth manual play
            
        print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        env.close()
    else:
        print("Running in headless mode. Skipping manual play.")
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.close()