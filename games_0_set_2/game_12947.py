import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:41:45.257747
# Source Brief: brief_02947.md
# Brief Index: 2947
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must maintain equilibrium across
    three interconnected water tanks by controlling valves.
    """
    game_description = (
        "Control the valves between three interconnected water tanks to maintain their levels "
        "within a target zone and achieve equilibrium."
    )
    user_guide = (
        "Use ↑↓ arrow keys to control the first valve, ←→ for the second, and Space/Shift "
        "for the outflow valve. Balance the water levels in all three tanks."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("Consolas", 16)
        self.font_md = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_lg = pygame.font.SysFont("Consolas", 28, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 1200
        self.VICTORY_STEPS = 150
        self.VALVE_ADJUST_RATE = 4.0
        self.DT = 1.0 / 30.0 # Simulation delta time

        # Physics constants - tuned for challenging but manageable gameplay
        self.INFLOW_RATE = 0.6
        self.FLOW_CONSTANT = 0.012
        self.OUTFLOW_CONSTANT = 0.01

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_PIPE = (80, 90, 100)
        self.COLOR_TANK = (120, 130, 140)
        self.COLOR_WATER = (50, 150, 255)
        self.COLOR_WATER_SURFACE = (150, 200, 255)
        self.COLOR_TARGET_ZONE = (0, 255, 120, 50) # RGBA
        self.COLOR_OVERFLOW = (255, 80, 80)
        self.COLOR_TEXT = (230, 240, 250)
        self.COLOR_VALVE_ACTIVE = (255, 200, 0)

        # --- Tank Configuration ---
        self.tank_defs = [
            {"pos": (100, 100), "size": (100, 250), "target": 60.0},
            {"pos": (270, 100), "size": (100, 250), "target": 50.0},
            {"pos": (440, 100), "size": (100, 250), "target": 40.0},
        ]
        self.TARGET_RANGE = 10.0 # +/- this percentage from target

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tank_levels = np.zeros(3, dtype=float)
        self.valve_openings = np.zeros(3, dtype=float) # [v1-2, v2-3, v3-out]
        self.victory_counter = 0
        self.particles = []
        self.last_action = [0, 0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory_counter = 0
        self.particles = []
        self.last_action = [0, 0, 0]

        # Initial state to provide a starting challenge
        self.tank_levels = np.array([20.0, 5.0, 10.0])
        self.valve_openings = np.array([50.0, 50.0, 25.0])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.last_action = action
        self._handle_actions(action)
        
        reward = self._update_physics()
        self.score += reward
        
        self._update_game_logic()
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Timeout
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Valve 1 (Tanks 1-2) controlled by Up/Down
        if movement == 1: # Up
            self.valve_openings[0] += self.VALVE_ADJUST_RATE
        elif movement == 2: # Down
            self.valve_openings[0] -= self.VALVE_ADJUST_RATE
            
        # Valve 2 (Tanks 2-3) controlled by Left/Right
        if movement == 3: # Left
            self.valve_openings[1] -= self.VALVE_ADJUST_RATE
        elif movement == 4: # Right
            self.valve_openings[1] += self.VALVE_ADJUST_RATE

        # Valve 3 (Outflow) controlled by Space/Shift
        if space_held:
            self.valve_openings[2] += self.VALVE_ADJUST_RATE
        if shift_held:
            self.valve_openings[2] -= self.VALVE_ADJUST_RATE
            
        self.valve_openings = np.clip(self.valve_openings, 0, 100)

    def _update_physics(self):
        reward = 0
        
        # Calculate flows based on Torricelli's law principle (proportional to level difference)
        v_norm = self.valve_openings / 100.0
        
        # Flow between tank 1 and 2
        flow_12 = self.FLOW_CONSTANT * v_norm[0] * (self.tank_levels[0] - self.tank_levels[1])
        
        # Flow between tank 2 and 3
        flow_23 = self.FLOW_CONSTANT * v_norm[1] * (self.tank_levels[1] - self.tank_levels[2])
        
        # Outflow from tank 3
        outflow_3 = self.OUTFLOW_CONSTANT * v_norm[2] * self.tank_levels[2]
        
        # Update levels
        level_deltas = np.array([
            self.INFLOW_RATE - flow_12,
            flow_12 - flow_23,
            flow_23 - outflow_3
        ])
        self.tank_levels += level_deltas * self.DT
        self.tank_levels = np.clip(self.tank_levels, 0, 105) # Allow slight overflow for detection

        # Check for overflow
        for i in range(3):
            if self.tank_levels[i] > 100:
                self._create_overflow_particles(i)
                self.tank_levels[i] = 0 # Harsh penalty as per brief
                self.game_over = True
                return -10.0 # Event-based reward
        
        # Continuous reward
        in_zone = self._are_tanks_in_target_zone(margin=20.0)
        reward += 1.0 if in_zone else -0.1
        return reward

    def _update_game_logic(self):
        # Update victory counter
        if self._are_tanks_in_target_zone(margin=self.TARGET_RANGE):
            self.victory_counter += 1
        else:
            self.victory_counter = 0

        # Check for victory
        if self.victory_counter >= self.VICTORY_STEPS:
            self.score += 100 # Goal-oriented reward
            self.game_over = True
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0] * self.DT
            p['pos'][1] += p['vel'][1] * self.DT
            p['vel'][1] += 150 * self.DT # Gravity
            p['life'] -= 1

    def _are_tanks_in_target_zone(self, margin):
        for i in range(3):
            target = self.tank_defs[i]['target']
            if not (target - margin <= self.tank_levels[i] <= target + margin):
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_pipes()
        self._render_tanks()
        self._render_valves()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tank_levels": self.tank_levels,
            "valve_openings": self.valve_openings,
            "victory_progress": self.victory_counter / self.VICTORY_STEPS
        }

    # --- Rendering Methods ---

    def _render_pipes(self):
        # Pipe 1-2
        y_pipe = self.tank_defs[0]['pos'][1] + self.tank_defs[0]['size'][1] - 30
        x1 = self.tank_defs[0]['pos'][0] + self.tank_defs[0]['size'][0]
        x2 = self.tank_defs[1]['pos'][0]
        pygame.draw.line(self.screen, self.COLOR_PIPE, (x1, y_pipe), (x2, y_pipe), 8)

        # Pipe 2-3
        x1 = self.tank_defs[1]['pos'][0] + self.tank_defs[1]['size'][0]
        x2 = self.tank_defs[2]['pos'][0]
        pygame.draw.line(self.screen, self.COLOR_PIPE, (x1, y_pipe), (x2, y_pipe), 8)

        # Outflow pipe
        x_out = self.tank_defs[2]['pos'][0] + self.tank_defs[2]['size'][0]
        pygame.draw.line(self.screen, self.COLOR_PIPE, (x_out, y_pipe), (x_out + 40, y_pipe), 8)

        # Inflow pipe
        x_in = self.tank_defs[0]['pos'][0]
        pygame.draw.line(self.screen, self.COLOR_PIPE, (x_in - 40, 60), (x_in, 60), 8)
        # inflow animation
        if self.steps % 10 < 5:
             pygame.draw.circle(self.screen, self.COLOR_WATER, (x_in - 20, 60), 5)


    def _render_tanks(self):
        for i, tank in enumerate(self.tank_defs):
            x, y = tank['pos']
            w, h = tank['size']
            level_px = h * (self.tank_levels[i] / 100.0)
            
            # Target zone
            target_h = h * (self.TARGET_RANGE * 2 / 100.0)
            target_y = y + h - h * ((tank['target'] + self.TARGET_RANGE) / 100.0)
            target_rect = pygame.Rect(x, target_y, w, target_h)
            
            s = pygame.Surface((w, target_h), pygame.SRCALPHA)
            s.fill(self.COLOR_TARGET_ZONE)
            self.screen.blit(s, (x, target_y))

            # Water
            if level_px > 0:
                water_rect = pygame.Rect(x, y + h - level_px, w, level_px)
                pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect)
                # Water surface highlight
                pygame.draw.line(self.screen, self.COLOR_WATER_SURFACE, 
                                 (x, y + h - level_px), (x + w, y + h - level_px), 2)

            # Tank outline
            pygame.draw.rect(self.screen, self.COLOR_TANK, (x, y, w, h), 3)

    def _render_valves(self):
        # Valve 1-2
        y_pipe = self.tank_defs[0]['pos'][1] + self.tank_defs[0]['size'][1] - 30
        x1 = self.tank_defs[0]['pos'][0] + self.tank_defs[0]['size'][0]
        x2 = self.tank_defs[1]['pos'][0]
        valve_pos = ((x1 + x2) // 2, y_pipe)
        is_active = self.last_action[0] in [1, 2]
        self._draw_valve(valve_pos, self.valve_openings[0], is_active)

        # Valve 2-3
        x1 = self.tank_defs[1]['pos'][0] + self.tank_defs[1]['size'][0]
        x2 = self.tank_defs[2]['pos'][0]
        valve_pos = ((x1 + x2) // 2, y_pipe)
        is_active = self.last_action[0] in [3, 4]
        self._draw_valve(valve_pos, self.valve_openings[1], is_active)
        
        # Valve 3-out
        x_out = self.tank_defs[2]['pos'][0] + self.tank_defs[2]['size'][0] + 20
        valve_pos = (x_out, y_pipe)
        is_active = self.last_action[1] == 1 or self.last_action[2] == 1
        self._draw_valve(valve_pos, self.valve_openings[2], is_active)

    def _draw_valve(self, pos, opening, is_active):
        color = self.COLOR_VALVE_ACTIVE if is_active else self.COLOR_TANK
        radius = 12
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color)

        angle = (opening / 100.0) * math.pi * 1.8 - math.pi * 0.9
        x_end = pos[0] + (radius - 2) * math.cos(angle)
        y_end = pos[1] + (radius - 2) * math.sin(angle)
        pygame.draw.line(self.screen, color, pos, (x_end, y_end), 3)

    def _create_overflow_particles(self, tank_index):
        # sound placeholder: # sfx_overflow_splash()
        tank = self.tank_defs[tank_index]
        x_start = tank['pos'][0] + tank['size'][0] / 2
        y_start = tank['pos'][1]
        for _ in range(30):
            self.particles.append({
                'pos': [x_start + random.uniform(-20, 20), y_start],
                'vel': [random.uniform(-50, 50), random.uniform(-100, -20)],
                'life': random.randint(20, 40),
                'color': self.COLOR_OVERFLOW
            })

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = (*p['color'], alpha)
            size = max(1, int(4 * (p['life'] / 40.0)))
            
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (p['pos'][0]-size, p['pos'][1]-size))

    def _render_ui(self):
        # Tank level text
        for i, tank in enumerate(self.tank_defs):
            text = f"{self.tank_levels[i]:.1f}%"
            txt_surf = self.font_md.render(text, True, self.COLOR_TEXT)
            x = tank['pos'][0] + tank['size'][0] / 2 - txt_surf.get_width() / 2
            y = tank['pos'][1] - 30
            self.screen.blit(txt_surf, (x, y))

        # Score and Steps
        score_text = f"Score: {self.score:.1f}"
        steps_text = f"Step: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(score_text, (10, 10), self.font_sm)
        self._draw_text(steps_text, (10, 30), self.font_sm)
        
        # Victory progress bar
        if self.victory_counter > 0:
            progress = self.victory_counter / self.VICTORY_STEPS
            bar_w = self.WIDTH - 20
            bar_h = 15
            fill_w = bar_w * progress
            pygame.draw.rect(self.screen, self.COLOR_PIPE, (10, self.HEIGHT - 25, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_TARGET_ZONE[:3], (10, self.HEIGHT - 25, fill_w, bar_h))
            self._draw_text("SYSTEM STABLE", (self.WIDTH / 2, self.HEIGHT - 26), self.font_sm, center=True)

        # Game Over/Victory Text
        if self.game_over:
            if self.victory_counter >= self.VICTORY_STEPS:
                msg = "EQUILIBRIUM ACHIEVED"
                color = self.COLOR_TARGET_ZONE[:3]
            else:
                msg = "SYSTEM FAILURE"
                color = self.COLOR_OVERFLOW
            
            txt_surf = self.font_lg.render(msg, True, color)
            pos = (self.WIDTH / 2 - txt_surf.get_width() / 2, self.HEIGHT / 2 - txt_surf.get_height() / 2)
            self.screen.blit(txt_surf, pos)


    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        txt_surf = font.render(text, True, color)
        if center:
            pos = (pos[0] - txt_surf.get_width() / 2, pos[1])
        self.screen.blit(txt_surf, pos)
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    # The validate_implementation method is removed from the main execution block
    # as it's primarily for development and testing.
    # It is called once at the end of __init__ to ensure correctness.
    
    # We will re-initialize pygame with a visible display for human play
    pygame.quit() # Close the dummy display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.init()
    pygame.font.init()

    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Override Pygame screen for direct rendering
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Water Tank Equilibrium")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if terminated:
            # Game over, wait for reset
            pass
        else:
            # --- Player Controls ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Render ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth human playback
        
    env.close()