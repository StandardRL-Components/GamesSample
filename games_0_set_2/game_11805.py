import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:38:21.040933
# Source Brief: brief_01805.md
# Brief Index: 1805
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import copy

class GameEnv(gym.Env):
    """
    Gymnasium environment for Magnetic Protein Puzzle.

    **Gameplay:**
    The player controls an aiming reticle to shoot proteins into target slots.
    The goal is to assemble a target molecular structure before time runs out.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement): Controls the aiming direction.
        - 0: No-op
        - 1: Aim Up
        - 2: Aim Down
        - 3: Aim Left
        - 4: Aim Right
    - `action[1]` (Spacebar): Fires the selected protein.
        - 0: Released
        - 1: Held - Charges the shot power. Releasing the button fires.
    - `action[2]` (Shift): Cycles through available proteins.
        - 0: Released
        - 1: Pressed - Selects the next available (undocked) protein.

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - **+100.0** for successfully completing the level.
    - **+0.1** per step for each correctly docked protein.
    - **-0.01** per step as a time penalty.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Assemble a target molecular structure by shooting proteins into their corresponding slots before time runs out."
    user_guide = "Use arrow keys to aim. Hold space to charge and release to fire. Press shift to cycle between proteins."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (25, 30, 45)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 180, 0)
    COLOR_TIMER_CRIT = (255, 80, 80)
    COLOR_TARGET_OUTLINE = (255, 215, 0) # Gold
    PROTEIN_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 150, 80),  # Orange
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 28, bold=True)
        self.render_mode = render_mode

        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level = 1
        
        self.proteins = []
        self.target_structure = []
        self.particles = []

        self.active_protein_idx = 0
        self.shot_angle = 0.0
        self.shot_charge = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.validate_implementation() # This is not part of the standard API, can be removed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.level = options['level']
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self._generate_level()
        
        self.active_protein_idx = 0
        # Find the first undocked protein to be active
        for i, p in enumerate(self.proteins):
            if not p['docked']:
                self.active_protein_idx = i
                break

        self.shot_angle = math.pi / 4
        self.shot_charge = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # We need to return the last observation, a reward of 0, and info.
            # The episode is terminated, so terminated=True.
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        self._update_physics()
        
        reward = self._calculate_reward()
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Shift: Cycle Active Protein ---
        if shift_held and not self.prev_shift_held:
            # Find all undocked proteins
            undocked_indices = [i for i, p in enumerate(self.proteins) if not p['docked']]
            if undocked_indices:
                try:
                    current_list_idx = undocked_indices.index(self.active_protein_idx)
                    next_list_idx = (current_list_idx + 1) % len(undocked_indices)
                    self.active_protein_idx = undocked_indices[next_list_idx]
                except ValueError: # If current active protein was just docked
                    self.active_protein_idx = undocked_indices[0]
            # Sound: pygame.mixer.Sound("cycle.wav").play()

        # --- Movement: Aiming ---
        aim_speed = 0.08
        if movement == 1: self.shot_angle -= aim_speed  # Up
        if movement == 2: self.shot_angle += aim_speed  # Down
        if movement == 3: self.shot_angle -= aim_speed  # Left
        if movement == 4: self.shot_angle += aim_speed  # Right

        # --- Space: Charge and Fire ---
        if space_held:
            self.shot_charge = min(1.0, self.shot_charge + 0.05)
        
        if self.prev_space_held and not space_held:
            # Fire on release
            active_protein = self.proteins[self.active_protein_idx]
            if not active_protein['docked']:
                power = 1 + self.shot_charge * 14
                active_protein['vel'] = pygame.Vector2(
                    math.cos(self.shot_angle) * power,
                    math.sin(self.shot_angle) * power
                )
                self._create_particles(active_protein['pos'], active_protein['color'], 20, power)
                # Sound: pygame.mixer.Sound("fire.wav").play()
            self.shot_charge = 0.0

    def _update_physics(self):
        # --- Update Protein Positions & Handle Collisions ---
        dt = 1.0  # Discrete time step
        for i, p in enumerate(self.proteins):
            if p['docked']:
                p['vel'] = pygame.Vector2(0, 0)
                continue

            # Apply drag
            p['vel'] *= 0.98

            # Update position
            p['pos'] += p['vel'] * dt
            
            # Wall collisions
            if p['pos'].x - p['radius'] < 0 or p['pos'].x + p['radius'] > self.WIDTH:
                p['vel'].x *= -0.9
                p['pos'].x = np.clip(p['pos'].x, p['radius'], self.WIDTH - p['radius'])
            if p['pos'].y - p['radius'] < 0 or p['pos'].y + p['radius'] > self.HEIGHT:
                p['vel'].y *= -0.9
                p['pos'].y = np.clip(p['pos'].y, p['radius'], self.HEIGHT - p['radius'])

            # Protein-protein collisions
            for j in range(i + 1, len(self.proteins)):
                p2 = self.proteins[j]
                dist_vec = p['pos'] - p2['pos']
                dist_sq = dist_vec.length_squared()
                min_dist = p['radius'] + p2['radius']
                
                if dist_sq < min_dist ** 2 and dist_sq > 0:
                    dist = math.sqrt(dist_sq)
                    overlap = (min_dist - dist) / 2.0
                    
                    # Resolve overlap
                    p['pos'] += (dist_vec / dist) * overlap
                    p2['pos'] -= (dist_vec / dist) * overlap
                    
                    # Elastic collision response
                    normal = dist_vec.normalize()
                    tangent = pygame.Vector2(-normal.y, normal.x)
                    
                    dp_tan1 = p['vel'].dot(tangent)
                    dp_tan2 = p2['vel'].dot(tangent)
                    
                    dp_norm1 = p['vel'].dot(normal)
                    dp_norm2 = p2['vel'].dot(normal)
                    
                    m1 = (dp_norm1 * (p['mass'] - p2['mass']) + 2 * p2['mass'] * dp_norm2) / (p['mass'] + p2['mass'])
                    m2 = (dp_norm2 * (p2['mass'] - p['mass']) + 2 * p['mass'] * dp_norm1) / (p['mass'] + p2['mass'])
                    
                    p['vel'] = tangent * dp_tan1 + normal * m1
                    p2['vel'] = tangent * dp_tan2 + normal * m2
                    self._create_particles((p['pos'] + p2['pos']) / 2, (200, 200, 200), 5, 2)


            # Docking check
            target = self.target_structure[i]
            if p['color_idx'] == target['color_idx']:
                dist_to_target = p['pos'].distance_to(target['pos'])
                if dist_to_target < p['radius'] * 1.5 and p['vel'].length() < 0.5:
                    p['docked'] = True
                    p['pos'] = pygame.Vector2(target['pos'])
                    self._create_particles(p['pos'], self.COLOR_TARGET_OUTLINE, 30, 4)
                    # Sound: pygame.mixer.Sound("dock.wav").play()

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _calculate_reward(self):
        reward = -0.01  # Time penalty
        
        num_docked = sum(1 for p in self.proteins if p['docked'])
        reward += num_docked * 0.1

        if self._check_win_condition():
            reward += 100.0
            
        return reward

    def _check_termination(self):
        if self._check_win_condition():
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _check_win_condition(self):
        return all(p['docked'] for p in self.proteins)

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "level": self.level,
            "proteins_docked": sum(1 for p in self.proteins if p['docked']),
            "total_proteins": len(self.proteins)
        }
        
    def _generate_level(self):
        self.proteins.clear()
        self.target_structure.clear()
        
        num_proteins = min(8, 3 + (self.level - 1) // 2)
        
        # Generate target positions in a pleasing arrangement
        center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        target_radius = min(self.WIDTH, self.HEIGHT) / 4
        
        for i in range(num_proteins):
            angle = 2 * math.pi * i / num_proteins
            pos = center + pygame.Vector2(math.cos(angle), math.sin(angle)) * target_radius
            color_idx = i % len(self.PROTEIN_COLORS)
            self.target_structure.append({
                'pos': pos,
                'color_idx': color_idx,
                'radius': 15
            })

        # Generate protein starting positions randomly
        for i in range(num_proteins):
            while True:
                start_pos = pygame.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                # Ensure it doesn't start too close to a target
                if all(start_pos.distance_to(t['pos']) > 100 for t in self.target_structure):
                    break

            color_idx = self.target_structure[i]['color_idx']
            self.proteins.append({
                'pos': start_pos,
                'vel': pygame.Vector2(0, 0),
                'radius': 15,
                'mass': 1,
                'color_idx': color_idx,
                'color': self.PROTEIN_COLORS[color_idx],
                'docked': False
            })

    def _render_all(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        # Render all game elements
        self._render_game_elements()
        
        # Render UI overlay
        self._render_ui()

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # --- Draw Target Structure ---
        for target in self.target_structure:
            color = self.PROTEIN_COLORS[target['color_idx']]
            pygame.gfxdraw.aacircle(
                self.screen, int(target['pos'].x), int(target['pos'].y), 
                target['radius'] + 2, self.COLOR_TARGET_OUTLINE
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(target['pos'].x), int(target['pos'].y), 
                target['radius'], (*color, 20)
            )

        # --- Draw Particles ---
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # --- Draw Proteins ---
        for i, p in enumerate(self.proteins):
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            if p['docked']:
                # Draw solid, bright docked protein
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'], p['color'])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p['radius'], self.COLOR_TARGET_OUTLINE)
            else:
                # Draw regular protein with a darker fill
                fill_color = tuple(c // 2 for c in p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'], fill_color)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p['radius'], p['color'])
        
        # --- Draw Aiming UI for Active Protein ---
        if not self.game_over and self.active_protein_idx < len(self.proteins):
            active_p = self.proteins[self.active_protein_idx]
            if not active_p['docked']:
                pos = active_p['pos']
                
                # Highlight active protein
                glow_radius = int(active_p['radius'] + 4 + 4 * abs(math.sin(self.steps * 0.1)))
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), glow_radius, (*active_p['color'], 50))
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), glow_radius-1, (*active_p['color'], 50))

                # Aiming line
                aim_len = 30 + 100 * self.shot_charge
                end_pos = pos + pygame.Vector2(math.cos(self.shot_angle), math.sin(self.shot_angle)) * aim_len
                pygame.draw.aaline(self.screen, (255, 255, 255, 150), pos, end_pos, 1)

                # Charge indicator
                if self.shot_charge > 0:
                    charge_color = (255, 255, 255)
                    if self.shot_charge >= 1.0:
                        charge_color = self.COLOR_TARGET_OUTLINE
                    for i in range(int(self.shot_charge * 10)):
                        angle = self.shot_angle + math.pi + (i - 4.5) * 0.1
                        offset = pygame.Vector2(math.cos(angle), math.sin(angle)) * (active_p['radius'] + 10)
                        pygame.draw.circle(self.screen, charge_color, pos + offset, 2)


    def _render_ui(self):
        # --- Score ---
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # --- Time ---
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_color = self.COLOR_UI_TEXT
        if time_left < 10: time_color = self.COLOR_TIMER_CRIT
        elif time_left < 30: time_color = self.COLOR_TIMER_WARN
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # --- Level ---
        level_text = self.font_level.render(f"LEVEL {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH / 2 - level_text.get_width() / 2, 10))
        
        # --- Docking Progress ---
        docked = sum(1 for p in self.proteins if p['docked'])
        total = len(self.proteins)
        progress_text = self.font_ui.render(f"DOCKED: {docked}/{total}", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (10, self.HEIGHT - progress_text.get_height() - 10))

    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale * 0.5
            life = self.np_random.integers(10, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Magnetic Protein Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()