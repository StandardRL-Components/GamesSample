import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:35:22.870302
# Source Brief: brief_02259.md
# Brief Index: 2259
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating a clockwork solar system.
    The agent's goal is to place gears to connect terraformed planets
    to a central core, generating energy. This energy is then used to
    terraform more planets.

    The game is won by terraforming all planets.
    The game is lost if energy runs out.
    """
    game_description = (
        "Connect planets to a central power core using gears to generate energy. "
        "Use the energy to terraform all planets and win the game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to place a gear or terraform a planet. "
        "Press 'shift' to switch between placement modes."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    MAX_ENERGY = 1000
    INITIAL_ENERGY = 250
    TERRAFORM_COST = 200
    ENERGY_PER_TICK_PER_PLANET = 2
    CURSOR_SPEED = 8
    GEAR_RADIUS = 15
    PLANET_RADIUS = 20
    CORE_RADIUS = 25

    # --- Colors (Steampunk/Sci-Fi Palette) ---
    COLOR_BG = (15, 10, 25)
    COLOR_STAR = (200, 200, 220)
    COLOR_CORE = (255, 200, 0)
    COLOR_GEAR = (212, 175, 55)
    COLOR_GEAR_TEETH = (160, 130, 40)
    COLOR_PLANET_BARREN = (140, 130, 120)
    COLOR_PLANET_TERRAFORMED = (60, 180, 75)
    COLOR_ENERGY_FLOW = (0, 220, 255)
    COLOR_CURSOR_GEAR = (212, 175, 55, 150)
    COLOR_CURSOR_TERRAFORM = (60, 180, 75, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_ENERGY_BAR = (0, 220, 255)
    COLOR_ENERGY_BAR_LOW = (255, 70, 70)
    COLOR_ENERGY_BAR_BG = (50, 50, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_mode = pygame.font.SysFont("Consolas", 22, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0
        self.cursor_pos = [0, 0]
        self.placement_mode = 'TERRAFORM'
        self.last_space_held = False
        self.last_shift_held = False
        self.core = {}
        self.planets = []
        self.gears = []
        self._starfield_surface = None

        # self.reset() is not called here to avoid duplicate initialization
        # self.validate_implementation() is also removed from init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = self.INITIAL_ENERGY
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.placement_mode = 'TERRAFORM'
        self.last_space_held = False
        self.last_shift_held = False

        self.core = {
            'pos': np.array([self.WIDTH / 2, self.HEIGHT / 2]),
            'radius': self.CORE_RADIUS,
            'type': 'core',
            'connected': True,
            'rotation': 0
        }
        self.gears = []
        self._initialize_planets()

        if self._starfield_surface is None:
            self._starfield_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
            self._draw_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        action_reward = self._handle_input(action)
        reward += action_reward

        energy_reward = self._update_connections_and_energy()
        reward += energy_reward

        self._update_animations()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.energy <= 0:
                reward -= 100  # Failure penalty
            elif all(p['terraformed'] for p in self.planets):
                reward += 100  # Victory bonus
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = 0

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # --- Switch Mode (on press) ---
        if shift_held and not self.last_shift_held:
            # sfx: UI_switch.wav
            self.placement_mode = 'GEAR' if self.placement_mode == 'TERRAFORM' else 'TERRAFORM'

        # --- Action (on press) ---
        if space_held and not self.last_space_held:
            if self.placement_mode == 'TERRAFORM':
                target_planet = self._get_closest_entity(self.cursor_pos, self.planets)
                if target_planet and np.linalg.norm(np.array(self.cursor_pos) - target_planet['pos']) < target_planet['radius']:
                    if not target_planet['terraformed'] and self.energy >= self.TERRAFORM_COST:
                        # sfx: terraform_success.wav
                        self.energy -= self.TERRAFORM_COST
                        target_planet['terraformed'] = True
                        action_reward += 1.0
                        self.score += 1
                    else:
                        # sfx: action_fail.wav
                        pass
            elif self.placement_mode == 'GEAR':
                # sfx: place_gear.wav
                self.gears.append({
                    'pos': np.array(self.cursor_pos),
                    'radius': self.GEAR_RADIUS,
                    'type': 'gear',
                    'connected': False,
                    'rotation': self.np_random.uniform(0, 360)
                })

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return action_reward

    def _update_connections_and_energy(self):
        all_objects = [self.core] + self.planets + self.gears
        for obj in all_objects:
            obj['connected'] = False

        queue = deque([self.core])
        self.core['connected'] = True
        
        visited = {id(self.core)}

        while queue:
            current_obj = queue.popleft()
            for other_obj in all_objects:
                if id(other_obj) not in visited:
                    dist = np.linalg.norm(current_obj['pos'] - other_obj['pos'])
                    if dist < current_obj['radius'] + other_obj['radius']:
                        other_obj['connected'] = True
                        visited.add(id(other_obj))
                        queue.append(other_obj)

        energy_generated = 0
        for planet in self.planets:
            if planet['terraformed'] and planet['connected']:
                energy_generated += self.ENERGY_PER_TICK_PER_PLANET
        
        self.energy = min(self.MAX_ENERGY, self.energy + energy_generated)
        return energy_generated * 0.1 # Reward for energy generation

    def _update_animations(self):
        self.core['rotation'] = (self.core['rotation'] + 0.5) % 360
        rotation_speed = 2
        for obj in self.planets + self.gears:
            if obj['connected']:
                obj['rotation'] = (obj['rotation'] - rotation_speed) % 360
            else:
                 obj['rotation'] = (obj['rotation'] + rotation_speed * 0.2) % 360

    def _check_termination(self):
        if self.energy <= 0:
            # sfx: game_over_lose.wav
            return True
        if all(p['terraformed'] for p in self.planets):
            # sfx: game_over_win.wav
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self._starfield_surface:
            self.screen.blit(self._starfield_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Connections ---
        all_movable = self.planets + self.gears
        for i in range(len(all_movable)):
            for j in range(i + 1, len(all_movable)):
                obj1 = all_movable[i]
                obj2 = all_movable[j]
                if obj1['connected'] and obj2['connected']:
                    dist = np.linalg.norm(obj1['pos'] - obj2['pos'])
                    if dist < obj1['radius'] + obj2['radius']:
                        self._draw_energy_line(obj1['pos'], obj2['pos'])
        for obj in all_movable:
            if obj['connected']:
                dist = np.linalg.norm(obj['pos'] - self.core['pos'])
                if dist < obj['radius'] + self.core['radius']:
                    self._draw_energy_line(obj['pos'], self.core['pos'])

        # --- Render Objects ---
        self._draw_entity(self.core)
        for planet in self.planets:
            self._draw_entity(planet)
        for gear in self.gears:
            self._draw_entity(gear)

        # --- Render Cursor ---
        self._draw_cursor()

    def _render_ui(self):
        # --- Energy Bar ---
        bar_width = 300
        bar_height = 20
        energy_ratio = self.energy / self.MAX_ENERGY
        current_bar_width = int(bar_width * energy_ratio)
        bar_color = self.COLOR_ENERGY_BAR if energy_ratio > 0.2 else self.COLOR_ENERGY_BAR_LOW
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (10, 10, bar_width, bar_height))
        if current_bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 10, current_bar_width, bar_height))
        
        energy_text = self.font_ui.render(f"ENERGY: {int(self.energy)}/{self.MAX_ENERGY}", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (15, 12))

        # --- Score ---
        score_text = self.font_ui.render(f"PLANETS TERRAFORMED: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # --- Mode Indicator ---
        mode_str = "MODE: TERRAFORM PLANET" if self.placement_mode == 'TERRAFORM' else "MODE: PLACE GEAR"
        mode_color = self.COLOR_CURSOR_TERRAFORM[:3] if self.placement_mode == 'TERRAFORM' else self.COLOR_GEAR
        mode_text = self.font_mode.render(mode_str, True, mode_color)
        self.screen.blit(mode_text, (self.WIDTH // 2 - mode_text.get_width() // 2, self.HEIGHT - 40))

    def _draw_entity(self, entity):
        pos = (int(entity['pos'][0]), int(entity['pos'][1]))
        radius = int(entity['radius'])
        
        if entity['type'] == 'core':
            pulse_radius = radius + 5 * (1 + math.sin(pygame.time.get_ticks() * 0.002))
            self._draw_glow(pos, pulse_radius, self.COLOR_CORE)
            self._draw_gear(pos, radius, self.COLOR_CORE, self.COLOR_GEAR_TEETH, entity['rotation'], 12)
        elif entity['type'] == 'planet':
            color = self.COLOR_PLANET_TERRAFORMED if entity['terraformed'] else self.COLOR_PLANET_BARREN
            if entity['connected']:
                self._draw_glow(pos, radius * 1.3, self.COLOR_ENERGY_FLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, tuple(c*0.8 for c in color))
        elif entity['type'] == 'gear':
            if entity['connected']:
                 self._draw_glow(pos, radius * 1.3, self.COLOR_ENERGY_FLOW)
            self._draw_gear(pos, radius, self.COLOR_GEAR, self.COLOR_GEAR_TEETH, entity['rotation'], 8)

    def _draw_gear(self, pos, radius, color, tooth_color, rotation, num_teeth):
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 0.8), color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius * 0.8), color)
        
        tooth_width = 8
        tooth_height = 5
        angle_step = 360 / num_teeth
        for i in range(num_teeth):
            angle = math.radians(i * angle_step + rotation)
            center_x = pos[0] + (radius - tooth_height/2) * math.cos(angle)
            center_y = pos[1] + (radius - tooth_height/2) * math.sin(angle)
            
            points = [
                (-tooth_width / 2, -tooth_height / 2),
                (tooth_width / 2, -tooth_height / 2),
                (tooth_width / 2, tooth_height / 2),
                (-tooth_width / 2, tooth_height / 2)
            ]
            
            rotated_points = []
            for x, y in points:
                new_x = x * math.cos(angle) - y * math.sin(angle) + center_x
                new_y = x * math.sin(angle) + y * math.cos(angle) + center_y
                rotated_points.append((int(new_x), int(new_y)))

            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, tooth_color)
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, tooth_color)

    def _draw_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        if self.placement_mode == 'GEAR':
            surface = pygame.Surface((self.GEAR_RADIUS*2, self.GEAR_RADIUS*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(surface, self.GEAR_RADIUS, self.GEAR_RADIUS, self.GEAR_RADIUS, self.COLOR_CURSOR_GEAR)
            pygame.gfxdraw.aacircle(surface, self.GEAR_RADIUS, self.GEAR_RADIUS, self.GEAR_RADIUS, self.COLOR_CURSOR_GEAR)
            self.screen.blit(surface, (pos[0] - self.GEAR_RADIUS, pos[1] - self.GEAR_RADIUS))
        else: # TERRAFORM
            target_planet = self._get_closest_entity(self.cursor_pos, self.planets)
            if target_planet and np.linalg.norm(np.array(self.cursor_pos) - target_planet['pos']) < target_planet['radius'] * 2:
                target_pos = (int(target_planet['pos'][0]), int(target_planet['pos'][1]))
                radius = int(target_planet['radius'] * 1.2)
                color = self.COLOR_CURSOR_TERRAFORM
                pygame.draw.line(self.screen, color, (target_pos[0] - radius, target_pos[1]), (target_pos[0] - radius//2, target_pos[1]), 2)
                pygame.draw.line(self.screen, color, (target_pos[0] + radius, target_pos[1]), (target_pos[0] + radius//2, target_pos[1]), 2)
                pygame.draw.line(self.screen, color, (target_pos[0], target_pos[1] - radius), (target_pos[0], target_pos[1] - radius//2), 2)
                pygame.draw.line(self.screen, color, (target_pos[0], target_pos[1] + radius), (target_pos[0], target_pos[1] + radius//2), 2)

    def _draw_starfield(self):
        self._starfield_surface.fill(self.COLOR_BG)
        for _ in range(200):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.HEIGHT)
            size = self.np_random.choice([1, 2])
            alpha = self.np_random.integers(50, 150)
            color = self.COLOR_STAR + (alpha,)
            if size == 1:
                self._starfield_surface.set_at((x, y), color)
            else:
                pygame.draw.circle(self._starfield_surface, color, (x, y), 1)

    def _draw_energy_line(self, pos1, pos2):
        p1 = (int(pos1[0]), int(pos1[1]))
        p2 = (int(pos2[0]), int(pos2[1]))
        # sfx: energy_hum_loop.wav
        pygame.draw.aaline(self.screen, self.COLOR_ENERGY_FLOW, p1, p2, 4)
        pygame.draw.aaline(self.screen, (200, 255, 255), p1, p2, 1)
        
    def _draw_glow(self, pos, radius, color):
        surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        glow_color = color + (30,)
        pygame.gfxdraw.filled_circle(surface, int(radius), int(radius), int(radius), glow_color)
        self.screen.blit(surface, (pos[0]-radius, pos[1]-radius))

    def _initialize_planets(self):
        self.planets = []
        num_planets = 5
        angle_step = 360 / num_planets
        for i in range(num_planets):
            angle = math.radians(i * angle_step + 15)
            distance = self.WIDTH / 3.5 + (i % 2) * 40
            x = self.WIDTH / 2 + distance * math.cos(angle)
            y = self.HEIGHT / 2 + distance * math.sin(angle)
            self.planets.append({
                'pos': np.array([x, y]),
                'radius': self.PLANET_RADIUS,
                'type': 'planet',
                'terraformed': False,
                'connected': False,
                'rotation': self.np_random.uniform(0, 360)
            })

    def _get_closest_entity(self, pos, entity_list):
        if not entity_list:
            return None
        closest_ent = min(entity_list, key=lambda e: np.linalg.norm(np.array(pos) - e['pos']))
        return closest_ent

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "mode": self.placement_mode,
            "planets_terraformed": sum(1 for p in self.planets if p['terraformed']),
            "gears_placed": len(self.gears)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The original code set the video driver to dummy, which prevents display.
    # For manual play, we need to unset it and initialize a real display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz" depending on OS
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Clockwork Solar System")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()