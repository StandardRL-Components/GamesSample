import gymnasium as gym
import os
import pygame
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls a phagocyte to contain infected cells.

    **Objective:** Neutralize all infected cells by trapping them in containment fields
    before they reach the vital organ.

    **Actions:**
    - Movement: Move the phagocyte (up, down, left, right).
    - Deploy Field: Use a limited resource to create a circular field that traps cells.
    - Speed Burst: A temporary speed boost with a cooldown.

    **Visuals:**
    - A microscopic, bio-organic theme with clean, high-contrast elements.
    - Smooth animations, particle effects, and glowing objects create a polished feel.

    **Rewards:**
    - Rewards are structured to encourage proactive containment, resource management,
      and protecting the vital organ.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a phagocyte to contain fast-moving infected cells. "
        "Deploy containment fields to neutralize them before they reach the vital organ."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move. Press space to deploy a containment field "
        "and shift for a temporary speed burst."
    )
    auto_advance = True

    # --- Colors and Style ---
    COLOR_BG = (16, 16, 32)
    COLOR_GRID = (24, 24, 48)
    COLOR_PLAYER = (0, 255, 127)
    COLOR_INFECTED = (255, 64, 64)
    COLOR_CONTAINMENT = (0, 191, 255)
    COLOR_ORGAN = (138, 43, 226)
    COLOR_PARTICLE = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BAR_BG = (40, 40, 60)
    COLOR_BAR_FILL = (220, 30, 30)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = 640
        self.screen_height = 400

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 18, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = None
        self.player_radius = 15
        self.player_speed = 4.0
        self.speed_burst_timer = 0
        self.speed_burst_cooldown = 0

        # Infected cells state
        self.infected_cells = []
        self.infected_cell_radius = 8
        self.infected_cell_base_speed = 1.0
        self.infected_cell_speed_modifier = 1.0
        self.initial_infected_count = 3

        # Containment fields state
        self.containment_fields = []
        self.field_max_radius = 60
        self.field_lifetime = 150 # 5 seconds at 30fps
        self.max_fields = 5
        self.fields_available = self.max_fields
        
        # Vital organ state
        self.vital_organ_pos = np.array([self.screen_width - 60, self.screen_height / 2], dtype=np.float32)
        self.vital_organ_radius = 40
        
        # Action handling state
        self.space_was_held = False
        self.shift_was_held = False

        # Visual effects
        self.particles = []

        # This will be initialized in reset()
        self.np_random = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([100.0, self.screen_height / 2.0], dtype=np.float32)
        self.speed_burst_timer = 0
        self.speed_burst_cooldown = 0
        
        self.infected_cell_speed_modifier = 1.0
        self.infected_cells = []
        self._spawn_infected_cells(self.initial_infected_count)

        self.containment_fields = []
        self.fields_available = self.max_fields
        
        self.particles = []
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = self._handle_logic(movement, space_held, shift_held)
        
        terminated = self.game_over or self.steps >= 2000
        
        if terminated and not self.game_over: # Win condition or timeout
            # Check if all cells are neutralized
            if not self.infected_cells:
                reward += 100
                self.score += 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_logic(self, movement, space_held, shift_held):
        """Processes actions, updates game state, and calculates rewards."""
        # --- Handle Actions ---
        reward = 0
        
        # Speed Burst (Shift)
        shift_pressed = shift_held and not self.shift_was_held
        if shift_pressed and self.speed_burst_cooldown == 0:
            self.speed_burst_timer = 15 # 0.5s at 30fps
            self.speed_burst_cooldown = 120 # 4s cooldown

        # Movement
        current_speed = self.player_speed * 2 if self.speed_burst_timer > 0 else self.player_speed
        if movement == 1: self.player_pos[1] -= current_speed
        elif movement == 2: self.player_pos[1] += current_speed
        elif movement == 3: self.player_pos[0] -= current_speed
        elif movement == 4: self.player_pos[0] += current_speed
        
        # Player boundary constraints
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.screen_width - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.screen_height - self.player_radius)

        if self.speed_burst_timer > 0 and movement != 0:
            self._create_particles(self.player_pos, 2)

        # Deploy Containment Field (Space)
        space_pressed = space_held and not self.space_was_held
        if space_pressed and self.fields_available > 0:
            self.fields_available -= 1
            self.containment_fields.append({
                'pos': self.player_pos.copy(),
                'radius': 0,
                'lifetime': self.field_lifetime,
                'is_active': True
            })
            reward -= 1
        
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        # --- Update Game State ---
        # Timers
        if self.speed_burst_timer > 0: self.speed_burst_timer -= 1
        if self.speed_burst_cooldown > 0: self.speed_burst_cooldown -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.infected_cell_speed_modifier += 0.05
        if self.steps > 0 and self.steps % 500 == 0:
            self._spawn_infected_cells(1)
        
        # Update infected cells
        cells_to_remove = []
        for cell in self.infected_cells:
            prev_dist = np.linalg.norm(cell['pos'] - self.vital_organ_pos)
            
            is_contained = False
            for field in self.containment_fields:
                if field['is_active'] and np.linalg.norm(cell['pos'] - field['pos']) < field['radius']:
                    is_contained = True
                    break
            
            if not is_contained:
                direction = self.vital_organ_pos - cell['pos']
                dist = np.linalg.norm(direction)
                if dist > 1: # Avoid division by zero
                    direction /= dist
                
                # Add some brownian motion
                noise = self.np_random.normal(0, 0.3, 2)
                cell['vel'] = cell['vel'] * 0.95 + direction * 0.05 + noise
                cell['vel'] /= np.linalg.norm(cell['vel']) # Normalize
                
                cell['pos'] += cell['vel'] * self.infected_cell_base_speed * self.infected_cell_speed_modifier
            
            # Reward calculation for cell state
            if is_contained:
                reward += 0.1 # Continuous reward for being contained
                if not cell['was_contained']:
                    reward += 5 # Event reward for first containment
                    self.score += 5
                    cell['was_contained'] = True
            else:
                new_dist = np.linalg.norm(cell['pos'] - self.vital_organ_pos)
                if new_dist < prev_dist:
                    reward -= 0.01 # Penalty for getting closer to organ

            # Check for organ breach
            if np.linalg.norm(cell['pos'] - self.vital_organ_pos) < self.vital_organ_radius + self.infected_cell_radius:
                self.game_over = True
                reward -= 100
                self.score -= 100

        # Update containment fields
        fields_to_remove = []
        for field in self.containment_fields:
            if field['radius'] < self.field_max_radius:
                field['radius'] += 2
            
            field['lifetime'] -= 1
            if field['lifetime'] <= 0:
                field['is_active'] = False
                fields_to_remove.append(field)
                self.fields_available += 1
                
                # Check for neutralized cells
                for cell in self.infected_cells:
                    if np.linalg.norm(cell['pos'] - field['pos']) < field['radius']:
                        cells_to_remove.append(cell)
        
        self.containment_fields = [f for f in self.containment_fields if f not in fields_to_remove]
        self.infected_cells = [c for c in self.infected_cells if c not in cells_to_remove]

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

        return reward

    def _spawn_infected_cells(self, count):
        for _ in range(count):
            # Spawn on the left third of the screen, away from edges
            pos = np.array([
                self.np_random.uniform(self.infected_cell_radius, self.screen_width / 3),
                self.np_random.uniform(self.infected_cell_radius, self.screen_height - self.infected_cell_radius)
            ], dtype=np.float32)

            # Initial velocity towards the organ
            direction = self.vital_organ_pos - pos
            direction /= np.linalg.norm(direction)
            
            self.infected_cells.append({
                'pos': pos,
                'vel': direction,
                'was_contained': False
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for x in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*self.COLOR_PARTICLE, alpha)
            self._draw_glowing_circle(self.screen, p['pos'], p['radius'], color, 0.5)

        # Vital Organ
        pulse = (math.sin(self.steps * 0.05) + 1) / 2
        organ_color = tuple(int(c * (0.8 + 0.2 * pulse)) for c in self.COLOR_ORGAN)
        self._draw_glowing_circle(self.screen, self.vital_organ_pos, self.vital_organ_radius, organ_color, 0.4)

        # Containment Fields
        for field in self.containment_fields:
            pos = (int(field['pos'][0]), int(field['pos'][1]))
            radius = int(field['radius'])
            
            # Fade out effect
            alpha = 255
            if field['lifetime'] < 30: # Start fading in the last second
                alpha = int(255 * (field['lifetime'] / 30))
            
            # Draw expanding ring
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_CONTAINMENT, alpha))
                
            # Draw translucent fill
            fill_color = (*self.COLOR_CONTAINMENT, int(alpha * 0.15))
            if radius > 0:
                 self._draw_filled_circle_alpha(self.screen, pos, radius, fill_color)

        # Infected Cells
        for cell in self.infected_cells:
            self._draw_glowing_circle(self.screen, cell['pos'], self.infected_cell_radius, self.COLOR_INFECTED, 0.6)

        # Player
        player_color = self.COLOR_PARTICLE if self.speed_burst_timer > 0 else self.COLOR_PLAYER
        self._draw_glowing_circle(self.screen, self.player_pos, self.player_radius, player_color, 0.7)

    def _render_ui(self):
        # Containment Fields Available
        fields_text = self.font_large.render(f"FIELDS: {self.fields_available}/{self.max_fields}", True, self.COLOR_TEXT)
        self.screen.blit(fields_text, (10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.screen_width / 2, 22))
        self.screen.blit(score_text, score_rect)

        # Proximity Warning
        bar_width = 150
        bar_height = 15
        bar_x = self.screen_width - bar_width - 10
        bar_y = 15
        
        closest_dist = float('inf')
        uncontained_cells_exist = False
        for cell in self.infected_cells:
            is_contained = any(np.linalg.norm(cell['pos'] - f['pos']) < f['radius'] for f in self.containment_fields if f['is_active'])
            if not is_contained:
                uncontained_cells_exist = True
                dist = np.linalg.norm(cell['pos'] - self.vital_organ_pos) - self.vital_organ_radius - self.infected_cell_radius
                closest_dist = min(closest_dist, dist)

        proximity_ratio = 0
        if uncontained_cells_exist:
            max_dist = np.linalg.norm(np.array([0,0]) - self.vital_organ_pos)
            proximity_ratio = 1.0 - np.clip(closest_dist / max_dist, 0, 1)

        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if proximity_ratio > 0:
            fill_width = int(bar_width * proximity_ratio)
            pygame.draw.rect(self.screen, self.COLOR_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        
        prox_text = self.font_small.render("PROXIMITY", True, self.COLOR_TEXT)
        self.screen.blit(prox_text, (bar_x + bar_width + 5, 13))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "infected_cells": len(self.infected_cells),
            "fields_available": self.fields_available
        }
    
    def _create_particles(self, pos, count):
        for _ in range(count):
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-5, 5, 2),
                'vel': self.np_random.uniform(-1, 1, 2) * 1.5,
                'radius': self.np_random.uniform(2, 5),
                'life': life,
                'max_life': life
            })

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_factor=0.5):
        """Draws a circle with a soft glow effect."""
        pos_int = (int(pos[0]), int(pos[1]))
        
        # Glow layers
        for i in range(3, 0, -1):
            glow_radius = int(radius * (1 + i * 0.15 * glow_factor))
            glow_alpha = int(50 * (1 - i / 4))
            glow_color = (*color[:3], glow_alpha)
            self._draw_filled_circle_alpha(surface, pos_int, glow_radius, glow_color)
            
        # Main circle
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)

    def _draw_filled_circle_alpha(self, surface, pos, radius, color):
        """Draws a filled circle with alpha transparency."""
        target_rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius * 2, radius * 2)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage and Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup for manual play
    # We need to unset the dummy driver to see the window
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    pygame.display.set_caption("Phagocyte Containment")
    screen_display = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    while running:
        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated:
            print(f"Episode Finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(1000) # Pause before reset

        clock.tick(30) # Run at 30 FPS
        
    env.close()