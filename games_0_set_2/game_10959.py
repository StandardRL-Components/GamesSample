import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:22:58.334028
# Source Brief: brief_00959.md
# Brief Index: 959
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Design and build an energy grid by placing components and creating portals. "
        "Optimize your layout to generate the target energy output."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place a component. "
        "Press shift on two different slots to create or remove a portal."
    )
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 2
    GRID_COLS = 5
    GRID_SLOTS = GRID_ROWS * GRID_COLS
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (50, 40, 90)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PORTAL_SELECT = (255, 200, 0)
    COLOR_PORTAL_LINE = (0, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    COMPONENT_COLORS = {
        'generator': (0, 150, 255),
        'amplifier': (0, 255, 150),
        'capacitor': (255, 220, 0)
    }
    
    # Game Parameters
    MAX_STEPS = 1000
    INITIAL_TARGET_ENERGY = 10.0
    TARGET_INCREASE_RATE = 1.1
    
    # Component Physics
    BASE_GENERATOR_OUTPUT = 5.0
    AMPLIFIER_FACTOR = 1.5
    CAPACITOR_THRESHOLD = 20.0
    ENERGY_PROPAGATION_ITERATIONS = 10 # To handle loops and complex paths

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # Game State (initialized in reset)
        self.steps = 0
        self.game_over = False
        self.cursor_pos = 0
        self.grid_slots = []
        self.portals = []
        self.particles = []
        
        self.current_energy_output = 0.0
        self.max_episode_energy = 0.0
        
        self.previous_space_state = 0
        self.previous_shift_state = 0
        self.portal_selection_start = None
        
        self.available_components = ['generator', 'amplifier', 'capacitor']
        self.component_select_idx = 0

        # This persists across resets for difficulty progression
        self.target_energy_output = self.INITIAL_TARGET_ENERGY
        
        self._calculate_grid_positions()
        # self.reset() is called by the wrapper/runner
        
    def _calculate_grid_positions(self):
        self.slot_positions = []
        padding_x = 100
        padding_y = 80
        w = self.SCREEN_WIDTH - 2 * padding_x
        h = self.SCREEN_HEIGHT - 2 * padding_y
        
        for i in range(self.GRID_SLOTS):
            row = i // self.GRID_COLS
            col = i % self.GRID_COLS
            x = padding_x + col * (w / (self.GRID_COLS - 1)) if self.GRID_COLS > 1 else padding_x + w / 2
            y = padding_y + row * (h / (self.GRID_ROWS - 1)) if self.GRID_ROWS > 1 else padding_y + h / 2
            self.slot_positions.append(pygame.Vector2(x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.cursor_pos = 0
        
        self.grid_slots = []
        for i in range(self.GRID_SLOTS):
            self.grid_slots.append({
                'type': 'none',
                'energy_in': 0.0,
                'energy_out': 0.0,
                'charge': 0.0,
            })
            
        self.portals = []
        self.particles = []
        self.current_energy_output = 0.0
        self.max_episode_energy = 0.0
        
        self.previous_space_state = 0
        self.previous_shift_state = 0
        self.portal_selection_start = None
        self.component_select_idx = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.previous_space_state
        shift_pressed = shift_held and not self.previous_shift_state
        self.previous_space_state = space_held
        self.previous_shift_state = shift_held

        previous_energy = self.current_energy_output
        
        self._handle_actions(movement, space_pressed, shift_pressed)
        self._update_game_state()
        
        reward = self._calculate_reward(previous_energy)
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, movement, space_pressed, shift_pressed):
        # --- Cursor Movement ---
        if movement != 0:
            row, col = self.cursor_pos // self.GRID_COLS, self.cursor_pos % self.GRID_COLS
            if movement == 1: row = (row - 1 + self.GRID_ROWS) % self.GRID_ROWS # Up
            elif movement == 2: row = (row + 1) % self.GRID_ROWS # Down
            elif movement == 3: col = (col - 1 + self.GRID_COLS) % self.GRID_COLS # Left
            elif movement == 4: col = (col + 1) % self.GRID_COLS # Right
            self.cursor_pos = row * self.GRID_COLS + col
        
        # --- Place Component ---
        if space_pressed:
            # SFX: place_component.wav
            slot = self.grid_slots[self.cursor_pos]
            slot['type'] = self.available_components[self.component_select_idx]
            slot['charge'] = 0.0 # Reset capacitor charge on placement
            self.component_select_idx = (self.component_select_idx + 1) % len(self.available_components)

        # --- Create/Remove Portal ---
        if shift_pressed:
            if self.portal_selection_start is None:
                self.portal_selection_start = self.cursor_pos
                # SFX: portal_select.wav
            else:
                start_idx, end_idx = self.portal_selection_start, self.cursor_pos
                self.portal_selection_start = None
                
                if start_idx != end_idx:
                    portal = tuple(sorted((start_idx, end_idx)))
                    if portal in self.portals:
                        self.portals.remove(portal)
                        # SFX: portal_close.wav
                    else:
                        self.portals.append(portal)
                        # SFX: portal_open.wav

    def _update_game_state(self):
        self._calculate_energy_flow()
        self._update_particles()

    def _calculate_energy_flow(self):
        # Reset transient energy values
        for slot in self.grid_slots:
            slot['energy_in'] = 0.0
            slot['energy_out'] = 0.0

        # Iteratively propagate energy
        for _ in range(self.ENERGY_PROPAGATION_ITERATIONS):
            prev_out_energies = [s['energy_out'] for s in self.grid_slots]
            
            for i, slot in enumerate(self.grid_slots):
                # 1. Gather input energy
                energy_in = 0.0
                # From left neighbor on the grid
                if i % self.GRID_COLS != 0:
                    energy_in += prev_out_energies[i - 1]
                # From portals
                for p_start, p_end in self.portals:
                    if p_end == i: energy_in += prev_out_energies[p_start]
                    if p_start == i: energy_in += prev_out_energies[p_end]
                
                slot['energy_in'] = energy_in

                # 2. Process component logic
                energy_out = 0.0
                if slot['type'] == 'generator':
                    energy_out = self.BASE_GENERATOR_OUTPUT + energy_in
                elif slot['type'] == 'amplifier':
                    energy_out = energy_in * self.AMPLIFIER_FACTOR
                elif slot['type'] == 'capacitor':
                    slot['charge'] += energy_in
                    if slot['charge'] >= self.CAPACITOR_THRESHOLD:
                        energy_out = slot['charge']
                        slot['charge'] = 0.0
                else: # 'none' or other
                    energy_out = energy_in
                
                slot['energy_out'] = energy_out
        
        # Calculate total output from the rightmost column
        output = 0.0
        for i in range(self.GRID_ROWS):
            output += self.grid_slots[i * self.GRID_COLS + self.GRID_COLS - 1]['energy_out']
        self.current_energy_output = output

    def _update_particles(self):
        # Move and fade existing particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Spawn new particles based on energy flow
        if not hasattr(self, 'np_random'):
            self.reset()
        for i, slot in enumerate(self.grid_slots):
            if slot['energy_out'] > 0.1:
                # Grid flow
                if i % self.GRID_COLS < self.GRID_COLS - 1:
                    start_pos = self.slot_positions[i]
                    end_pos = self.slot_positions[i+1]
                    self._spawn_particles(start_pos, end_pos, slot['energy_out'])
        # Portal flow
        for start_idx, end_idx in self.portals:
            energy = max(self.grid_slots[start_idx]['energy_out'], self.grid_slots[end_idx]['energy_out'])
            if energy > 0.1:
                start_pos = self.slot_positions[start_idx]
                end_pos = self.slot_positions[end_idx]
                self._spawn_particles(start_pos, end_pos, energy, is_portal=True)

    def _spawn_particles(self, start_pos, end_pos, energy, is_portal=False):
        num_particles = min(10, int(math.log(energy + 1)))
        color = self.COLOR_PORTAL_LINE if is_portal else (255, 255, 255)
        
        for _ in range(num_particles):
            if (end_pos - start_pos).length() == 0: continue
            
            direction = (end_pos - start_pos).normalize()
            speed = self.np_random.uniform(2, 4)
            
            self.particles.append({
                'pos': start_pos + direction * self.np_random.uniform(0, 10),
                'vel': direction * speed,
                'life': self.np_random.integers(20, 41),
                'color': color,
                'size': self.np_random.uniform(1, 2.5)
            })

    def _calculate_reward(self, previous_energy):
        reward = 0.0
        
        # Continuous feedback for energy increase
        energy_increase = self.current_energy_output - previous_energy
        if energy_increase > 0:
            reward += energy_increase * 0.1
        
        # Event-based reward for new best
        if self.current_energy_output > self.max_episode_energy:
            reward += 1.0
            self.max_episode_energy = self.current_energy_output
            
        # Goal-oriented reward
        if self.current_energy_output >= self.target_energy_output:
            reward += 100.0
            
        return reward

    def _check_termination(self):
        if self.current_energy_output >= self.target_energy_output:
            self.game_over = True
            self.target_energy_output *= self.TARGET_INCREASE_RATE # Difficulty progression
            # SFX: level_complete.wav
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid lines
        for i in range(self.GRID_SLOTS):
            if i % self.GRID_COLS < self.GRID_COLS - 1:
                pygame.draw.line(self.screen, self.COLOR_GRID, self.slot_positions[i], self.slot_positions[i+1], 2)
            if i < self.GRID_SLOTS - self.GRID_COLS:
                pygame.draw.line(self.screen, self.COLOR_GRID, self.slot_positions[i], self.slot_positions[i+self.GRID_COLS], 2)

        # Render portals
        for start_idx, end_idx in self.portals:
            self._draw_glowing_line(self.slot_positions[start_idx], self.slot_positions[end_idx], self.COLOR_PORTAL_LINE, 2)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'].x - p['size'], p['pos'].y - p['size']))

        # Render components and slots
        for i, slot in enumerate(self.grid_slots):
            pos = self.slot_positions[i]
            # Draw base slot
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 12, self.COLOR_GRID)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_GRID)

            # Draw component
            if slot['type'] != 'none':
                color = self.COMPONENT_COLORS[slot['type']]
                if slot['type'] == 'generator':
                    self._draw_glowing_circle(pos, color, 10)
                    pygame.draw.line(self.screen, self.COLOR_BG, pos - (0, 5), pos + (0, 5), 2)
                    pygame.draw.line(self.screen, self.COLOR_BG, pos - (5, 0), pos + (5, 0), 2)
                elif slot['type'] == 'amplifier':
                    points = [ (pos.x - 8, pos.y - 8), (pos.x + 8, pos.y), (pos.x - 8, pos.y + 8)]
                    self._draw_glowing_polygon(points, color)
                elif slot['type'] == 'capacitor':
                    rect = pygame.Rect(pos.x - 8, pos.y - 8, 16, 16)
                    self._draw_glowing_rect(rect, color)
                    charge_ratio = min(1.0, slot['charge'] / self.CAPACITOR_THRESHOLD)
                    if charge_ratio > 0:
                        charge_rect = pygame.Rect(rect.left, rect.bottom - rect.height * charge_ratio, rect.width, rect.height * charge_ratio)
                        s = pygame.Surface(charge_rect.size, pygame.SRCALPHA)
                        s.fill((255, 255, 255, 100))
                        self.screen.blit(s, charge_rect.topleft)


    def _render_ui(self):
        # Cursor
        cursor_pos = self.slot_positions[self.cursor_pos]
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        cursor_color = self.COLOR_CURSOR
        
        # Portal selection highlight
        if self.portal_selection_start is not None:
            start_pos = self.slot_positions[self.portal_selection_start]
            self._draw_glowing_circle(start_pos, self.COLOR_PORTAL_SELECT, 22, width=2, intensity=pulse)
            cursor_color = self.COLOR_PORTAL_SELECT # Change cursor color during portal selection

        self._draw_glowing_circle(cursor_pos, cursor_color, 20, width=2, intensity=pulse)

        # Text UI
        # Energy Output
        energy_text = f"OUTPUT: {self.current_energy_output:.1f} / {self.target_energy_output:.1f}"
        text_surf = self.font_main.render(energy_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 20))

        # Step counter
        step_text = f"STEP: {self.steps} / {self.MAX_STEPS}"
        text_surf = self.font_small.render(step_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, self.SCREEN_HEIGHT - 30))

        # Next component
        next_comp_type = self.available_components[self.component_select_idx]
        next_comp_color = self.COMPONENT_COLORS[next_comp_type]
        text_surf = self.font_main.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 20))
        pygame.draw.rect(self.screen, next_comp_color, (90, 22, 20, 20))

    def _get_info(self):
        return {
            "score": self.current_energy_output,
            "steps": self.steps,
            "target": self.target_energy_output
        }

    def _draw_glowing_circle(self, pos, color, radius, width=0, intensity=1.0):
        for i in range(3):
            alpha = int(100 * (0.5 ** i) * intensity)
            s = pygame.Surface((radius*2.5, radius*2.5), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (s.get_width()//2, s.get_height()//2), radius + i * 2)
            self.screen.blit(s, (pos.x - s.get_width()//2, pos.y - s.get_height()//2))
        pygame.draw.circle(self.screen, color, pos, radius, width)

    def _draw_glowing_line(self, p1, p2, color, width):
        for i in range(3):
            alpha = int(80 * (0.5 ** i))
            s = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.line(s, (*color, alpha), p1, p2, width + i * 4)
            self.screen.blit(s, (0,0))
        pygame.draw.line(self.screen, color, p1, p2, width)

    def _draw_glowing_polygon(self, points, color):
        for i in range(3):
            alpha = int(100 * (0.5 ** i))
            pygame.gfxdraw.aapolygon(self.screen, points, (*color, alpha))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_glowing_rect(self, rect, color):
        for i in range(3):
            alpha = int(100 * (0.5 ** i))
            glow_rect = rect.inflate(i * 4, i * 4)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for developer convenience and is not called by the test suite.
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            test_obs = self._get_observation()
            assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            # Test reset
            obs, info = self.reset()
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except Exception as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required environment implementation
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Energy Grid Designer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()