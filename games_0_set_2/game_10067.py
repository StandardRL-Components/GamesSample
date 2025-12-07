import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:52:50.610337
# Source Brief: brief_00067.md
# Brief Index: 67
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a sci-fi puzzle game.
    The player must charge energy nodes, which form connections to power a central reactor.
    The goal is to reach 500 energy within 60 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Charge energy nodes to form connections and power a central reactor against the clock."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select an energy node. Hold space to charge the selected node."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_INACTIVE = (50, 80, 150)
    COLOR_CHARGING = (255, 220, 100)
    COLOR_CHARGED = (100, 255, 150)
    COLOR_CONNECTION = (100, 220, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    
    NODE_RADIUS = 25
    REACTOR_RADIUS = 40
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 20)
        
        # --- Game Layout ---
        self.node_positions = [
            (180, 120), (320, 120), (460, 120),
            (180, 220), (320, 220), (460, 220)
        ]
        self.node_adjacencies = {
            0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
            3: [0, 4], 4: [1, 3, 5], 5: [2, 4]
        }
        self.reactor_pos = (320, 340)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.timer = 0
        self.max_steps = self.FPS * 60
        self.reactor_energy = 0.0
        self.game_over = False
        self.win = False
        self.selected_node_idx = 0
        self.nodes = []
        self.connections = set()
        self.particles = []
        self.last_move_time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.timer = self.max_steps
        self.reactor_energy = 0.0
        self.game_over = False
        self.win = False
        self.selected_node_idx = 0
        
        self.nodes = [
            {'charge': 0.0, 'id': i} for i in range(6)
        ]
        self.connections = set()
        self.particles = []
        self.last_move_time = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Update Game Logic ---
        self._update_cursor(movement)
        reward += self._update_charging(space_held)
        reward += self._update_connections()
        self._update_reactor()
        self._update_particles()
        
        self.timer -= 1
        self.steps += 1

        # --- Check Termination ---
        terminated = False
        if self.reactor_energy >= 500:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
        elif self.timer <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100.0
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_cursor(self, movement):
        if movement != 0 and (self.steps - self.last_move_time) > self.FPS / 10:
            self.last_move_time = self.steps
            current = self.selected_node_idx
            if movement == 1: # Up
                self.selected_node_idx = current - 3 if current >= 3 else current
            elif movement == 2: # Down
                self.selected_node_idx = current + 3 if current < 3 else current
            elif movement == 3: # Left
                if current % 3 == 0: self.selected_node_idx = current + 2
                else: self.selected_node_idx = current - 1
            elif movement == 4: # Right
                if current % 3 == 2: self.selected_node_idx = current - 2
                else: self.selected_node_idx = current + 1

    def _update_charging(self, space_held):
        if not space_held:
            return 0.0
            
        node = self.nodes[self.selected_node_idx]
        if node['charge'] < 100.0:
            # Sfx: charging_sound.loop()
            charge_increase = 1.0
            node['charge'] = min(100.0, node['charge'] + charge_increase)
            
            if self.steps % 2 == 0:
                pos = self.node_positions[self.selected_node_idx]
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                self._spawn_particle(pos, vel, self.COLOR_CHARGING, 5, 15)
            
            return 0.1 * charge_increase
        else:
            # Sfx: charged_fully.play()
            return 0.0

    def _update_connections(self):
        new_connections = set()
        for i in range(len(self.nodes)):
            if self.nodes[i]['charge'] >= 100:
                for neighbor_idx in self.node_adjacencies[i]:
                    if self.nodes[neighbor_idx]['charge'] >= 100:
                        connection = tuple(sorted((i, neighbor_idx)))
                        new_connections.add(connection)
        
        newly_formed = new_connections - self.connections
        reward = len(newly_formed) * 1.0
        
        if newly_formed:
            # Sfx: connection_formed.play()
            pass
            
        self.connections = new_connections
        return reward

    def _update_reactor(self):
        energy_gain = len(self.connections) * 0.15
        if energy_gain > 0:
            self.reactor_energy = min(500.0, self.reactor_energy + energy_gain)

    def _update_particles(self):
        if self.steps % 5 == 0:
            for n1_idx, n2_idx in self.connections:
                start_pos = self.node_positions[n1_idx]
                end_pos = self.node_positions[n2_idx]
                self._spawn_particle(start_pos, (0,0), self.COLOR_CONNECTION, 3, 30, end_pos)
        
        alive_particles = []
        for p in self.particles:
            if p['target']:
                dx, dy = p['target'][0] - p['pos'][0], p['target'][1] - p['pos'][1]
                dist = math.hypot(dx, dy)
                if dist < 4: p['life'] = 0
                else: p['pos'] = (p['pos'][0] + dx/dist * 4, p['pos'][1] + dy/dist * 4)
            else:
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            
            p['life'] -= 1
            if p['life'] > 0:
                alive_particles.append(p)
        self.particles = alive_particles

    def _spawn_particle(self, pos, vel, color, radius, life, target=None):
        self.particles.append({
            'pos': list(pos), 'vel': vel, 'color': color,
            'radius': radius, 'life': life, 'max_life': life, 'target': target
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            current_radius = int(p['radius'] * life_ratio)
            if current_radius > 0:
                alpha = int(255 * life_ratio)
                temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p['color'], alpha), (current_radius, current_radius), current_radius)
                self.screen.blit(temp_surf, (int(p['pos'][0]) - current_radius, int(p['pos'][1]) - current_radius))

        for n1_idx, n2_idx in self.connections:
            self._draw_glowing_line(self.node_positions[n1_idx], self.node_positions[n2_idx], self.COLOR_CONNECTION, 2)
            
        reactor_glow = self.reactor_energy / 500.0
        self._draw_glowing_circle(self.reactor_pos, self.REACTOR_RADIUS, self.COLOR_CONNECTION, reactor_glow)
        
        for i, node in enumerate(self.nodes):
            pos = self.node_positions[i]
            charge_ratio = node['charge'] / 100.0
            
            if charge_ratio < 1.0:
                color = self._lerp_color(self.COLOR_INACTIVE, self.COLOR_CHARGING, charge_ratio * 2 if charge_ratio < 0.5 else 1.0)
            else: color = self.COLOR_CHARGED
                
            pygame.draw.circle(self.screen, self.COLOR_INACTIVE, pos, self.NODE_RADIUS + 2, 2)
            
            fill_radius = int(self.NODE_RADIUS * charge_ratio)
            if fill_radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), fill_radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), fill_radius, color)

        cursor_pos = self.node_positions[self.selected_node_idx]
        t = self.steps * 0.2
        alpha = 128 + 127 * math.sin(t)
        radius = self.NODE_RADIUS + 8 + 3 * math.sin(t)
        self._draw_pulsing_ring(cursor_pos, radius, self.COLOR_CURSOR, alpha)

    def _render_ui(self):
        timer_seconds = math.ceil(self.timer / self.FPS)
        timer_text = f"TIME: {max(0, timer_seconds):02d}"
        timer_color = self.COLOR_TIMER_WARN if timer_seconds <= 10 and not self.game_over else self.COLOR_TEXT
        text_surf = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 10))
        
        energy_text = f"REACTOR: {int(self.reactor_energy)} / 500"
        text_surf = self.font_large.render(energy_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 10))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "REACTOR ONLINE" if self.win else "SYSTEM FAILURE"
            color = self.COLOR_CHARGED if self.win else self.COLOR_TIMER_WARN
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.reactor_energy,
            "steps": self.steps,
        }

    # --- Helper & Drawing Methods ---
    def _lerp(self, a, b, t): return a + (b - a) * t
    def _lerp_color(self, c1, c2, t): return tuple(self._lerp(c1[i], c2[i], t) for i in range(3))

    def _draw_glowing_circle(self, pos, radius, color, glow_intensity):
        if glow_intensity <= 0: return
        max_glow_radius = radius * 2.5
        for i in range(5):
            p = i / 4.0
            current_radius = int(self._lerp(radius, max_glow_radius, p * glow_intensity))
            alpha = int(100 * (1 - p) * glow_intensity)
            if alpha > 0 and current_radius > 0:
                temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*color, alpha), (current_radius, current_radius), current_radius)
                self.screen.blit(temp_surf, (int(pos[0]) - current_radius, int(pos[1]) - current_radius))
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

    def _draw_glowing_line(self, pos1, pos2, color, width):
        pygame.draw.aaline(self.screen, (*color, 60), pos1, pos2, blend=1)
        pygame.draw.line(self.screen, color, pos1, pos2, width)

    def _draw_pulsing_ring(self, pos, radius, color, alpha):
        temp_surf = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, int(alpha)), (radius, radius), radius, 2)
        self.screen.blit(temp_surf, (int(pos[0]) - radius, int(pos[1]) - radius))

    def render(self):
        # This method is not strictly required by the new API but can be useful
        # for displaying the game screen during development or for human play.
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It will create a window and let you control the game.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Energy Reactor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space_held, shift_held = 0, 0, 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- RESET ---")
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Key polling for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Optional: auto-reset on termination
            # obs, info = env.reset()
            # total_reward = 0
            
    pygame.quit()